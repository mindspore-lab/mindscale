# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""clip grad and scale grad"""

import mindspore.ops as ops
import mindspore.common.dtype as mstype
import mindspore._checkparam as validator
import mindspore.nn as nn
from mindspore import mint
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.communication import get_group_size
from mindspore.communication.comm_func import all_reduce

from mindformers.experimental.distri_cores.create_comm import get_tp_group, get_tp_rank
from mindformers.experimental.distri_cores.register import ModuleType, ModuleRegistry


def inplace_apply_to_tensor_list(func: callable):
    """Apply a function to a list of tensors in place.

    Args:
        func (callable): The function to apply to each tensor in the list.
    Returns:
        callable: The function that applies the input function to each tensor in the list in place.
    """

    def inplace_apply_func(tensor_list, *args, **kwargs):
        for idx in range(len(tensor_list)):
            tensor_list[idx] = func(tensor_list[idx], *args, **kwargs)

    return inplace_apply_func


@ModuleRegistry.register_decorator(ModuleType.GRAD_PROCESS_FUNC)
class GradClipByValue(nn.Cell):
    """
    Clips the gradients by a specified value inplace.

    Args:
        clip_value (float): The value to clip the gradients.

    Inputs:
        - **grads** (list[Tensor]) - The gradients of parameters, the shape is the same as parameters.
    """
    def __init__(self, clip_value):
        super(GradClipByValue, self).__init__()
        self.clip_value = clip_value
        self.clip_func = inplace_apply_to_tensor_list(ops.clip_by_value)

    def construct(self, grads):
        self.clip_func(grads, -self.clip_value, self.clip_value)


get_square_sum = C.MultitypeFuncGraph("get_square_sum")


@get_square_sum.register("Tensor")
def _get_square_sum(grad):
    norm = mint.sum(mint.square(P.Cast()(grad, mstype.float32)))
    norm = norm.expand_dims(0)
    return norm


@ModuleRegistry.register_decorator(ModuleType.GRAD_PROCESS_FUNC)
class ClipGlobalNorm(nn.Cell):
    """
    Clips the gradients by global norm value.

    Args:
        params (List): Parameters of the network.
        reduce_comm_group: The communication group to reduce norm value.
        clip_value (float): The value to clip the gradients. Default: 1.0.
        norm_type (str): The type of global norm value. Default: 'l2'

    Inputs:
        - **grads** (list[Tensor]) - The gradients of parameters, the shape is the same as parameters.
    """
    def __init__(self, params, reduce_comm_group, clip_value=1.0, norm_type='l2'):
        super(ClipGlobalNorm, self).__init__()
        self.params = params
        self.clip_value = clip_value
        self.hyper_map = C.HyperMap()
        self.norm_type = norm_type
        self.reduce_comm_group = reduce_comm_group
        self.clip_func = inplace_apply_to_tensor_list(self.grad_scale_func)

    def grad_scale_func(self, grad, scale):
        """ function of scaling grads """
        return grad * scale

    def get_grads(self, grads):
        """
        get grads to norm, include weight/bias(not duplicate) and layernorm(duplicate, only pick grad on rank0)
        """
        rank_id = get_tp_rank()
        norm_grads = ()
        for i, param in enumerate(self.params):
            is_duplicate_grad = (
                ("norm" in param.name)
                or ("mlp.projection.bias" in param.name)
                or ("attention.out_proj.bias" in param.name)
            )
            if is_duplicate_grad:
                if rank_id == 0:
                    norm_grads = norm_grads + (grads[i],)
            else:
                norm_grads = norm_grads + (grads[i],)
        return norm_grads

    def get_grad_norm_fp32(self, grads):
        """Compute grad norm."""
        if self.norm_type == "l2":
            square_sum = self.hyper_map(get_square_sum, grads)
            square_reduce_sum = ops.addn(square_sum)
        else:
            raise NotImplementedError("for global norm, l2 norm only support now.")
        if get_group_size(self.reduce_comm_group) > 1:
            square_reduce_sum = all_reduce(square_reduce_sum, "sum", self.reduce_comm_group)
        total_norm = mint.sqrt(square_reduce_sum)
        return total_norm

    def construct(self, grads):
        """clip grad by global norm."""
        norm_grads = self.get_grads(grads)
        total_norm = self.get_grad_norm_fp32(norm_grads)
        clip_coeff = self.clip_value / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            self.clip_func(grads, clip_coeff)
        return total_norm


def get_grad_process_func(training_config, return_instance=True, **kwargs):
    """
    Get the gradient processing function based on the provided training configuration.

    Args:
        training_config (TrainingConfig): The training configuration object.
        return_instance (bool, optional): Whether to return an instance of the gradient processing function.
            Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        Union[Type[GradProcessFunc], GradProcessFunc]: The gradient processing function or its instance.
    """
    grad_process_func_kwargs = training_config.grad_clip_kwargs.copy()
    grad_clip_type = grad_process_func_kwargs.pop("grad_clip_type")
    grad_clip_cls = ModuleRegistry.get_item(module_type=ModuleType.GRAD_PROCESS_FUNC, item_name=grad_clip_type)
    if return_instance:
        grad_process_func_kwargs.update(kwargs)
        grad_process_func_kwargs = ModuleRegistry.get_needed_params_for_init(grad_clip_cls, grad_process_func_kwargs)
        if grad_clip_type == "ClipGlobalNorm":
            if "params" not in grad_process_func_kwargs:
                raise ValueError("params is required for ClipGlobalNorm")
            reduce_comm_group = get_tp_group()
            grad_process_func_kwargs["reduce_comm_group"] = reduce_comm_group
        return grad_clip_cls(**grad_process_func_kwargs)
    return grad_clip_cls


class GradAccumulator:
    '''
    Gradient accumulator.

    Args:
        micro_batch_num (int): Gradient accumulation steps.
        op (str): Operate on the result of gradient accumulation. like sum, mean. Default: "mean".

    Inputs:
        - **grads** (tuple[Tensor]) - The gradients of parameters, the shape is the same as parameters.

    Outputs:
        - Tensor, accumulated gradients, the shape and type is the same asgradients.

    Raises:
        NotImplementedError: If `op` is not mean or sum.

    Examples:
        >>> from mindformers.experimental.distri_cores.grad_handler import GradAccumulator
        >>> micro_batch_num = 2
        >>> accumulator = GradAccumulator(micro_batch_num)
        >>> grad_func = ops.value_and_grad(network, grad_position=0, weights=optimizer.parameters)
        >>> loss, grads = grad_func(input_ids, labels)
        >>> grads = accumulator(grads)
        >>> if grads is not None:
        ...     print("do optimizer")
        ...     optimizer(grads)
    '''
    def __init__(self, micro_batch_num, op="mean"):
        self.counter = 0
        validator.check_non_negative_int(micro_batch_num, "accumulate_step")
        self.accumulate_step = micro_batch_num
        if op not in ["mean", "sum"]:
            raise NotImplementedError(f"{op} is not supported in GradAccumulator yet.")
        self.is_mean_op = op == "mean"
        self.map = ops.HyperMap()
        self.has_init = False
        self.need_clear = False
        self.inner_grads = None

        self.zeroslike = ops.ZerosLike()

    def _init_inner_grads(self, param):
        return self.zeroslike(param)

    def _clear_value(self, inner_grads):
        zeros = self.zeroslike(inner_grads)
        inner_grads.assign_value(zeros)

    def _mean_value(self, inner_grads):
        inner_grads.assign_value(inner_grads / self.accumulate_step)

    def __call__(self, grads):
        if not self.has_init:
            self.inner_grads = self.map(ops.partial(self._init_inner_grads), grads)
            self.has_init = True
        if self.need_clear:
            self.map(ops.partial(self._clear_value), self.inner_grads)
            self.need_clear = False
        self.map(ops.partial(ops.assign_add), self.inner_grads, grads)
        self.counter += 1
        if self.counter % self.accumulate_step == 0:
            if self.is_mean_op:
                self.map(ops.partial(self._mean_value), self.inner_grads)
            self.need_clear = True
            return self.inner_grads
        return None
