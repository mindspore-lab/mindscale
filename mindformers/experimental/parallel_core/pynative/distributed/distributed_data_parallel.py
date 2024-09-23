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
""" Distributed data parallel wrapper. """
from contextlib import contextmanager
from mindspore import mint, ops
from mindspore.common import dtype as mstype
from mindformers.experimental.parallel_core.pynative.parallel_state import get_data_parallel_world_size, \
    get_pipeline_model_parallel_rank, get_data_parallel_group, get_data_modulo_expert_parallel_group
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.experimental.parallel_core.pynative.distributed.param_and_grad_buffer import ParamAndGradBuffer


__all__ = ['DistributedDataParallel']


class DistributedDataParallel(Module):
    """
    DistributedDataParallel wrapper. DistributedDataParallel allocates contiguous memory buffer for parameters
    and gradients. It also support gradient back-propagation computation and communication. When enable overlapping,
    parameters and gradients will be break up into bucekts which is the unit to conduct all-reduce/reduce-scatter
    communication among data parallel group.

    Args:
        config (TrainingConfig): The TrainingConfig object containing the training related configurations.
        ddp_config (DistributedDataParallelConfig): The DistributedDataParallelConfig object containing the ddp
            related configurations.
        module (Module): The module to be wrapped with DDP.
        disable_bucketing (bool): Disable bucketing, which means all parameters and gradients will be assigned
            to one bucket. Default: False.

    Returns:
        Model wrapped with DistributedDataParallel.

    Examples:
        >>> from mindformers.experimental.distri_cores.distributed import DistributedDataParallel, \
        >>>     DistributedDataParallelConfig
        >>> network = Model()
        >>> ddp_config = DistributedDataParallelConfig()
        >>> network = DistributedDataParallel(trainig_config, ddp_config, network)
    """
    def __init__(
            self,
            config,
            ddp_config,
            module,
            disable_bucketing=False,
        ):
        super(DistributedDataParallel, self).__init__(auto_prefix=False)
        self.config = config
        self.ddp_config = ddp_config
        self.module = module
        self.param_to_buffer = {}

        if self.ddp_config.bucket_size is None:
            dp_size = get_data_parallel_world_size()
            # bucket_size elem consumes memory: if use fp32(4B), then one bucket ranges from 4M(dp_size=1) to 160M(max)
            self.ddp_config.bucket_size = max(40000000, 1000000 * dp_size)

        self.bucket_size = self.ddp_config.bucket_size
        if get_pipeline_model_parallel_rank() > 0 or disable_bucketing or not self.ddp_config.overlap_grad_reduce:
            self.bucket_size = None

        dense_params = []
        expert_parallel_params = []
        for _, param in self.module.parameters_and_names():
            if not param.requires_grad:
                continue
            param.grad = None
            param.main_grad = None

            param.grad_accumulated = False

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        if config.calculate_per_token_loss:
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = 1.0
            else:
                data_parallel_world_size = get_data_parallel_world_size()
                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # allocate buffer for common params and expert params
        self.buffers = self.allocate_buffers_for_parameters(
            dense_params,
            group=get_data_parallel_group(with_context_parallel=True),
            gradient_scaling_factor=gradient_scaling_factor,
        )
        self.expert_parallel_buffers = self.allocate_buffers_for_parameters(
            expert_parallel_params,
            group=get_data_modulo_expert_parallel_group(),
            gradient_scaling_factor=expert_gradient_scaling_factor,
        )

        # register hook for bucket grad reduce
        self.register_hook_for_params()

    def allocate_buffers_for_parameters(self, input_params, group, gradient_scaling_factor):
        """ allocate buffers for parameters in different dtype group. """
        param_and_grad_dtype_to_params = {}
        # group all params by parameter's data type and their gradient's data type.
        for param in input_params:
            param_dtype = param.dtype
            grad_dtype = mstype.float32 if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            if (param_dtype, grad_dtype) not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = []
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)].append(param)

        buffers = []
        # allocate buffer for each group separately
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                ParamAndGradBuffer(
                    ddp_config=self.ddp_config,
                    param_dtype=param_dtype,
                    grad_dtype=grad_dtype,
                    params=params,
                    data_parallel_group=group,
                    bucket_size=self.bucket_size,
                    param_to_name=None,
                    gradient_scaling_factor=gradient_scaling_factor,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]

        return buffers

    def issue_grad_reduce(self):
        """ issue grad reduce for each buffer. """
        for buffer in self.buffers + self.expert_paralle_buffers:
            buffer.issue_grad_reduce()

    def final_grad_reduce(self):
        """ finalize grad reduce for each buffer. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.final_grad_reduce()

    def register_hook_for_params(self):
        """ register backward hook for each params. """
        for param in self.module.get_parameters():
            if param.requires_grad:
                param.register_hook(self._make_param_hook(param, self.param_to_buffer))

    def set_input_tensor(self, input_tensor):
        """ set input tensor for model"""
        self.module.set_input_tensor(input_tensor)

    def construct(self, *inputs, **inputs_dict):
        """ construct for DistributedDataParallel. """
        output = self.module(*inputs, **inputs_dict)
        return output

    def zero_grad_buffer(self):
        """ reset buffers for the next train iteration. """
        for param in self.module.get_parameters():
            if param.requires_grad:
                param.grad_accumulated = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()

    def enable_sync(self, enable):
        """ enable grad buffer sync or not. """
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.sync_enabled = enable

    @contextmanager
    def no_sync(self):
        """ context manager helper function. """
        self.enable_sync(False)
        try:
            yield
        finally:
            self.enable_sync(True)

    def _make_param_hook(
            self,
            param,
            param_to_buffer,
        ):
        """ make closure function as the param hook. """
        def param_hook(grad):
            buffer = param_to_buffer[param]
            if not param.grad_accumulated:
                param.main_grad.copy_(mint.add(param.main_grad, grad.astype(buffer.grad_dtype)))
            if self.ddp_config.overlap_grad_reduce:
                buffer.register_grad_ready(param)
            if param.grad is None:
                return ops.Tensor(0, param.dtype)
            return param.grad

        return param_hook
