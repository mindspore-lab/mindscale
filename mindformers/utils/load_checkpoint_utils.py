#  Copyright 2024 Huawei Technologies Co., Ltd
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ============================================================================
"""utils of load checkpoint file"""
import os
import shutil
import time
from enum import Enum
from glob import glob
from multiprocessing import Process

import mindspore as ms
from mindspore import context
from mindspore.communication.comm_func import barrier

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.logger import logger
from mindformers.tools.utils import is_main_rank, get_epoch_and_step_from_ckpt_name, get_real_rank
from mindformers.utils import convert_hf_safetensors_multiprocess, check_safetensors_key


class CkptFormat(Enum):
    """
    Enum class for MindFormers support checkpoints formats.
    """

    CKPT = 'ckpt'
    SAFETENSORS = 'safetensors'

    @classmethod
    def support_type(cls):
        return [member.value for member in cls]


class CheckpointFileMode(Enum):
    """
    Enum class for MindFormers load checkpoint file cases.
    """
    SINGLE_CHECKPOINT_FILE = 'single_checkpoint_file'
    MULTI_CHECKPOINT_FILE = 'multi_checkpoint_file'
    MULTI_CHECKPOINT_FILE_WITH_RANK_ID = 'multi_checkpoint_file_with_rank_id'


def _check_checkpoint_path(path):
    """check checkpoint path."""
    if not isinstance(path, str) or isinstance(path, os.PathLike):
        raise ValueError(f"config.load_checkpoint must be a str, but got {path} as type {type(path)}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.load_checkpoint {path} does not exist.")

    if path[-1] == '/':  # remove last '/' in path
        return path[:-1]
    return path


def _get_checkpoint_mode(config):
    """get checkpoint place mode."""
    checkpoint_path = config.load_checkpoint

    if os.path.isfile(checkpoint_path):
        # check if checkpoint_path upper folder is rank_x/
        upper_dir_name = os.path.basename(os.path.dirname(checkpoint_path))
        if upper_dir_name.startswith('rank_'):
            return CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value
        return CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value

    # check path is dir
    if not os.path.isdir(checkpoint_path):
        raise ValueError("Provided path is neither a file nor a directory.")

    dir_files = os.listdir(checkpoint_path)
    if any(folder_name.startswith('rank_') for folder_name in dir_files):
        return CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value

    if any(file_name.endswith(config.load_ckpt_format) for file_name in dir_files):
        return CheckpointFileMode.MULTI_CHECKPOINT_FILE.value

    raise ValueError("not support mode: no valid checkpoint files found")


def _get_src_strategy(config):
    """search and get strategy file path from load_checkpoint directory."""
    if os.path.isfile(config.load_checkpoint):
        upper_dir = '/'.join(config.load_checkpoint.split('/')[:-3])
    else:
        upper_dir = os.path.dirname(config.load_checkpoint)

    input_src_strategy = config.get('src_strategy_path_or_dir')
    if os.path.exists(os.path.join(upper_dir, 'strategy')):
        src_strategy_path = os.path.join(upper_dir, 'strategy')
    elif input_src_strategy and os.path.isdir(input_src_strategy):
        src_strategy_path = input_src_strategy
    else:
        raise ValueError("when use checkpoint after train/finetune, src_strategy_path_or_dir should be set "
                         "as a folder contained strategy ckpt files.")
    logger.info(f"load source strategy from {src_strategy_path}.")
    return src_strategy_path


def _is_distributed_checkpoint(checkpoint_file, ckpt_format='safetensors'):
    """check if checkpoint_file is a distributed checkpoint."""
    is_distributed = True
    file_suffix = None
    try:
        epoch, step = get_epoch_and_step_from_ckpt_name(checkpoint_file, ckpt_format)
        is_distributed = False
        file_suffix = f"{epoch}_{step}"
    except ValueError as e:
        logger.info(f"Get epoch and step in {checkpoint_file} failed, check if it's "
                    f"distributed checkpoint and ignore error {e}")
    except Exception as e:
        raise ValueError(f"get_epoch_and_step_from_ckpt_name from {checkpoint_file} failed.") from e
    return is_distributed, file_suffix


def _get_src_file_suffix(config):
    """get file_suffix from config.load_checkpoint."""
    if config.resume_training:
        epoch, step = get_epoch_and_step_from_ckpt_name(config.resume_training, config.load_ckpt_format)
        logger.info(f"Load resume checkpoint from {config.load_checkpoint}, epoch: {epoch}, step: {step}.")
        file_suffix = f"{epoch}_{step}"
        return config.load_checkpoint, file_suffix

    if os.path.isfile(config.load_checkpoint):
        # only support path format: path/rank_x/prefix-{epoch}_{step}.{config.load_ckpt_format}
        file_name = os.path.basename(config.load_checkpoint)
        epoch, step = get_epoch_and_step_from_ckpt_name(file_name, config.load_ckpt_format)
        checkpoint_dir = '/'.join(config.load_checkpoint.split('/')[:-2])
        return checkpoint_dir, f"{epoch}_{step}"

    # config.load_checkpoint is folder
    rank_id = get_real_rank()
    rank_path = f"{config.load_checkpoint}/rank_{rank_id}"
    if not os.path.exists(rank_path):
        raise FileNotFoundError(f"{rank_path} not found.")

    last_checkpoint = get_last_checkpoint(rank_path, config.load_ckpt_format)
    is_distributed, file_suffix = _is_distributed_checkpoint(
        last_checkpoint, config.load_ckpt_format)
    logger.info(f"Last checkpoint in {rank_path}: {last_checkpoint}, is_distributed: {is_distributed}, "
                f"file_suffix: {file_suffix}")
    return config.load_checkpoint, file_suffix


def prepare_strategy_unified_path(config, strategy_path):
    """prepare save path of merged strategy and unified safetensors."""
    # prepare merged strategy directory
    merged_strategy = os.path.join(config.output_dir, 'merged_strategy')
    os.makedirs(merged_strategy, exist_ok=True)

    # set src_strategy_path
    src_strategy = _get_src_strategy(config)
    dst_strategy = os.path.join(merged_strategy, 'src_strategy.ckpt')
    src_strategy_path = (src_strategy, dst_strategy)

    # set dst_strategy_path
    dst_strategy_path = (
        os.path.dirname(strategy_path),
        os.path.join(merged_strategy, 'dst_strategy.ckpt')
    )
    unified_path = os.path.join(config.output_dir, 'unified_checkpoint/')
    return src_strategy_path, dst_strategy_path, unified_path


def load_checkpoint_with_safetensors(config, model, network, input_data, do_eval=False, do_predict=False):
    """load different format checkpoint interface."""
    logger.info(f"......Start load checkpoint from {config.load_ckpt_format}......")
    config.load_checkpoint = _check_checkpoint_path(config.load_checkpoint)
    load_checkpoint = config.load_checkpoint
    logger.info(f"Load checkpoint from {config.load_checkpoint}.")

    pet_config = config.model.model_config.get("pet_config")
    if pet_config and pet_config.pet_type == "slora" and network.lora_list:
        raise ValueError(f"slora only support .ckpt file, {config.load_ckpt_format} file will be compatible soon.")

    # reduce compile time in prediction
    if do_eval or do_predict:
        logger.info("Set network.set_train=False, reduce compile time in prediction.")
        network.set_train(False)

    if config.use_parallel:
        logger.info(f"......Start build model in parallel mode......")
        build_model(config, model, input_data, do_eval=do_eval, do_predict=do_predict)
        barrier()

    ckpt_file_mode = _get_checkpoint_mode(config)
    load_checkpoint_files = []
    strategy_path = ms.get_auto_parallel_context('strategy_ckpt_save_file')
    #depend on ms, support soon
    if ckpt_file_mode == CheckpointFileMode.SINGLE_CHECKPOINT_FILE.value:
        logger.info(f"......Use single checkpoint file mode......")
        raise ValueError(f"single safetensors file is not supported now: {config.load_checkpoint}.")
    if ckpt_file_mode == CheckpointFileMode.MULTI_CHECKPOINT_FILE.value:
        logger.info(f"......Use multi checkpoint file mode......")
        load_checkpoint_files = glob(
            os.path.join(load_checkpoint, f"*.{config.load_ckpt_format}"))
        load_checkpoint_files.sort()
    elif ckpt_file_mode == CheckpointFileMode.MULTI_CHECKPOINT_FILE_WITH_RANK_ID.value:
        logger.info(f"......Use multi checkpoint file with rank id mode......")
        src_strategy_path, dst_strategy_path, unified_path = prepare_strategy_unified_path(config, strategy_path)
        load_checkpoint, file_suffix = _get_src_file_suffix(config)
        merge_and_unified(load_checkpoint,
                          src_strategy_path,
                          dst_strategy_path,
                          unified_path,
                          use_parallel=config.use_parallel,
                          file_suffix=file_suffix,
                          remove_redundancy=config.get('remove_redundancy', False))
        load_checkpoint = unified_path
        load_checkpoint_files = glob(
            os.path.join(load_checkpoint, f"*.{config.load_ckpt_format}"))
        load_checkpoint_files.sort()
        strategy_path = dst_strategy_path[1]

        # use resume_training in train/finetune mode
        if config.resume_training:
            # pylint: disable=W0212
            network = model._train_network

    # only execute qkv concat check on the main rank in predict mode
    if do_predict and is_main_rank(ignore_check_modelarts=True):
        qkv_concat_config = config.model.model_config.get("qkv_concat", False)
        validate_qkv_concat(network, qkv_concat_config, load_checkpoint)
    if config.use_parallel:
        barrier()

    enable_stand_alone = (config.parallel.parallel_mode == 'STAND_ALONE')
    if config.use_parallel and enable_stand_alone:
        from mindformers.experimental.infer.core.utils import generate_state_dict
        from mindformers.experimental.parallel_core.pynative.utils import save_strategy_file
        from mindformers.tools.utils import get_output_root_path
        strategy_ckpt_save_dir = os.path.join(get_output_root_path(), "strategy")
        strategy_path = os.path.join(strategy_ckpt_save_dir, "ckpt_strategy.ckpt")
        if is_main_rank(ignore_check_modelarts=True):
            if os.path.exists(strategy_ckpt_save_dir):
                shutil.rmtree(strategy_ckpt_save_dir)
                logger.info(f"Existed strategy directory {strategy_ckpt_save_dir} has been deleted.")
            os.makedirs(strategy_ckpt_save_dir, exist_ok=True)
            shard_state_dict = generate_state_dict(network)
            save_strategy_file(shard_state_dict, strategy_path)
            logger.info(f"Strategy file for stand alone mode has been saved in {strategy_path}.")
        barrier()
    load_safetensors_checkpoint(config, load_checkpoint_files, network, strategy_path, load_checkpoint)


def merge_and_unified(src_checkpoint, src_strategy_path, dst_strategy_path, unified_path, use_parallel=False,
                      file_suffix=None, remove_redundancy=False):
    """merge strategy and unified safetensors."""
    logger.info("start merge strategy and unified safetensors.")
    if is_main_rank(ignore_check_modelarts=True):
        # merge src strategy
        ms.merge_pipeline_strategys(
            src_strategy_dirs=src_strategy_path[0],
            dst_strategy_file=src_strategy_path[1])
        # combine checkpoints
        logger.info(f"unified safetensors with file_suffix:{file_suffix}, remove_redundancy: {remove_redundancy}")
        logger.info(f"unified safetensors with save path:{unified_path}")
        ms.unified_safetensors(
            src_dir=src_checkpoint,
            src_strategy_file=src_strategy_path[1],
            dst_dir=unified_path,
            file_suffix=file_suffix,
            merge_with_redundancy=not remove_redundancy
        )
        if use_parallel:
            # merge dst strategy
            logger.info("merge dst strategy in parallel mode.")
            ms.merge_pipeline_strategys(
                src_strategy_dirs=dst_strategy_path[0],
                dst_strategy_file=dst_strategy_path[1])
    if use_parallel:
        barrier()
    logger.info("merge strategy and unified safetensors finished.")


def load_safetensors_checkpoint(config, load_checkpoint_files, network, strategy_path, load_ckpt_path):
    """load checkpoint into net."""
    if config.use_parallel:
        logger.info("......Start load distributed checkpoint to model......")
        ms.load_distributed_checkpoint(
            network=network,
            predict_strategy=strategy_path,
            unified_safetensors_dir=load_ckpt_path,
            format=config.load_ckpt_format
        )
    else:
        logger.info("......Start load checkpoint to model......")
        params_dict = dict()
        for checkpoint_file in load_checkpoint_files:
            params_dict.update(ms.load_checkpoint(
                ckpt_file_name=checkpoint_file,
                format=config.load_ckpt_format
            ))
        ms.load_param_into_net(network, params_dict)


def process_hf_checkpoint(model, output_dir=None, load_checkpoint=None):
    """process huggingface checkpoint."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = './output'
        logger.warning(f'Output directory is set to ./output, '
                       f'due to the output_dir {output_dir} does not exist.')
    converted_dir = os.path.join(output_dir, './ms_safetensors')
    if is_main_rank(ignore_check_modelarts=True):
        p = Process(target=convert_hf_safetensors_multiprocess,
                    args=[load_checkpoint, converted_dir, model, model.config.qkv_concat])
        p.start()
        p.join()

    return converted_dir


def build_model(config, model, dataset, do_eval=False, do_predict=False):
    """build model and generate strategy file."""
    parallel_mode = context.get_auto_parallel_context('parallel_mode')
    if parallel_mode not in ('semi_auto_parallel', 'auto_parallel', 'hybrid_parallel'):
        return

    if not config.runner_config.sink_mode:
        raise ValueError("When distributed loads are sliced weights, sink_mode must be set True.")
    if do_predict or do_eval:
        model.infer_predict_layout(*dataset)
    else:
        build_time_start = time.time()
        model.build(train_dataset=dataset, epoch=config.runner_config.epochs,
                    sink_size=config.runner_config.sink_size)
        build_time_end = time.time()
        logger.info("Time spent building the model: %.2fs", build_time_end - build_time_start)


def get_last_checkpoint(checkpoint_dir, ckpt_format='ckpt'):
    """get last checkpoint for resuming or finetune."""
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(
            f"{checkpoint_dir} is not a real directory,"
            f"When distributed loads are sliced weights,"
            f"load_checkpoint should be a checkpoint directory containing the directory of rank_{{0-*}},"
            f"The directory structure is as follows: **checkpoint_root_dir/rank_{{0-*}}/**.{ckpt_format}")
    output_checkpoint_path = [
        checkpoint for checkpoint in os.listdir(checkpoint_dir)
        if checkpoint.endswith(f'.{ckpt_format}')
    ]
    if not output_checkpoint_path:
        return None
    output_checkpoint_path = sorted(output_checkpoint_path,
                                    key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
    return os.path.join(checkpoint_dir, output_checkpoint_path[-1])


def validate_qkv_concat(model_cls_or_instance, qkv_concat_config, load_checkpoint):
    """
    Check whether qkv_concat configuration and qkv concat weight convert are the same.
    Currently only safetensors format is supported.
    """
    # check the type of model_cls_or_instance
    if not (
            isinstance(model_cls_or_instance, PreTrainedModel) or
            (isinstance(model_cls_or_instance, type) and issubclass(model_cls_or_instance, PreTrainedModel))
    ):
        logger.warning(f"Cur model_cls_or_instance: {model_cls_or_instance} is not "
                       f"a subclass or an instance of PreTrainedModel, "
                       f"will not execute qkv concat check.")
        return

    concat_key_list = model_cls_or_instance.obtain_qkv_ffn_concat_keys()
    if concat_key_list is None:
        return

    logger.info(".........Starting qkv concat check.........")
    is_qkv_concat = True
    for concat_key in concat_key_list:
        is_qkv_concat = check_safetensors_key(load_checkpoint, concat_key) and is_qkv_concat
        if not is_qkv_concat:
            break

    if is_qkv_concat and not qkv_concat_config:
        raise ValueError("The qkv concat check failed! The qkv in the model weights has been concatenated,"
                         " but qkv_concat is set to false.")
    if not is_qkv_concat and qkv_concat_config:
        raise ValueError("The qkv concat check failed! The qkv in the model weights has been not concatenated,"
                         " but qkv_concat is set to true.")
    if is_qkv_concat and qkv_concat_config:
        logger.info("The qkv concat check succeed! The qkv in the model weights has been concatenated and "
                    "qkv_concat is set to true.")
    if not is_qkv_concat and not qkv_concat_config:
        logger.info("The qkv concat check succeed! The qkv in the model weights has been not concatenated and "
                    "qkv_concat is set to false.")
