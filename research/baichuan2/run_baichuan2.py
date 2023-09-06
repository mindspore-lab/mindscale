# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Baichuan2 Train/Finetune/Eval/Predict scripts."""
import os
import sys
import argparse

# pylint: disable=W0611
from mindformers import Trainer, MindFormerConfig
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.tools.utils import check_in_modelarts, set_remote_save_url, str2bool
from mindformers.tools.cloud_adapter import cloud_monitor
from mindformers.core.context import build_context, build_profile_cb

import baichuan2_7b
import baichuan2_13b
from baichuan2_tokenizer import Baichuan2Tokenizer

import mindspore as ms

sys.path.insert(0, os.getcwd().split('research')[0])

def context_init(use_parallel=False, optimizer_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


@cloud_monitor()
def main(task='text_generation',
         config='run_baichuan2_7b.yaml',
         run_mode='train',
         use_parallel=False,
         ckpt=None,
         auto_trans_ckpt=False,
         resume=False,
         train_dataset='',
         eval_dataset='',
         predict_data='',
         max_length=512,
         op=True,
         remote_save_url=None,
         device_id=0):
    """main function."""

    # 适配aicc
    if check_in_modelarts() and remote_save_url:
        print("remote_save_url is %s, the output file will be uploaded to here.", remote_save_url)
        set_remote_save_url(remote_save_url)

    # 环境初始化
    if os.path.exists(config) and config.endswith(('.yaml', '.yml')):
        config = MindFormerConfig(os.path.realpath(config))
        config.use_parallel = use_parallel
        config.device_id = device_id
        build_context(config)
        # define callback and add profile callback
        if config.profile:
            config.profile_cb = build_profile_cb(config)
    else:
        context_init(use_parallel, op, device_id)

    # 定义任务，预先准备好相应数据集
    if run_mode == 'train':
        trainer = Trainer(args=config,
                          task=task,
                          train_dataset=train_dataset)
        trainer.train(train_checkpoint=ckpt, auto_trans_ckpt=auto_trans_ckpt, resume=resume)
    elif run_mode == 'finetune':
        trainer = Trainer(args=config,
                          task=task,
                          train_dataset=train_dataset)
        trainer.finetune(finetune_checkpoint=ckpt, auto_trans_ckpt=auto_trans_ckpt, resume=resume)
    elif run_mode == 'eval':
        trainer = Trainer(args=config,
                          task=task,
                          eval_dataset=eval_dataset)
        trainer.evaluate(eval_checkpoint=ckpt, auto_trans_ckpt=auto_trans_ckpt)
    elif run_mode == 'predict':
        trainer = Trainer(args=config,
                          task=task)
        result = trainer.predict(input_data=predict_data,
                                 predict_checkpoint=ckpt,
                                 auto_trans_ckpt=auto_trans_ckpt,
                                 max_length=int(max_length))
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type.')
    parser.add_argument('--config', default='run_baichuan2_7b.yaml', type=str,
                        help='set task type.')
    parser.add_argument('--run_mode', default='train', type=str,
                        help='set run mode for model.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model.')
    parser.add_argument('--load_checkpoint', default="", type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--auto_trans_ckpt', default=False, type=bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='whether resume training.')
    parser.add_argument('--train_dataset', default='', type=str,
                        help='set train dataset.')
    parser.add_argument('--eval_dataset', default='', type=str,
                        help='set eval dataset.')
    parser.add_argument('--predict_data', default='', type=str,
                        help='input predict data.')
    parser.add_argument('--predict_length', default=512, type=int,
                        help='max length for predict output.')
    parser.add_argument('--optimizer_parallel', default=True, type=str2bool,
                        help='whether use optimizer parallel. Default: None')
    parser.add_argument('--remote_save_url', default="", type=str,
                        help='whether use optimizer parallel. Default: None')
    parser.add_argument('--device_id', default=1, type=int,
                        help='device id set when run on single card. Default: 0')
    args = parser.parse_args()

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         use_parallel=args.use_parallel,
         ckpt=args.load_checkpoint,
         auto_trans_ckpt=args.auto_trans_ckpt,
         resume=args.resume,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data,
         max_length=args.predict_length,
         op=args.optimizer_parallel,
         remote_save_url=args.remote_save_url,
         device_id=args.device_id)
