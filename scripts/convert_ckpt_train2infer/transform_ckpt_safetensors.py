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
"""transform ckpt"""
import argparse
import time

import mindspore as ms


def one_to_n(src_ckpt_dir, dst_ckpt_strategy, dst_ckpt_dir):
    """Convert ckpt to safetensor to distribute weights"""
    # 1.train的strategy合并
    merge_strategy_start_time = time.time()
    merge_dst_strategy_file = dst_ckpt_strategy + "_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys(src_strategy_dirs=dst_ckpt_strategy, dst_strategy_file=merge_dst_strategy_file)
    merge_strategy_end_time = time.time()

    # 2. ckpt to safetensors
    ckpt_to_safetensors_start_time = time.time()
    ms.ckpt_to_safetensors(file_path=src_ckpt_dir, save_path=src_ckpt_dir + "_safetensors")
    ckpt_to_safetensors_end_time = time.time()

    # 3. 切分
    transform_checkpoints_start_time = time.time()
    ms.transform_checkpoints(src_checkpoints_dir=src_ckpt_dir + "_safetensors", dst_checkpoints_dir=dst_ckpt_dir,
                             ckpt_prefix="checkpoint_",
                             src_strategy_file=None, dst_strategy_file=merge_dst_strategy_file,
                             process_num=2, output_format="ckpt")
    transform_checkpoints_end_time = time.time()

    # # 3.权重在线罗盘切分
    # load_distributed_checkpoint_start_time = time.time()
    # dst_safetensors_path = dst_ckpt_dir + "_safetensors"
    # ms.load_distributed_checkpoint(network=None, predict_strategy=merge_dst_strategy_file, format='safetensors',
    #                                unified_safetensors_dir=src_ckpt_dir + "_safetensors",
    #                                dst_safetensors_dir=dst_safetensors_path)
    # load_distributed_checkpoint_end_time = time.time()
    #
    # # 4.safetensors转ckpt
    # safetensors_to_ckpt_start_time = time.time()
    # ms.safetensors_to_ckpt(dst_safetensors_path, dst_ckpt_dir, processes_num=64)
    # safetensors_to_ckpt_end_time = time.time()

    print(f"merge_strategy time: {merge_strategy_end_time - merge_strategy_start_time}, \
     ckpt_to_safetensors time: {ckpt_to_safetensors_end_time - ckpt_to_safetensors_start_time}, \
     transform_checkpoints time: {transform_checkpoints_end_time - transform_checkpoints_start_time} ")
    # load_distributed_checkpoint time: {load_distributed_checkpoint_end_time - load_distributed_checkpoint_start_time}, \
    # safetensors_to_ckpt time: {safetensors_to_ckpt_end_time - safetensors_to_ckpt_start_time} ")


def m_to_n(src_ckpt_strategy, src_ckpt_dir, dst_ckpt_strategy, dst_ckpt_dir):
    """Convert ckpt to safetensor to distribute weights"""
    # 1.train的strategy合并
    merge_strategy_start_time = time.time()
    merge_src_strategy_file = src_ckpt_strategy + "_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys(src_strategy_dirs=src_ckpt_strategy, dst_strategy_file=merge_src_strategy_file)

    merge_dst_strategy_file = dst_ckpt_strategy + "_merge/merged_strategy.ckpt"
    ms.merge_pipeline_strategys(src_strategy_dirs=dst_ckpt_strategy, dst_strategy_file=merge_dst_strategy_file)
    merge_strategy_end_time = time.time()

    # 2. ckpt to safetensors
    ckpt_to_safetensors_start_time = time.time()
    ms.ckpt_to_safetensors(file_path=src_ckpt_dir, save_path=src_ckpt_dir + "_safetensors")
    ckpt_to_safetensors_end_time = time.time()

    # 3.离线合并safetensors
    unified_safetensors_start_time = time.time()
    ms.unified_safetensors(src_dir=src_ckpt_dir + "_safetensors", src_strategy_file=merge_src_strategy_file,
                           dst_dir=src_ckpt_dir + "_merge_safetensors")
    unified_safetensors_end_time = time.time()

    # 4.权重在线罗盘切分
    load_distributed_checkpoint_start_time = time.time()
    dst_safetensors_path = dst_ckpt_dir + "_safetensors"
    ms.load_distributed_checkpoint(network=None, predict_strategy=merge_dst_strategy_file, format='safetensors',
                                   unified_safetensors_dir=src_ckpt_dir + "_merge_safetensors",
                                   dst_safetensors_dir=dst_safetensors_path)
    load_distributed_checkpoint_end_time = time.time()

    # 5.safetensors转ckpt
    safetensors_to_ckpt_start_time = time.time()
    ms.safetensors_to_ckpt(dst_safetensors_path, dst_ckpt_dir, processes_num=64)
    safetensors_to_ckpt_end_time = time.time()

    print(f"merge_strategy time: {merge_strategy_end_time - merge_strategy_start_time}, \
     ckpt_to_safetensors time: {ckpt_to_safetensors_end_time - ckpt_to_safetensors_start_time}, \
     unified_safetensors time: {unified_safetensors_end_time - unified_safetensors_start_time}, \
     load_distributed_checkpoint time: {load_distributed_checkpoint_end_time - load_distributed_checkpoint_start_time}, \
     safetensors_to_ckpt time: {safetensors_to_ckpt_end_time - safetensors_to_ckpt_start_time} ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_ckpt_strategy',
                        default="",
                        help='path of src ckpt strategy')
    parser.add_argument('--dst_ckpt_strategy',
                        default="",
                        help='path of dst ckpt strategy')
    parser.add_argument('--src_ckpt_dir',
                        default="",
                        type=str,
                        help='path of src ckpt')
    parser.add_argument('--dst_ckpt_dir',
                        default="",
                        type=str,
                        help='path where to save dst ckpt')
    args = parser.parse_args()

    print("......Start transform......")
    if args.src_ckpt_strategy:
        m_to_n(args.src_ckpt_strategy, args.src_ckpt_dir, args.dst_ckpt_strategy, args.dst_ckpt_dir)
    else:
        one_to_n(args.src_ckpt_dir, args.dst_ckpt_strategy, args.dst_ckpt_dir)
    print("......Transform succeed!......")
