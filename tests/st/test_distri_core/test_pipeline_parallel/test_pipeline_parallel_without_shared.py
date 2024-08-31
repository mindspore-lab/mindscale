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
"""Test Pipeline Parallel Without Shared Weight."""
import os
import numpy as np
import pytest
from tests.st.test_distri_core.utils import read_loss_from_log



class TestPipelineParallel:
    """A test class for pipeline parallel without shared weight. """

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.skip(reason="Get golden loss from records")
    def test_generate_pipeline_net_golden_without_shared_weight(self):
        """
        Feature: generate pipeline net without shared weight golden loss
        Description: run pynative mode pipeline net to generate golden loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)
        log_dir = "standalone_without_share_log"
        scripts_cmd = f"{scripts_path} --run_mode standalone_without_share"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8118 " + \
                    f"--log_dir={log_dir} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_pipeline_net_loss_without_shared_weight(self):
        """
        Feature: test pynative pipeline net without shared weight
        Description: run pynative mode pipeline net without shared weight to generate pynative loss
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "1"
        scripts_name = "run_pipeline_net.py"
        device_num = 4
        log_dir = "pp_without_share_log"
        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --run_mode pp_without_share"
        cmd = f"msrun --worker_num={device_num} " + \
                    f"--local_worker_num={device_num} " + \
                    f"--master_port=8119 " + \
                    f"--log_dir={log_dir} " + \
                    f"--join=True " + \
                    f"--cluster_time_out=300 " + \
                    f"{scripts_cmd}"
        print(f"\nrun cmd is:\n{cmd}")
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/{log_dir}/worker_0.log -C 3")
        assert ret == 0, f"msrun failed, please check {log_dir}/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    def test_compare_loss_without_shared_weight(self):
        """
        Feature: test_compare_loss_without_shared_weight
        Description: compare relative error between pipeline loss and golden loss which without shared weight
        Expectation: relative error smaller than 1e-3
        """
        pp_log_path = './pp_without_share_log/worker_3.log'
        pp_loss = read_loss_from_log(pp_log_path)

        pp_loss = np.array(pp_loss, np.float32)
        golden_loss = np.array([6.1333547, 6.078737], np.float32)

        print(f"pp loss without shared weight: {pp_loss}", flush=True)
        print(f"golden loss without shared weight: {golden_loss}", flush=True)
        assert np.allclose(pp_loss, golden_loss, atol=1e-3), "Pipeline parallel loss without shared" \
                                                                "weight accuracy test fail !"
        print("============== Pipeline parallel loss without shared weight accuracy test pass !!! ==============")
