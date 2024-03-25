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
""" test image processor """
import os
from pathlib import Path
import unittest
import tempfile

os.environ["OPENMIND_HUB_ENDPOINT"] = "https://modelfoundrysh.test.osinfra.cn"

from mindformers import (
    Blip2Processor, Blip2ImageProcessor, BertTokenizerFast, BertTokenizer,
    GPT2Processor, GPT2Tokenizer, GPT2TokenizerFast
)


TEST_BLIP2_NAME = "blip2_stage1_vit_g"
TEST_GPT2_NAME = "gpt2"

BLIP2_REMOTE_PATH = "mindformersinfra/test_blip2"
GPT2_REMOTE_PATH = "mindformersinfra/test_gpt2"


class ProcessorTest(unittest.TestCase):
    """Processor test case"""
    def test_blip2_load_and_save(self):
        """use Blip2 model to test ProcessorMixin.from_pretraine() and save_pretrained() method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "blip2"
            save_dir = str(Path(tmp_dir) / "save")
            os.environ["CHECKPOINT_DOWNLOAD_FOLDER"] = str(cache_dir)

            # 1. test from_pretrained_experimental
            # load preprocessor.json from repo
            processor_exp = Blip2Processor.from_pretrained(BLIP2_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(processor_exp.image_processor, Blip2ImageProcessor)
            self.assertIsInstance(processor_exp.tokenizer, BertTokenizerFast)
            # 2. test save_pretrained_experimental
            processor_exp.save_pretrained(save_directory=save_dir, save_json=True)
            # load preprocessor.json from local dir
            Blip2Processor.from_pretrained(save_dir)

            # 3. test from_pretrained_origin
            # load blip2_stage1_vit_g.yaml from this project file
            processor_ori = Blip2Processor.from_pretrained(TEST_BLIP2_NAME)
            self.assertIsInstance(processor_ori.image_processor, Blip2ImageProcessor)
            self.assertIsInstance(processor_ori.tokenizer, BertTokenizer)
            # 4. test save_pretrained_origin
            processor_ori.save_pretrained(save_directory=save_dir, save_json=False)
            assert os.path.exists(f"{save_dir}/preprocessor_config.json")
            assert os.path.exists(f"{save_dir}/mindspore_model.yaml")
            # 5. load yaml when exists json file and yaml file at the same time
            # Blip2Processor.from_pretrained(save_dir)

    def test_gpt2_load_and_save(self):
        """use GPT2 model to test ProcessorMixin.from_pretraine() and save_pretrained() method"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "gpt2"
            save_dir = str(Path(tmp_dir) / "save")
            os.environ["CHECKPOINT_DOWNLOAD_FOLDER"] = str(cache_dir)

            # 1. test from_pretrained_experimental
            # load preprocessor.json from repo
            processor_exp = GPT2Processor.from_pretrained(GPT2_REMOTE_PATH, cache_dir=cache_dir)
            self.assertIsInstance(processor_exp.tokenizer, GPT2TokenizerFast)
            # 2. test save_pretrained_experimental
            processor_exp.save_pretrained(save_directory=save_dir, save_json=True)
            # load preprocessor.json from local dir
            GPT2Processor.from_pretrained(save_dir)

            # 3. test from_pretrained_origin
            # load gpt2.yaml from this project file
            processor_ori = GPT2Processor.from_pretrained("gpt2")
            self.assertIsInstance(processor_ori.tokenizer, GPT2Tokenizer)
            # 4. test save_pretrained_origin
            processor_ori.save_pretrained(save_directory=save_dir, save_json=False)
            assert os.path.exists(f"{save_dir}/tokenizer_config.json")
            assert os.path.exists(f"{save_dir}/mindspore_model.yaml")
            # 5. load yaml when exists json file and yaml file at the same time
            # GPT2Processor.from_pretrained(save_dir)
