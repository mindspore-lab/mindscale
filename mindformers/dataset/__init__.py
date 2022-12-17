# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindFormers Dataset."""
from .dataloader import *
from .mask import *
from .transforms import *
from .sampler import *
from .mim_dataset import MIMDataset
from .img_cls_dataset import ImageCLSDataset
from .contrastive_language_image_pretrain_dataset import ContrastiveLanguageImagePretrainDataset
from .build_dataset import build_dataset
from .base_dataset import BaseDataset
from .bert_pretrain_dataset import BertPretrainDataset
from .utils import check_dataset_config


__all__ = ['MIMDataset', 'ImageCLSDataset', 'build_dataset', 'BaseDataset', 'check_dataset_config',
           'BertPretrainDataset', 'ContrastiveLanguageImagePretrainDataset']
__all__.extend(dataloader.__all__)
__all__.extend(mask.__all__)
__all__.extend(transforms.__all__)
__all__.extend(sampler.__all__)
