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
"""Token classification Dataset."""
from typing import Optional, Union, Callable

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.version_control import get_dataset_map

from .dataloader import build_dataset_loader
from ..models.build_tokenizer import build_tokenizer
from .transforms import build_transforms
from .sampler import build_sampler
from .base_dataset import BaseDataset


@MindFormerRegister.register(MindFormerModuleType.DATASET)
class TokenClassificationDataset(BaseDataset):
    """
    Token classification Dataset.

    Args:
        dataset_config (Optional[dict]):
            Config for dataset.
        data_loader (Union[dict, Callable]):
            Config for data loader or a data loader object.
        tokenizer (Union[dict, list]):
            Tokenizer configuration or object.
        text_transforms (Union[dict, list]):
            Configurations or objects of one or more transformers of text.
        label_transforms (Union[dict, list]):
            Configurations or objects of one or more transformers of label.
        sampler (Union[dict, list]):
            Sampler configuration or object.
        input_columns (list):
            Column name before the map function.
        output_columns (list):
            Column name after the map function.
        batch_size (int):
            Size of each batch. Default: 8.
        drop_remainder (bool):
            Whether to discard the last batch when the number of data items contained
            in the last batch is smaller than batch_size. Default: True.
        num_parallel_workers (int):
            Specifies the number of concurrent processes or threads for map operations
            to accelerate processing. Default: 8.
        python_multiprocessing (bool):
            Enabling the Python Multi-Process Mode to Accelerate Map Operations. Default: False.
        repeat (int):
            Number of times this dataset is repeated. Default: 1.
        seed (int):
            Random seed number. Default: 0.
        prefetch_size (int):
            Buffer queue size of each data processing operation in the pipeline. Default: 1.
        numa_enable (bool):
            Indicates whether to use the NUMA binding function. Default: False.
        auto_tune (bool):
            Indicates whether to enable automatic optimization of data processing parameters. Default: False.
        autotune_per_step (int):
            Specifies the interval for adjusting the configuration step of automatic data acceleration. Default: 10.
        filepath_prefix (str):
            Path for saving optimized parameter configurations. Default: './autotune'.
        profile (bool):
            Whether to enable data collection. Default: False.

    Returns:
        A dataset for TokenClassificationDataset.

    Examples:
        >>> # 1) Create an instance using a MindFormerConfig.
        >>> from mindformers.tools.register import MindFormerConfig
        >>> from mindformers import MindFormerBook
        >>> from mindformers.dataset import TokenClassificationDataset
        >>> from mindformers.dataset import check_dataset_config
        >>> config_dict_list = MindFormerBook.get_trainer_support_task_list()
        >>> config_path = config_dict_list['token_classification']['tokcls_bert_base_chinese']
        >>> # Initialize a MindFormerConfig instance with a specific config file of yaml.
        >>> config = MindFormerConfig(config_path)
        >>> config.train_dataset.data_loader.dataset_dir = "The required task dataset path"
        >>> check_dataset_config(config)
        >>> # use class to build dataset
        >>> dataset_from_class = TokenClassificationDataset(config.train_dataset_task.dataset_config)
        >>>
        >>> # 2) Creating an instance using other parameters.
        >>> from mindformers import AutoTokenizer
        >>> from mindformers.dataset import TokenClassificationDataset, CLUENERDataLoader
        >>> from mindformers.dataset import TokenizeWithLabel, LabelPadding
        >>> tokenizer = AutoTokenizer.from_pretrained('tokcls_bert_base_chinese_cluener')
        >>> data_loader = CLUENERDataLoader(dataset_dir="The required task dataset path",
        ...                                 stage='train', column_names=['text', 'label_id'])
        >>> text_transforms = TokenizeWithLabel(max_length=128, padding='max_length', tokenizer=tokenizer)
        >>> label_transforms = LabelPadding(max_length=128, padding_value=0)
        >>> dataset_from_param = TokenClassificationDataset(data_loader=data_loader, text_transforms=text_transforms,
        ...                                                 label_transforms=label_transforms, tokenizer=tokenizer,
        ...                                                 input_columns=['text', 'label_id'],
        ...                                                 output_columns=['input_ids', 'token_type_ids',
        ...                                                                 'attention_mask', 'label_id'])
    """

    # pylint: disable=W0613
    def __new__(cls,
                dataset_config: Optional[dict] = None,
                data_loader: Union[dict, Callable] = None,
                tokenizer: Union[dict, Callable] = None,
                text_transforms: Union[dict, list] = None,
                label_transforms: Union[dict, list] = None,
                sampler: Union[dict, Callable] = None,
                input_columns: list = None,
                output_columns: list = None,
                batch_size: int = 8,
                drop_remainder: bool = True,
                num_parallel_workers: int = 8,
                python_multiprocessing: bool = False,
                repeat: int = 1,
                seed: int = 0,
                prefetch_size: int = 1,
                numa_enable: bool = False,
                auto_tune: bool = False,
                filepath_prefix: str = './autotune',
                autotune_per_step: int = 10,
                profile: bool = False,
                **kwargs):
        """new method"""
        logger.info("Now Create Token classification Dataset.")
        dataset_config = cls.check_dataset_config(dataset_config, locals())
        cls.init_dataset_config(dataset_config)
        rank_id, device_num = cls._generate_shard_info()

        if isinstance(dataset_config.data_loader, dict):
            dataset = build_dataset_loader(dataset_config.data_loader,
                                           default_args={'num_shards': device_num, 'shard_id': rank_id})
        else:
            dataset = dataset_config.data_loader

        if isinstance(dataset_config.tokenizer, dict):
            tokenizer = build_tokenizer(dataset_config.tokenizer)
        else:
            tokenizer = dataset_config.tokenizer

        if (isinstance(dataset_config.text_transforms, list) and isinstance(dataset_config.text_transforms[0], dict)) \
                or isinstance(dataset_config.text_transforms, dict):
            text_transforms = build_transforms(dataset_config.text_transforms, default_args={"tokenizer": tokenizer})
        else:
            text_transforms = dataset_config.text_transforms

        if (isinstance(dataset_config.label_transforms, list) and isinstance(dataset_config.label_transforms[0], dict))\
                or isinstance(dataset_config.label_transforms, dict):
            label_transforms = build_transforms(dataset_config.label_transforms)
        else:
            label_transforms = dataset_config.label_transforms

        if isinstance(dataset_config.sampler, dict):
            sampler = build_sampler(dataset_config.sampler)
        else:
            sampler = dataset_config.sampler

        if sampler is not None:
            dataset = dataset.use_sampler(sampler)

        if text_transforms is not None:
            dataset = get_dataset_map(dataset,
                                      input_columns=dataset_config.input_columns,
                                      operations=text_transforms,
                                      output_columns=dataset_config.output_columns,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        if label_transforms is not None:
            dataset = get_dataset_map(dataset,
                                      input_columns=dataset_config.input_columns[1],
                                      operations=label_transforms,
                                      num_parallel_workers=dataset_config.num_parallel_workers,
                                      python_multiprocessing=dataset_config.python_multiprocessing)

        dataset = dataset.batch(dataset_config.batch_size,
                                drop_remainder=dataset_config.drop_remainder,
                                num_parallel_workers=dataset_config.num_parallel_workers)
        dataset = dataset.repeat(dataset_config.repeat)
        return dataset
