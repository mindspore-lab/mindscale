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
"""get mock data for dataset"""
import os
import json
import pickle
import numpy as np
import cv2
from mindspore.mindrecord import FileWriter

np.random.seed(0)


# pylint: disable=W0703
def get_cifar100_data(data_path, data_num: int = 1):
    """get cifar100 data"""
    np.random.seed(0)
    meta_dict = {
        b"fine_label_names": [b"fine_1", b"fine_2"],
        b"coarse_label_names": [b"coarse_1", b"coarse_2"]
    }

    train_dict = {
        b"fine_labels": list(np.random.randint(0, 2, size=data_num)),
        b"coarse_labels": list(np.random.randint(0, 2, size=data_num)),
        b"data": np.random.randint(0, 256, size=(data_num, 3*32*32))
    }

    test_dict = {
        b"fine_labels": list(np.random.randint(0, 2, size=data_num)),
        b"coarse_labels": list(np.random.randint(0, 2, size=data_num)),
        b"data": np.random.randint(0, 256, size=(data_num, 3*32*32))
    }

    meta_path = os.path.join(data_path, "meta")
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(meta_path, "wb") as w_meta, open(train_path, "wb") as w_train, \
                    open(test_path, "wb") as w_test:
                pickle.dump(meta_dict, w_meta)
                pickle.dump(train_dict, w_train)
                pickle.dump(test_dict, w_test)
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(meta_path):
                os.remove(meta_path)
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(test_path):
                os.remove(test_path)
            print(f"cifar100 data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"cifar100 data initialize failed for {count} times.")


# pylint: disable=W0703
def get_llava_data(data_path, data_num: int = 1):
    """get llava data"""
    test_image_name = "test.jpg"
    os.makedirs(os.path.join(data_path, "train2014"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "text"), exist_ok=True)
    test_jpg_path = os.path.join(data_path, "train2014", f"COCO_train2014_{test_image_name}")
    data = [
        {
            "id": "000000442786",
            "image": test_image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "What do you see happening in this image?\n<image>"
                },
                {
                    "from": "gpt",
                    "value": "The scene depicts a lively plaza area with several people walking and enjoying their "
                             "time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. "
                             "The kite has multiple sections attached to it, spread out in various directions as "
                             "if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and "
                             "interacting with others. Some of these individuals are carrying handbags, and others have"
                             " backpacks. The image captures the casual, social atmosphere of a bustling plaza on a "
                             "nice day."
                }
            ]
        }
    ] * data_num

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            txt_path = os.path.join(data_path, "text", "detail_23k.json")
            with open(txt_path, "w", encoding="utf-8") as w_data:
                w_data.write(json.dumps(data))

            image = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.imwrite(test_jpg_path, image)
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if os.path.exists(test_jpg_path):
                os.remove(test_jpg_path)
            print(f"llava data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

        if not success_sig:
            raise RuntimeError(f"llava data initialize failed for {count} times.")


# pylint: disable=W0703
def get_mindrecord_data(data_path, data_num: int = 1, seq_len: int = 16):
    """get mindrecord data"""
    np.random.seed(0)
    output_file = os.path.join(data_path, "test.mindrecord")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            writer = FileWriter(output_file)
            data_schema = {
                "input_ids": {"type": "int32", "shape": [-1]},
                "attention_mask": {"type": "int32", "shape": [-1]},
                "labels": {"type": "int32", "shape": [-1]}
            }
            writer.add_schema(data_schema, "test-schema")
            for _ in range(data_num):
                features = {}
                features["input_ids"] = np.random.randint(0, 64, size=seq_len).astype(np.int32)
                features["attention_mask"] = features["input_ids"]
                features["labels"] = features["input_ids"]
                writer.write_raw_data([features])
            writer.commit()
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(output_file):
                os.remove(output_file)
            if os.path.exists(output_file + ".db"):
                os.remove(output_file + ".db")
            print(f"mindrecord data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"mindrecord data initialize failed for {count} times.")


# pylint: disable=W0703
def get_adgen_data(data_path, is_json_error: bool = False):
    """get adgen data"""
    np.random.seed(0)
    if not is_json_error:
        data1 = {
            "content": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
            "summary": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"
        }
        data2 = {
            "mock_key_1": "类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤",
            "mock_key_2": "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"
        }
        data3 = {
            "content": "",
            "summary": ""
        }
        data = [data1, data2, data3]
        train_path = os.path.join(data_path, "train.json")
    else:
        data = ["类型#裤*版型#宽松*风格#性感*图案#线条*裤型#阔腿裤", "宽松的阔腿裤这两年真的吸粉不少，明星时尚达人的心头爱。"]
        train_path = os.path.join(data_path, "json_error_train.json")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w:
                if not is_json_error:
                    write_data = [json.dumps(item, ensure_ascii=False) for item in data] + ["   "]
                    w.write("\n".join(write_data))
                else:
                    w.write("\n".join(data))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"adgen data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"adgen data initialize failed for {count} times.")


# pylint: disable=W0703
def get_cluener_data(data_path, data_num: int = 1):
    """get cluener data"""
    np.random.seed(0)

    train_data = {
        "text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，",
        "label": {
            "name": {"叶老桂": [[9, 11]]},
            "company": {"浙商银行": [[0, 0]]}
        }
    }

    test_data = {
        "id": 0,
        "text": "四川敦煌学”。近年来，丹棱县等地一些不知名的石窟迎来了海内外的游客，他们随身携带着胡文和的著作。"
    }

    dev_data = {
        "text": "彭小军认为，国内银行现在走的是台湾的发卡模式，先通过跑马圈地再在圈的地里面选择客户，",
        "label": {
            "address": {"台湾": [[15, 16]]},
            "name": {"彭小军": [[0, 2]]}
        }
    }

    train_path = os.path.join(data_path, "train.json")
    test_path = os.path.join(data_path, "test.json")
    dev_path = os.path.join(data_path, "dev.json")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train, \
                    open(test_path, "w", encoding="utf-8") as w_test, \
                    open(dev_path, "w", encoding="utf-8") as w_dev:
                w_train.write("\n".join([json.dumps(train_data, ensure_ascii=False)] * data_num))
                w_test.write("\n".join([json.dumps(test_data, ensure_ascii=False)] * data_num))
                w_dev.write("\n".join([json.dumps(dev_data, ensure_ascii=False)] * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(test_path):
                os.remove(test_path)
            if os.path.exists(dev_path):
                os.remove(dev_path)
            print(f"cluener data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"cluener data initialize failed for {count} times.")


# pylint: disable=W0703
def get_flickr8k_data(data_path, data_num: int = 1):
    """get flickr8k data"""
    test_jpg_path = os.path.join(data_path, "Flickr8k_Dataset", "Flickr8k_Dataset")
    os.makedirs(test_jpg_path)
    text_dir = os.path.join(data_path, "Flickr8k_text")
    os.makedirs(text_dir)
    train_path = os.path.join(text_dir, "Flickr_8k.trainImages.txt")
    test_path = os.path.join(text_dir, "Flickr_8k.testImages.txt")
    dev_path = os.path.join(text_dir, "Flickr_8k.devImages.txt")
    token_path = os.path.join(text_dir, "Flickr8k.token.txt")

    token_content = [
        "mock.jpg#0\tA child in a pink dress is climbing up a set of stairs in an entry way .",
        "mock.jpg#1\tA girl going into a wooden building .",
        "mock.jpg#2\tA little girl climbing into a wooden playhouse ."
    ]
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train, \
                    open(test_path, "w", encoding="utf-8") as w_test, \
                    open(dev_path, "w", encoding="utf-8") as w_dev, \
                    open(token_path, "w", encoding="utf-8") as w_token:
                w_train.write("\n".join(["mock.jpg"] * data_num))
                w_test.write("\n".join(["mock.jpg"] * data_num))
                w_dev.write("\n".join(["mock.jpg"] * data_num))
                w_token.write("\n".join(token_content))
                image = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(test_jpg_path, "mock.jpg"), image)
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(test_path):
                os.remove(test_path)
            if os.path.exists(dev_path):
                os.remove(dev_path)
            if os.path.exists(token_path):
                os.remove(token_path)
            print(f"flickr8k data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"flickr8k data initialize failed for {count} times.")


# pylint: disable=W0703
def get_squad_data(data_path, data_num: int = 1):
    """get squad data"""
    train_data = {
        "data": [
            {
                "title": "Super_Bowl_50",
                "paragraphs": [
                    {
                        "context": "An increasing sequence: one, two, three.",
                        "qas": [
                            {
                                "answers": [
                                    {
                                        "answer_start": 24,
                                        "text": "one"
                                    }
                                ],
                                "question": "华为是一家总部位于中国深圳的多元化科技公司",
                                "id": "56be4db0acb8001400a502ec"
                            }
                        ]
                    },
                ]
            }
        ]
    }

    dev_data = {
        "data": [
            {
                "title": "University_of_Notre_Dame",
                "paragraphs": [
                    {
                        "context": "An increasing sequence: one, two, three.",
                        "qas": [
                            {
                                "answers": [
                                    {
                                        "answer_start": 24,
                                        "text": "one"
                                    }
                                ],
                                "question": "华为是一家总部位于中国深圳的多元化科技公司",
                                "id": "5733be284776f41900661182"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    train_path = os.path.join(data_path, "train-v1.1.json")
    dev_path = os.path.join(data_path, "dev-v1.1.json")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train, \
                    open(dev_path, "w", encoding="utf-8") as w_dev:
                train_data["data"] = train_data["data"] * data_num
                dev_data["data"] = dev_data["data"] * data_num
                w_train.write(json.dumps(train_data, ensure_ascii=False))
                w_dev.write(json.dumps(dev_data, ensure_ascii=False))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(dev_path):
                os.remove(dev_path)
            print(f"squad data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"squad data initialize failed for {count} times.")


# pylint: disable=W0703
def get_wikitext_data(data_path, data_num: int = 1):
    """get wikitext data"""
    train_path = os.path.join(data_path, "wiki.train.tokens")

    data = ["= 华为 =", "华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。", "== 华为 ==",
            "An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13."]
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train:
                w_train.write("\n".join(data * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"flickr8k data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"flickr8k data initialize failed for {count} times.")


# pylint: disable=W0703
def get_json_data(data_path, data_num: int = 1):
    """get json data"""
    train_path = os.path.join(data_path, "train.json")

    data = {"input": ["华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。"]}
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train:
                w_train.write("\n".join([json.dumps(data, ensure_ascii=False)] * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"json data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"json data initialize failed for {count} times.")


# pylint: disable=W0703
def get_wmt16_data(data_path, data_num: int = 1):
    """get wmt16 data"""
    source_path = os.path.join(data_path, "train.source")
    target_path = os.path.join(data_path, "train.target")

    source_data = "Membership of Parliament: see Minutes"
    target_data = "Componenţa Parlamentului: a se vedea procesul-verbal"
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(source_path, "w", encoding="utf-8") as w_source, \
                    open(target_path, "w", encoding="utf-8") as w_target:
                w_source.write("\n".join([source_data] * data_num))
                w_target.write("\n".join([target_data] * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(source_path):
                os.remove(source_path)
            if os.path.exists(target_path):
                os.remove(target_path)
            print(f"wmt16 data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"wmt16 data initialize failed for {count} times.")


# pylint: disable=W0703
def get_cmrc_data(data_path, data_num: int = 1):
    """get cmrc data"""
    data = {
        "version": "v1.0",
        "data": [
            {
                "paragraphs": [
                    {
                        "id": "TRAIN_186",
                        "context": "华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。",
                        "qas": [
                            {
                                "question":
                                    "An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.",
                                "id": "TRAIN_186_QUERY_0",
                                "answers": [
                                    {
                                        "text": "华为",
                                        "answer_start": 30
                                    }
                                ]
                            },
                        ]
                    }
                ],
                "id": "TRAIN_186",
                "title": "范廷颂"
            }
        ]
    }

    data_path = os.path.join(data_path, "train.json")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(data_path, "w", encoding="utf-8") as w:
                data["data"] = data["data"] * data_num
                w.write(json.dumps(data, ensure_ascii=False))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(data_path):
                os.remove(data_path)
            print(f"cmrc data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"cmrc data initialize failed for {count} times.")


# pylint: disable=W0703
def get_agnews_data(data_path, data_num: int = 1):
    """get agnews data"""
    data = ["\"1\"", "\"华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。\"",
            "\"An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13.\""]

    data_path = os.path.join(data_path, "agnews")

    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(data_path, "w", encoding="utf-8") as w:
                w.write("\n".join([",".join(data)] * data_num))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(data_path):
                os.remove(data_path)
            print(f"agnews data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"agnews data initialize failed for {count} times.")


# pylint: disable=W0703
def get_alpaca_data(data_path, data_num: int = 1):
    """get alpaca data"""
    train_path = os.path.join(data_path, "train.json")

    data = {
        "instruction": "华为是一家总部位于中国深圳的多元化科技公司,成立于1987年,是全球最大的电信设备制造商之一。",
        "input": "",
        "output": "An increasing sequence: one, two, three, five, six, seven, nine, 10, 11, 12, 13."
    }
    retry = True
    count = 0
    success_sig = False
    while retry:
        try:
            count += 1
            with open(train_path, "w", encoding="utf-8") as w_train:
                w_train.write(json.dumps([data] * data_num, ensure_ascii=False))
            retry = False
            success_sig = True
        except BaseException as e:
            if os.path.exists(train_path):
                os.remove(train_path)
            print(f"json data initialize failed, due to \"{e}\".")
            if count >= 3:
                retry = False

    if not success_sig:
        raise RuntimeError(f"json data initialize failed for {count} times.")
