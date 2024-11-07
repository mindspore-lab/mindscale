# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""ChatGLM2 Transformer."""
import copy
import math
from typing import Tuple

import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore import Tensor, nn, ops, Layout
from mindspore import dtype as mstype
from mindformers.modules.infer_attention import InferAttention
from mindformers.modules import LayerNorm
from mindformers.modules.layers import Linear
from mindformers.modules.flash_attention import FlashAttention
from mindformers.models.utils import LayerSetting, check_fine_grain_interleave_valid
from mindformers.pet.tuners.ptuning2_adapter import Ptuning2Adapter
from mindformers.version_control import get_dropout
from mindformers.tools.logger import logger

from .glm2_config import ChatGLM2Config
from .glm2_modules import ChatGLM2MLP, ChatGLM2RMSNorm, GetCompressMask, ChatGLM2RotaryEmbedding, GetEodResetMask


class CoreAttention(nn.Cell):
    """ChatGLM2 core attention."""

    def __init__(self, config: ChatGLM2Config, layer_number):
        super(CoreAttention, self).__init__()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.head_dim = config.kv_channels
        projection_size = config.kv_channels * config.num_attention_heads

        self.n_head = config.num_attention_heads
        self.norm_factor = math.sqrt(self.head_dim)
        self.mul_mask = P.Mul()
        self.add = P.Add()

        # Strided linear layer.
        self.attention_dropout = get_dropout(config.attention_dropout)

        parallel_config = config.parallel_config

        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.batch_matmul_q_k.shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.batch_matmul = P.BatchMatMul().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),
             (parallel_config.data_parallel, parallel_config.model_parallel, 1, 1)))
        self.softmax = nn.Softmax(axis=-1)

        self.merger_head_transpose = P.Transpose().shard(
            ((parallel_config.data_parallel, parallel_config.model_parallel, 1, 1),))
        self.reshape = P.Reshape()

        self.compute_dtype = config.compute_dtype
        self.multi_query_attention = config.multi_query_attention
        if self.multi_query_attention:
            self.n_kv_head = config.multi_query_group_num
            self.qkv_hidden_size = (
                projection_size + 2 * self.head_dim * config.multi_query_group_num)
        self.transpose = P.Transpose()
        self.cast = P.Cast()

    def construct(self, query_layer, key_layer, value_layer, attention_mask):
        """
        calculate attention function
        """
        # query_layer [b, heads, seq, hidden_size_per_head]
        # key_layer [b, heads, seq, hidden_size_per_head]
        # value_layer # [bs, heads, seq_len, hidden_size_per_head]

        # seqlen, batch, head, hidden_size

        if self.apply_query_key_layer_scaling:
            query_layer = query_layer / self.norm_factor

        # ===================================
        # Raw attention scores. [b, heads, s, s]
        # ===================================
        # [b, heads, seq_q, hidden_size_per_head] × [b, heads, seq_k, hidden_size_per_head]^T -> [b, heads, seq_q, seq_k]
        matmul_result = self.batch_matmul_q_k(query_layer, key_layer)

        # record original score dtype
        attention_scores_dtype = matmul_result.dtype
        # [b, heads, seq, seq]
        attention_scores = matmul_result

        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = ops.ones((attention_scores.shape[0],
                                       1,
                                       attention_scores.shape[2],
                                       attention_scores.shape[3]), dtype=mstype.bool_)
            attention_mask.tril()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_mask = self.mul_mask(attention_mask, -10000)
            attention_scores = self.add(attention_scores, attention_mask)

        if self.attention_softmax_in_fp32:
            attention_scores = F.cast(attention_scores, mstype.float32)

        attention_probs = self.softmax(attention_scores)
        attention_probs = F.cast(attention_probs, attention_scores_dtype)

        attention_probs = self.attention_dropout(attention_probs)

        # [bs, heads, seq_q, seq_k] x [bs, heads, seq_v, hidden_size_per_head] -> [b, heads, seq_q, hidden_size_per_head]
        context_layer = self.batch_matmul(attention_probs, value_layer)
        context_layer = F.cast(context_layer, self.compute_dtype)
        context_layer = self._merge_heads(context_layer)

        return context_layer

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge


class ChatGLM2SelfAttention(nn.Cell):
    """ChatGLM2 self-attention."""

    def __init__(self, config: ChatGLM2Config, layer_number):
        super(ChatGLM2SelfAttention, self).__init__()
        self.is_first_iteration = True
        self.layer_number = max(1, layer_number)
        self.head_dim = config.kv_channels
        self.projection_size = config.kv_channels * config.num_attention_heads
        # Per attention head and per partition values.
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.norm_factor = math.sqrt(self.head_dim)
        self.n_head = config.num_attention_heads
        self.params_dtype = config.param_init_type
        self.compute_dtype = config.compute_dtype
        self.batch_size = config.batch_size
        self.pre_seq_len = config.pre_seq_len
        self.n_rep = self.n_head // config.multi_query_group_num

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        self.kv_hidden_size = self.projection_size
        self.use_rearrange_rope = config.use_rearrange_rope
        self.mask_generate = config.mask_generate  # "inmap", "compress_reset"

        if self.multi_query_attention:
            self.n_kv_head = config.multi_query_group_num
            self.qkv_hidden_size = self.projection_size + 2 * self.head_dim * config.multi_query_group_num
            self.kv_hidden_size = self.n_kv_head * self.head_dim

        parallel_config = config.parallel_config
        dp, cp, mp = _parallel_decompose(config)
        self.cp_ds = 1
        self.cp_co = cp

        qkv_has_bias = config.add_bias_linear or config.add_qkv_bias
        kv_mp = self.n_kv_head if self.n_kv_head < mp else mp

        self.qkv_concat = config.qkv_concat
        if config.qkv_concat:
            self.query_key_value = Linear(config.hidden_size,
                                          self.qkv_hidden_size,
                                          has_bias=config.add_bias_linear or config.add_qkv_bias,
                                          param_init_type=self.params_dtype,
                                          compute_dtype=self.compute_dtype)
            self.split_qkv = ops.auto_generate.SplitWithSize()
            self.shard_wqkv_concat(cp, dp, mp, qkv_has_bias)
        else:
            self.wq = Linear(config.hidden_size,
                             self.projection_size,
                             has_bias=config.add_bias_linear or config.add_qkv_bias,
                             param_init_type=self.params_dtype,
                             compute_dtype=self.compute_dtype)

            self.wk = Linear(config.hidden_size,
                             self.kv_hidden_size,
                             has_bias=config.add_bias_linear or config.add_qkv_bias,
                             param_init_type=self.params_dtype,
                             compute_dtype=self.compute_dtype)
            self.wv = Linear(config.hidden_size,
                             self.kv_hidden_size,
                             has_bias=config.add_bias_linear or config.add_qkv_bias,
                             param_init_type=self.params_dtype,
                             compute_dtype=self.compute_dtype)
            self.shard_wqkv_non_concat(config, qkv_has_bias, kv_mp)
        self.shape = P.Shape()
        self.dense = Linear(self.projection_size,
                            config.hidden_size,
                            has_bias=config.add_bias_linear,
                            param_init_type=self.params_dtype,
                            compute_dtype=self.compute_dtype)
        self.dense.shard(strategy_matmul=((dp * cp, mp), (1, mp)), strategy_bias=((dp * cp, 1), (1,)))
        if config.parallel_config.use_seq_parallel and self.is_first_iteration:
            self.dense.shard(((dp * cp, mp), (1, mp)), strategy_bias=((dp * cp * mp, 1), (1,)),
                             out_strategy_matmul=((dp * cp * mp, 1),))
        self.use_flash_attention = config.use_flash_attention
        self.use_past = config.use_past
        if self.use_past:
            self.infer_attention = self.init_infer_attention(config, mp, parallel_config)
        else:
            self.core_attention = CoreAttention(config, self.layer_number)
            self.reshape = P.Reshape()
            self.stack = P.Stack(axis=-1)
            self.mul = P.Mul()
            self.sub = P.Sub()
            self.add = P.Add()
            self.concat = P.Concat(axis=-1)
            self.transpose = P.Transpose().shard(((dp, cp, mp, 1),))
            self.kv_transpose = P.Transpose().shard(((dp, cp, kv_mp, 1),))
            self.cast = P.Cast()
            self.tile_kv = P.Tile()
            if self.use_rearrange_rope:
                self.apply_rotary_emb = ChatGLM2RotaryEmbedding(compute_dtype=config.rotary_dtype)
                self.apply_rotary_emb.shard((dp, cp, mp, 1), kv_strategy=(dp, cp, kv_mp, 1))
            input_layout, sparse_mode = self.select_fa_configs()
            if self.use_flash_attention:
                self.flash_attention = FlashAttention(head_num=config.num_attention_heads,
                                                      scale_value=1. / math.sqrt(self.head_dim),
                                                      input_layout=input_layout,
                                                      sparse_mode=sparse_mode,
                                                      keep_prob=1. - config.attention_dropout,
                                                      pre_tokens=65536,
                                                      next_tokens=0)
                self.shard_fa(cp, dp, mp, parallel_config)
                self.cp_transpose_before = P.Transpose().shard(((dp, cp, mp, 1, 1),))
                self.cp_transpose_kv_before = P.Transpose().shard(((dp, cp, kv_mp, 1, 1),))
                self.cp_transpose_after = P.Transpose().shard(((dp, cp, 1, mp, 1),))
            self.get_attention_mask = self.select_mask_generate(config, parallel_config, sparse_mode)
            self.merger_head_transpose = P.Transpose().shard(((dp, mp, cp, 1),))

    def shard_wqkv_concat(self, cp, dp, mp, qkv_has_bias):
        """shard wqkv concat"""
        qkv_strategy_matmul = ((dp * cp, 1), (mp, 1))
        qkv_strategy_bias = ((dp * cp, mp), (mp,)) if qkv_has_bias else None
        self.query_key_value.shard(strategy_matmul=qkv_strategy_matmul, strategy_bias=qkv_strategy_bias)
        self.split_qkv.add_prim_attr("skip_redistribution", True)
        self.split_qkv.shard(((dp * cp, 1, 1),))

    def shard_wqkv_non_concat(self, config, qkv_has_bias, kv_mp):
        """shard wqkv non concat"""
        dp, cp, mp = _parallel_decompose(config)
        kv_strategy_matmul = ((dp * cp, 1), (kv_mp, 1))
        q_strategy_matmul = ((dp * cp, 1), (mp, 1))
        kv_strategy_bias = ((dp * cp, kv_mp), (kv_mp,)) if qkv_has_bias else None
        q_strategy_bias = ((dp * cp, mp), (mp,)) if qkv_has_bias else None
        self.wq.shard(strategy_matmul=q_strategy_matmul, strategy_bias=q_strategy_bias)
        self.wk.shard(strategy_matmul=kv_strategy_matmul, strategy_bias=kv_strategy_bias)
        self.wv.shard(strategy_matmul=kv_strategy_matmul, strategy_bias=kv_strategy_bias)
        return kv_mp

    def init_infer_attention(self, config, mp, parallel_config):
        """init infer attention"""
        rotary_cos_format = 2 if self.use_rearrange_rope else 3
        infer_attention = InferAttention(self.n_head,
                                         self.head_dim,
                                         self.n_kv_head,
                                         pa_n_head_split=self.n_head // mp,
                                         pa_n_kv_head_split=self.n_kv_head // mp,
                                         scale_value=1. / math.sqrt(self.head_dim),
                                         pre_tokens=65536,
                                         next_tokens=0,
                                         block_size=config.block_size,
                                         num_blocks=config.num_blocks,
                                         is_dynamic=config.is_dynamic,
                                         use_flash_attention=self.use_flash_attention,
                                         rotary_cos_format=rotary_cos_format,
                                         compute_dtype=self.compute_dtype)
        infer_attention.shard(parallel_config)
        return infer_attention

    def select_fa_configs(self):
        """select fa configs"""
        if self.mask_generate == "compress_reset":
            sparse_mode = 3
            input_layout = "TND"
        elif self.mask_generate == "inmap":
            sparse_mode = 0
            input_layout = "BSH"
        else:
            sparse_mode = 0
            input_layout = "BSH"
        return input_layout, sparse_mode

    def shard_fa(self, cp, dp, mp, parallel_config):
        """shard flash attention"""
        if self.n_kv_head >= mp:
            fa_parallel_config = copy.deepcopy(parallel_config)
            fa_parallel_config.model_parallel = mp * self.cp_ds
            fa_parallel_config.context_parallel = self.cp_co
            self.flash_attention.shard(fa_parallel_config)
        else:
            if not self.qkv_concat:
                self.wk.weight.parallel_optimizer = False
                self.wv.weight.parallel_optimizer = False
            layout = Layout((dp, self.n_kv_head, mp * self.cp_ds // self.n_kv_head, cp),
                            ("dp", "mp1", "mp2", "cp"))
            self.flash_attention.flash_attention.shard(
                (
                    layout("dp", "cp", ("mp1", "mp2")),
                    layout("dp", "None", "mp1"),
                    layout("dp", "None", "mp1"),
                    layout("dp", "None", "cp", "None")
                )
            )

    def select_mask_generate(self, config, parallel_config, sparse_mode):
        """select which mask to use"""
        if self.mask_generate == "compress_reset" or sparse_mode == 2:
            return GetCompressMask(mask_length=2048)
        if self.mask_generate == "inmap":
            return GetEodResetMask(seq_length=config.seq_length, parallel_config=parallel_config)
        return ops.Identity()

    def _repeat_kv(self, x, rep):
        """repeat kv"""
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 2d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # bs, seq_length, head, size_per_head
        x_shape = x.shape
        new_shape = (x_shape[0], x_shape[1], -1)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def apply_rotary_pos_emb(self, x: Tensor, rotary_pos_emb: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """apply rotary position embedding to q,k."""
        # x: [b, heads, seq, hidden_size_per_head]
        bs, num_heads, seq_len, _ = x.shape  # 1, 32，4, 128
        # rope_cache: first (seq_len, kv_channels//4, 2), other (1, kv_channels//4, 2)
        _, _, rope_cache = rotary_pos_emb
        rot_dim = rope_cache.shape[-2] * 2  # kv_channels // 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # ms not support variable sizes
        # truncate to support variable sizes
        # [bs, nh, sq, kv_channels//4, 2]
        xshaped = self.reshape(x, (bs, num_heads, seq_len, rot_dim // 2, 2))
        # [bs, 1, sq, kv_channels//4, 2]
        if rope_cache.dtype == mstype.bfloat16:
            rope_cache = self.cast(rope_cache, mstype.float32)
        rope_cache = self.reshape(rope_cache, (-1, 1, seq_len, xshaped.shape[3], 2))

        xshaped_0, xshaped_1 = ops.split(xshaped, 1, -1)
        rope_cache_0, rope_cache_1 = ops.split(rope_cache, 1, -1)
        x_out1 = self.sub(self.mul(xshaped_0, rope_cache_0), self.mul(xshaped_1, rope_cache_1))
        x_out2 = self.add(self.mul(xshaped_1, rope_cache_0), self.mul(xshaped_0, rope_cache_1))
        x_out = self.stack((x_out1, x_out2))
        x_out = self.reshape(x_out, (x_out.shape[0], x_out.shape[1], x_out.shape[2], -1))
        x_out = self.cast(x_out, x_pass.dtype)
        # [bs, sq, nh, hidden_size_per_head]
        return self.concat((x_out, x_pass))

    def add_prefix_if_need(self, prefix_key_value, key_layer, value_layer, attention_mask):
        """
        add p-tuning v2 prefix if need
        """
        if not isinstance(self.pre_seq_len, int) or self.pre_seq_len <= 0:
            return key_layer, value_layer, attention_mask

        if not self.use_rearrange_rope:
            key_layer = self.kv_transpose(key_layer, (0, 2, 1, 3))
            value_layer = self.kv_transpose(value_layer, (0, 2, 1, 3))
        seq_len = key_layer.shape[2]

        key_layer, value_layer = Ptuning2Adapter.add_prefix(
            prefix_key_value,
            key_layer,
            value_layer
        )

        if attention_mask is not None:
            batch_size = attention_mask.shape[0]
            prefix_mask = attention_mask.new_zeros((batch_size, 1, seq_len, self.pre_seq_len))
            m_cat = P.Concat(3)
            # [bs, 1, seq_len, pre_seq_len + seq_len]
            attention_mask = m_cat((prefix_mask, attention_mask))
        if not self.use_rearrange_rope:
            key_layer = self.kv_transpose(key_layer, (0, 2, 1, 3))
            value_layer = self.kv_transpose(value_layer, (0, 2, 1, 3))
        return key_layer, value_layer, attention_mask

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, batch_valid_length=None, prefix_key_value=None,
                  block_tables=None, slot_mapping=None):
        """Forward process of self-attention."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask: (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first -> (sen length, kv_channels//4, 2), after -> (1, kv_channels//4, 2]
        bs, seq_len, _ = self.shape(hidden_states)
        if self.qkv_concat:
            # [bs, seq_len, qkv_hidden_size]
            mixed_raw_layer = self.query_key_value(hidden_states)

            # not compatible with ms below 2.0
            query, key, value = self.split_qkv(mixed_raw_layer,
                                               (self.n_head * self.head_dim,
                                                self.n_kv_head * self.head_dim,
                                                self.n_kv_head * self.head_dim), 2)
        else:
            query = self.wq(hidden_states)
            key = self.wk(hidden_states)
            value = self.wv(hidden_states)

        # key and value for current token(s)
        if self.use_past:
            context_layer = self.infer_attention(query, key, value, batch_valid_length, block_tables, slot_mapping,
                                                 rotary_pos_emb, attention_mask)
        else:
            query = self.reshape(query, (bs, seq_len, self.n_head, self.head_dim))
            key = self.reshape(key, (bs, seq_len, self.n_kv_head, self.head_dim))
            value = self.reshape(value, (bs, seq_len, self.n_kv_head, self.head_dim))

            if self.use_rearrange_rope:
                query, key = self.apply_rotary_emb(query, key, rotary_pos_emb)  # dp, mp, 1, 1
            else:
                query = self.transpose(query, (0, 2, 1, 3))
                key = self.kv_transpose(key, (0, 2, 1, 3))
                query = self.apply_rotary_pos_emb(query, rotary_pos_emb)
                key = self.apply_rotary_pos_emb(key, rotary_pos_emb)

            key, value, attention_mask = self.add_prefix_if_need(
                prefix_key_value,
                key,
                value,
                attention_mask
            )
            if not self.use_rearrange_rope:
                query = self.transpose(query, (0, 2, 1, 3))
                key = self.kv_transpose(key, (0, 2, 1, 3))

            if self.use_flash_attention:
                if self.cp_ds > 1:
                    query = F.reshape(query, (bs, seq_len, self.n_head // self.cp_ds, self.cp_ds, self.head_dim))
                    key = F.reshape(key, (bs, seq_len, self.n_kv_head // self.cp_ds, self.cp_ds, self.head_dim))
                    value = F.reshape(value, (bs, seq_len, self.n_kv_head // self.cp_ds, self.cp_ds, self.head_dim))
                    query = self.cp_transpose_before(query, (0, 1, 3, 2, 4))
                    key = self.cp_transpose_kv_before(key, (0, 1, 3, 2, 4))
                    value = self.cp_transpose_kv_before(value, (0, 1, 3, 2, 4))
                if self.mask_generate == "compress_reset":
                    actual_seq_len = self.cast(self.reshape(attention_mask, (-1,)), mstype.int64)
                    attention_mask = self.get_attention_mask(attention_mask)

                    query = self.reshape(query, (bs * seq_len, self.n_head, self.head_dim))
                    key = self.reshape(key, (bs * seq_len, self.n_kv_head, self.head_dim))
                    value = self.reshape(value, (bs * seq_len, self.n_kv_head, self.head_dim))
                    context_layer = self.flash_attention(query, key, value, attention_mask,
                                                         None, None, None,
                                                         actual_seq_len, actual_seq_len)
                    if self.cp_ds > 1:
                        context_layer = self.cp_transpose_after(F.reshape(context_layer, (
                            bs, seq_len, self.cp_ds, self.n_head // self.cp_ds, self.head_dim)), (0, 1, 3, 2, 4))
                    context_layer = self.reshape(context_layer, (bs, seq_len, -1))
                else:
                    query = F.reshape(query, (bs, seq_len, self.n_head * self.head_dim))
                    key = F.reshape(key, (bs, seq_len, self.n_kv_head * self.head_dim))
                    value = F.reshape(value, (bs, seq_len, self.n_kv_head * self.head_dim))
                    if self.mask_generate == "inmap":
                        attention_mask = self.get_attention_mask(attention_mask)
                        attention_mask = F.reshape(attention_mask, (bs, 1, seq_len, seq_len))
                    context_layer = self.flash_attention(query, key, value, attention_mask)
                    if self.cp_ds > 1:
                        context_layer = self.cp_transpose_after(F.reshape(context_layer, (
                            bs, seq_len, self.cp_ds, self.n_head // self.cp_ds, self.head_dim)), (0, 1, 3, 2, 4))
                    context_layer = F.reshape(context_layer, (bs, seq_len, self.n_head * self.head_dim))

            else:
                query = self.transpose(query, (0, 2, 1, 3))
                key = self.kv_transpose(key, (0, 2, 1, 3))
                value = self.kv_transpose(value, (0, 2, 1, 3))
                key = self._repeat_kv(key, self.n_rep)
                value = self._repeat_kv(value, self.n_rep)
                attention_mask = F.reshape(self.get_attention_mask(attention_mask), (bs, 1, seq_len, seq_len))
                context_layer = self.core_attention(query, key, value, attention_mask)

        # # =================
        # # Output. [bs, seq_len, hidden_size]
        # # =================

        output = self.dense(context_layer)

        return output


def _parallel_decompose(config):
    dp, cp, mp = config.parallel_config.data_parallel, \
                 config.parallel_config.context_parallel, config.parallel_config.model_parallel
    return dp, cp, mp


class ChatGLM2Block(nn.Cell):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLM2Config, layer_number: int):
        super(ChatGLM2Block, self).__init__()
        self.is_first_iteration = True
        self.layer_number = layer_number
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.layernorm_dtype = config.layernorm_compute_type
        self.compute_dtype = config.compute_dtype

        layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                               param_init_type=self.layernorm_dtype)
        self.input_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)
        if config.parallel_config.pipeline_stage > 1:
            self.input_layernorm.set_comm_fusion(2)

        # Self attention.
        self.self_attention = ChatGLM2SelfAttention(config, layer_number)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                        param_init_type=self.layernorm_dtype)

        # MLP
        self.mlp = ChatGLM2MLP(config)

        dp, cp, mp = _parallel_decompose(config)
        self.dropout = get_dropout(self.hidden_dropout)
        self.dropout.dropout.shard(((config.parallel_config.data_parallel, config.parallel_config.model_parallel, 1),))
        self.add = P.Add()
        self.cast = P.Cast()

        self.residual_dtype = mstype.float32
        self.input_layernorm.shard(((dp, cp, 1),))
        self.dropout.dropout.shard(((dp, cp, 1),))
        if config.parallel_config.use_seq_parallel and self.is_first_iteration:
            self.add.shard(((dp, cp * mp, 1), (dp, cp * mp, 1)))
            self.input_layernorm.shard(((dp, cp * mp, 1),))
            self.post_attention_layernorm.shard(((dp, cp * mp, 1),))
            self.dropout.dropout.shard(((dp, cp * mp, 1),))
            self.mlp.dense_4h_to_h.shard(strategy_matmul=((dp * cp, mp), (1, mp)),
                                         strategy_bias=((dp * cp * mp, 1), (1,)),
                                         out_strategy_matmul=((dp * cp * mp, 1),))

    def set_select_recompute(self):
        self.input_layernorm.recompute(False)
        self.post_attention_layernorm.recompute(False)
        self.self_attention.recompute()
        self.mlp.recompute()
        self.dropout.dropout.recompute(False)
        self.cast.recompute(False)

    def construct(self, hidden_states, attention_mask, rotary_pos_emb, batch_valid_length=None, prefix_key_value=None,
                  block_tables=None, slot_mapping=None):
        """Forward process of the transformer layer."""
        # hidden_states: [bs, seq_len, hidden_size]
        # attention_mask first: (bs, 1, seq_len, seq_len), after: (bs, 1, 1, seq_len)
        # rotary_pos_emb: first: (seq_len, kv_channels//4, 2)， after: (1, kv_channels//4, 2)
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output = self.self_attention(
            F.cast(layernorm_output, self.compute_dtype),
            attention_mask,
            rotary_pos_emb,
            batch_valid_length,
            prefix_key_value,
            block_tables=block_tables,
            slot_mapping=slot_mapping
        )

        # Residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = self.dropout(attention_output)
        layernorm_input = self.add(F.cast(residual, self.residual_dtype), F.cast(layernorm_input, self.residual_dtype))

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(F.cast(layernorm_output, self.compute_dtype))

        # Second residual connection.
        # False on default.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.dropout(mlp_output)
        output = self.add(F.cast(residual, self.residual_dtype), F.cast(output, self.residual_dtype))

        return output


class ChatGLM2Transformer(nn.Cell):
    """Transformer class."""

    def __init__(self, config: ChatGLM2Config):
        super(ChatGLM2Transformer, self).__init__()

        self.post_layer_norm = config.post_layer_norm
        self.compute_dtype = config.compute_dtype

        # Number of layers.
        self.num_layers = config.num_layers

        self.pre_seq_len = config.pre_seq_len
        self.fine_grain_interleave = check_fine_grain_interleave_valid(config.fine_grain_interleave,
                                                                       config.parallel_config)
        is_fine_grain_interleave_partial_recompute = self._is_fine_grain_interleave_partial_recompute(config)

        self.layers = nn.CellList()
        self.layer_setting = LayerSetting(config.num_layers,
                                          config.offset,
                                          config.parallel_config,
                                          config.pp_interleave_num)
        if self.fine_grain_interleave and not is_fine_grain_interleave_partial_recompute:
            logger.warning("GLM use fine_grain_interleave")
        elif self.fine_grain_interleave:
            logger.warning("GLM use fine_grain_interleave with partial recompute")
        else:
            logger.warning("GLM do not use fine_grain_interleave")
        for layer_id in range(self.num_layers):
            layer = ChatGLM2Block(config, layer_id + 1)
            self.layer_setting(layer, layer_id)
            self.layers.append(layer)

        dp, cp, mp = _parallel_decompose(config)
        if self.post_layer_norm:
            layer_norm_func = ChatGLM2RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon,
                                                   param_init_type=config.layernorm_compute_type)
            self.final_layernorm.shard(((dp, cp, 1),))
            self.final_layernorm.set_comm_fusion(config.parallel_config.gradient_aggregation_group)

            if config.parallel_config.pipeline_stage > 1:
                self.final_layernorm.set_comm_fusion(2)
                self.final_layernorm.pipeline_stage = config.parallel_config.pipeline_stage - 1

            if config.parallel_config.use_seq_parallel:
                self.final_layernorm.shard(((dp, cp * mp, 1),))

    def _is_fine_grain_interleave_partial_recompute(self, config):
        """whether the fine grain interleave is turned on with recompute"""
        if not self.fine_grain_interleave:
            return False
        recompute_config = config.parallel_config.recompute.recompute
        return isinstance(recompute_config, list)

    def _shard_fine_grain_interleaved(self, layer, layer_id):
        """shard fine grain interleave"""
        dp = self.config.parallel_config.data_parallel
        cp = self.config.parallel_config.context_parallel
        mp = self.config.parallel_config.model_parallel
        transformer_layout = Layout((dp, cp, mp, self.config.fine_grain_interleave),
                                    ("dp", "cp", "mp", "interleaved_parallel"))
        if layer_id > 0:
            qkv_has_bias = self.config.add_bias_linear or self.config.add_qkv_bias
            layer.input_layernorm.norm.shard((transformer_layout("dp", ("cp", "interleaved_parallel", "mp"), "None"),
                                              transformer_layout("None")))
            qkv_strategy_matmul = (transformer_layout(("dp", "cp", "interleaved_parallel"), "None"),
                                   transformer_layout("mp", "None"))
            qkv_strategy_bias = (transformer_layout(("dp", "cp", "interleaved_parallel"), "mp"),
                                 transformer_layout("mp")) if qkv_has_bias else None
            qkv_strategy_activation = None
            qkv_out_strategy_matmul = None
            if not self.config.qkv_concat:
                layer.self_attention.wq.shard(strategy_matmul=qkv_strategy_matmul, strategy_bias=qkv_strategy_bias,
                                              strategy_activation=qkv_strategy_activation,
                                              out_strategy_matmul=qkv_out_strategy_matmul)
                layer.self_attention.wk.shard(strategy_matmul=qkv_strategy_matmul, strategy_bias=qkv_strategy_bias,
                                              strategy_activation=qkv_strategy_activation,
                                              out_strategy_matmul=qkv_out_strategy_matmul)
                layer.self_attention.wv.shard(strategy_matmul=qkv_strategy_matmul, strategy_bias=qkv_strategy_bias,
                                              strategy_activation=qkv_strategy_activation,
                                              out_strategy_matmul=qkv_out_strategy_matmul)
                if qkv_has_bias:
                    layer.self_attention.wq.bias_add.add_prim_attr("fine_grained_interleaved_index", layer_id)
                else:
                    layer.self_attention.wq.matmul.add_prim_attr("fine_grained_interleaved_index", layer_id)
            else:
                layer.self_attention.query_key_value.shard(strategy_matmul=qkv_strategy_matmul,
                                                           strategy_bias=qkv_strategy_bias,
                                                           strategy_activation=qkv_strategy_activation,
                                                           out_strategy_matmul=qkv_out_strategy_matmul)
                layer.self_attention.split_qkv.shard((transformer_layout("dp", ("cp", "interleaved_parallel"), "mp"),))
                layer.self_attention.split_qkv.add_prim_attr("fine_grained_interleaved_index", layer_id)
        if layer_id < self.config.num_layers - 1:
            layer.self_attention.dense.matmul.add_prim_attr("fine_grained_interleaved_index", layer_id)
            layer.add1.add_prim_attr("fine_grained_interleaved_index", layer_id)
            wo_strategy_matmul = (transformer_layout(("dp", "cp", "interleaved_parallel"), "mp"),
                                  transformer_layout("None", "mp"))
            wo_strategy_bias = None
            wo_strategy_activation = None
            wo_out_strategy_matmul = (transformer_layout(("dp", "cp", "interleaved_parallel", "mp"), "None"),)
            layer.self_attention.dense.shard(strategy_matmul=wo_strategy_matmul, strategy_bias=wo_strategy_bias,
                                             strategy_activation=wo_strategy_activation,
                                             out_strategy_matmul=wo_out_strategy_matmul)
            layer.add1.shard((transformer_layout("dp", ("cp", "interleaved_parallel", "mp"), "None"),
                              transformer_layout("dp", ("cp", "interleaved_parallel", "mp"), "None")))
            layer.add2.shard((transformer_layout("dp", ("cp", "interleaved_parallel", "mp"), "None"),
                              transformer_layout("dp", ("cp", "interleaved_parallel", "mp"), "None")))
            layer.post_attention_layernorm.norm.shard(
                (transformer_layout("dp", ("cp", "interleaved_parallel", "mp"), "None"),
                 transformer_layout("None")))
            ffn_strategy_matmul_w1 = (transformer_layout(("dp", "cp", "interleaved_parallel"), "None"),
                                      transformer_layout("mp", "None"))
            ffn_strategy_matmul_w3 = (transformer_layout(("dp", "cp", "interleaved_parallel"), "None"),
                                      transformer_layout("mp", "None"))
            ffn_strategy_activation_w1 = (transformer_layout(("dp", "cp", "interleaved_parallel"), "mp"),)
            ffn_strategy_matmul_w2 = (transformer_layout(("dp", "cp", "interleaved_parallel"), "mp"),
                                      transformer_layout("None", "mp"))
            ffn_out_strategy_matmul_w2 = (transformer_layout(("dp", "cp", "interleaved_parallel", "mp"), "None"),)
            if not self.config.qkv_concat:
                layer.mlp.dense_left.shard(strategy_matmul=ffn_strategy_matmul_w1, strategy_bias=None,
                                           strategy_activation=None, out_strategy_matmul=None)
                layer.mlp.dense_4h_to_h.shard(strategy_matmul=ffn_strategy_matmul_w2, strategy_bias=None,
                                              strategy_activation=None, out_strategy_matmul=ffn_out_strategy_matmul_w2)
                layer.mlp.dense_left.activation.shard(ffn_strategy_activation_w1)
                layer.mlp.dense_right.shard(strategy_matmul=ffn_strategy_matmul_w3, strategy_bias=None,
                                            strategy_activation=None, out_strategy_matmul=None)
                layer.mlp.mul.shard((transformer_layout("dp", ("cp", "interleaved_parallel"), "mp"),
                                     transformer_layout("dp", ("cp", "interleaved_parallel"), "mp")))

    def construct(self,
                  hidden_states,
                  attention_mask,
                  rotary_pos_emb,
                  batch_valid_length=None,
                  prefix_key_values=None,
                  block_tables=None,
                  slot_mapping=None):
        """Forward process of the transformer."""
        # hidden_states -> (bs, seq_len, hs)
        # attention_mask -> (bs, 1, seq_len, seq_len)
        # rotary_pos_emb: first: (sen length, kv_channels//2, 2)， after:[1, kv_channels // 2, 2]

        if batch_valid_length is not None and isinstance(self.pre_seq_len, int):
            batch_valid_length = batch_valid_length + self.pre_seq_len

        for i in range(self.num_layers):
            prefix_key_value = None
            if prefix_key_values is not None:
                prefix_key_value = prefix_key_values[i]
            layer = self.layers[i]

            hidden_states = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                batch_valid_length=batch_valid_length,
                prefix_key_value=prefix_key_value,
                block_tables=block_tables,
                slot_mapping=slot_mapping
            )

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = self.cast(hidden_states, self.compute_dtype)

        return hidden_states
