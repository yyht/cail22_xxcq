# -*- coding: utf-8 -*-
"""
GlobalPointer参考: https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/models/GlobalPointer.py
稀疏多标签交叉熵损失参考: bert4keras源码
"""
import torch
import torch.nn as nn
import numpy as np
import torch
import io
import torch.nn.functional as F
import random
import numpy as np
import torch.nn as nn
import logging
import warnings

def sparse_multilabel_categorical_crossentropy(y_true=None, y_pred=None, mask_zero=False):
    '''
    稀疏多标签交叉熵损失的torch实现
    '''
    shape = y_pred.shape
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))
    zeros = torch.zeros_like(y_pred[...,:1])
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        infs = zeros + 1e12
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = torch.mean(torch.sum(pos_loss + neg_loss, dim=1))
    return loss

def multilabel_categorical_crossentropy(y_true, y_pred, threshold=0.0):
    y_pred = (1 - 2 * y_true) * y_pred
    
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    threshold = torch.ones_like(y_pred[..., :1]) * threshold
    y_pred_neg = torch.cat([y_pred_neg, threshold], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, -threshold], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return neg_loss + pos_loss

class RawGlobalPointer(nn.Module):
    def __init__(self, hiddensize, ent_type_size, inner_dim, RoPE=True, tril_mask=True):
        '''
        :param encoder: BERT
        :param ent_type_size: 实体数目
        :param inner_dim: 64
        '''
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hiddensize
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE
        self.trail_mask = tril_mask

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, context_outputs,  attention_mask):
        self.device = attention_mask.device
        last_hidden_state = context_outputs[0]
        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        # 排除下三角
        if self.trail_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense = nn.Linear(hidden_size, heads * 2 * head_size, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(
            0, inputs.size(1), dtype=inputs.dtype, device=inputs.device
        )

        indices = torch.arange(
            0, output_dim // 2, dtype=inputs.dtype, device=inputs.device
        )
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(
            1, 2
        )
        return embeddings[None, :, :]

    def forward(self, context_outputs, attention_mask=None):
        
        self.device = attention_mask.device
        inputs = context_outputs[0]
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        
        inputs = self.dense(inputs)
        bs, seqlen = inputs.shape[:2]

        # method 1
        inputs = inputs.reshape(bs, seqlen, self.heads, 2, self.head_size)
        qw, kw = inputs.unbind(axis=-2)

        # method 2
        # inputs = inputs.reshape(bs, seqlen, self.heads, 2 * self.head_size)
        # qw, kw = inputs.chunk(2, axis=-1)

        # original
        # inputs = inputs.chunk(self.heads, axis=-1)
        # inputs = torch.stack(inputs, axis=-2)
        # qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]

        # RoPE编码
        if self.RoPE:
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., None, 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., None, ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1).reshape_as(qw)

            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * 1e12

        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """

    def __init__(
        self,
        hidden_size,
        heads=12,
        head_size=64,
        RoPE=True,
        use_bias=True,
        tril_mask=True,
    ):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.tril_mask = tril_mask
        self.dense1 = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.dense2 = nn.Linear(head_size * 2, heads * 2, bias=use_bias)

    def get_rotary_positions_embeddings(self, inputs, output_dim):
        position_ids = torch.arange(
            inputs.size(1), dtype=inputs.dtype, device=inputs.device
        )

        indices = torch.arange(
            output_dim // 2, dtype=inputs.dtype, device=inputs.device
        )
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum("n,d->nd", position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], axis=-1).flatten(
            1, 2
        )
        return embeddings[None, :, :]

    def forward(self, context_outputs, attention_mask=None):
        
        self.device = attention_mask.device
        inputs = context_outputs[0]
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        
        inputs = self.dense1(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = self.get_rotary_positions_embeddings(inputs, self.head_size)
            cos_pos = torch.repeat_interleave(pos[..., 1::2], 2, axis=-1)
            sin_pos = torch.repeat_interleave(pos[..., ::2], 2, axis=-1)

            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], axis=-1).reshape_as(qw)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], axis=-1).reshape_as(kw)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算内积
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.head_size ** 0.5
        bias = self.dense2(inputs).transpose(1, 2) / 2  # 'bnh->bhn'
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]

        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        # 排除下三角
        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)

            logits = logits - mask * 1e12

        return logits
    
# class Biaffine(nn.Module):
#     def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
#         super().__init__()
#         self.bias_x = bias_x
#         self.bias_y = bias_y
#         self.out_size = out_size
#         self.U = torch.nn.Parameter(
#             torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y))
#         )

#     def forward(self, x, y):
#         if self.bias_x:
#             x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
#         if self.bias_y:
#             y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

#         bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
#         return bilinar_mapping
    
# class BiaffineFlatten(nn.Module):
#     def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
#         super().__init__()
#         self.bias_x = bias_x
#         self.bias_y = bias_y
#         self.out_size = out_size
#         self.U = torch.nn.Parameter(
#             torch.randn(in_size + int(bias_x), out_size, in_size + int(bias_y))
#         )

#     def forward(self, x, y):
#         if self.bias_x:
#             x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
#         if self.bias_y:
#             y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

#         bilinar_mapping = torch.einsum('bxi,ioj,bxj->bxo', x, self.U, y)
#         return bilinar_mapping
    
def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = sequence_tensor.shape
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    
    device = sequence_tensor.device

    flat_offsets = torch.reshape(
      torch.arange(0, batch_size) * seq_length, [-1, 1]).to(device)

    flat_positions = torch.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = torch.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
    output_tensor = flat_sequence_tensor.index_select(0, flat_positions)
    output_tensor = torch.reshape(output_tensor, [batch_size, -1, width])
    return output_tensor

class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        return s


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, tril_mask=False,
                 dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.dropout = nn.Dropout(dropout)
        self.cls_num = cls_num
        self.tril_mask = tril_mask

    def forward(self, x, attention_mask=None):
        
        inputs = x[0]
        batch_size = inputs.size()[0]
        seq_len = inputs.size()[1]
        
        h = self.dropout(self.mlp1(inputs))
        t = self.dropout(self.mlp2(inputs))
        # [batch, cls_num, seq_len, seq_len]
        logits = self.biaffine(h, t)
        
        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.cls_num, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12
        
        # 排除padding
        if attention_mask is not None:  # huggingface's attention_mask
            attn_mask = (
                1 - attention_mask[:, None, None, :] * attention_mask[:, None, :, None]
            )
            logits = logits - attn_mask * 1e12

        if self.tril_mask:
            # 排除下三角
            mask = torch.tril(torch.ones_like(logits), diagonal=-1)
            logits = logits - mask * 1e12
        
        return logits




        