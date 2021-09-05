# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
#from multi_head_for_len_att import MultiHeadAttention_for_len
from .multiheadattn_origin import MultiHeadAttention
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import gin

@gin.configurable
class Transformer(nn.Module):

    def __init__(self, d_model, nhead, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        # N, 1100, 512
        bs, h, c = src.shape
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        # memory = self.encoder(src, pos=pos_embed)
        memory = src
        # print(f"memory shape{memory.shape}")
        hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                          pos=pos_embed, query_pos=query_embed)
        return hs#transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h)



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self,
                tgt,
                memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for count, layer in enumerate(self.layers):
            output = layer(count,output,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos,
                           query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output#.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        #self.multihead_attn = nn.multiheadattention(d_model, nhead, dropout=dropout)
        #self.multi_head_for_len_att = MultiHeadAttention_for_len(d_model, 100, 200, 1100, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, count, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # print(f"tgt.shape: {tgt.shape}")
        # print(f"query_pos.shape: {query_pos.shape}")
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # print(tgt.shape)
        tgt2 = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory)
        if count != 5:
            tgt2 = torch.sum(tgt2, dim=2).permute(1,0,2)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt
        else:
            tgt = self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = self.dropout3(tgt)
            tgt = self.norm3(tgt)

            return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, count, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(count, tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer_decoder():
    return Transformer()


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == "__main__":

    config_file = "./test/transformer_decoder_test.gin"
    gin.parse_config_file(config_file)

    trans = build_transformer_decoder()
    # self, src, tgt, query_embed, pos_embed

    src = torch.ones((3, 1100, 512))
    tgt = torch.ones((3, 50, 512))
    query_embed = torch.ones((50, 512))
    post_embed = torch.ones((3, 1100, 512))

    out = trans(src, tgt, query_embed, post_embed)

    print(f"out.shape: {out.shape}")
