import torch.nn as nn
import pdb
import torch
import numpy as np
from numpy import *
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
import copy
from copy import deepcopy

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Transformer(nn.Module):

    def __init__(self, output_channel=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 numbins=8, max_num_nodes=20, use_one=False, pos_enc=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(output_channel, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(output_channel)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.output_channel = output_channel
        self.nhead = nhead
        self.numbins = numbins
        self.max_num_nodes = max_num_nodes
        self.up_dimension = MLP(input_dim = 5, hidden_dim = 16, output_dim = 32, num_layers = 2)
        self.use_one = use_one
        self.relu = nn.ReLU()
        self.pos_enc = pos_enc

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def positional_encoding(self, pos_type, src):
        bs, n, output_channel = src.shape
        if pos_type == "z":
            pos_embed = (torch.linspace(-5,3,self.numbins+1)+8/(self.numbins*2))[:-1].to(device=src.device)
            pos_embed = pos_embed[(...,)+(None,)*2]/100
        elif pos_type == "sincos":
            position = torch.arange(self.numbins).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, output_channel, 2) * (-math.log(10000.0) / output_channel))
            pe = torch.zeros(self.numbins, 1, output_channel)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
            pos_embed = pe[:self.numbins].to(device=src.device)
        return pos_embed

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxels'], data_dict['voxel_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_unq_coords'], data_dict['v_unq_inv'], data_dict['v_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.numbins, voxels.shape[2]))
        voxel_norm = voxels.sum(1) / data_dict['voxel_num_points'].unsqueeze(1)

        src[unq_inv, voxel_coords[:, 1]] = voxel_norm
        occupied_mask = unq_cnt >=2 

        return src, occupied_mask


    def forward(self,query_embed, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        src = self.up_dimension(src)
        bs, n, c = src.shape
        if self.pos_enc:
            pos_embed = self.positional_encoding(self.pos_enc, src)
        else:
            pos_embed = torch.zeros_like(src)
            pos_embed = pos_embed.permute(1, 0, 2)
        src = src.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, src, memory_key_padding_mask=None,
            pos=pos_embed, query_pos=query_embed)
        hs = self.relu(hs)
        memory = torch.zeros_like(src)
        return hs.transpose(1, 2).squeeze(0).squeeze(1).contiguous(), occupied_mask
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

def attention(query, key,  value):
    dim = query.shape[1]
    scores_1 = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    scores_2 = torch.einsum('abcd, aced->abcd', key, scores_1)
    prob = torch.nn.functional.softmax(scores_2, dim=-1)
    output = torch.einsum('bnhm,bdhm->bdhn', prob, value)
    return output, prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(merge) for _ in range(3)])
        self.down_mlp = MLP(input_dim = self.dim, hidden_dim = 32, output_dim = 1, num_layers = 1)


    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # pdb.set_trace()
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        x = self.down_mlp(x)
        return x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiHeadedAttention(nhead, d_model)

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

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).permute(1,2,0),
                                key=self.with_pos_embed(memory, pos).permute(1,2,0),
                                value=memory.permute(1,2,0))
        tgt2 = tgt2.permute(2,0,1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
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

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args, numbins):
    return Transformer(
        output_channel=args.output_channel,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        numbins=numbins,
        pos_enc=args.get('pos_enc',False)
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
