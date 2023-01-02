
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch as t_b

class attn(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x.squeeze(-1)

class bin_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class ZAXIS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins,
                 max_num_nodes,
                 bin_loss,
                 num_occupied_bin,
                 use_emb,
                 use_hierachy):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins
        self.max_num_nodes=max_num_nodes
        self.bin_loss = bin_loss
        self.num_occupied_bin = num_occupied_bin
        self.use_emb = use_emb
        self.use_hierachy = use_hierachy
        if self.use_hierachy:
            bin_shuffle_list = []
            for num_bin in self.use_hierachy:
                bin_shuffle_model = bin_shuffle((self.in_channels)*num_bin, out_channels)
                bin_shuffle_list.append(bin_shuffle_model)
            self.bin_shuffle = nn.Sequential(*bin_shuffle_list)
        else:
            self.bin_shuffle = bin_shuffle((self.in_channels)*num_bins, out_channels)

        if self.use_emb:
            self.emb1 = nn.Parameter(torch.Tensor(1, in_channels))
            nn.init.normal_(self.emb1, mean=0., std=.02)
        if self.bin_loss:
            self.attention = attn(self.in_channels)

    def z_vfe(self, v_feat, min_val=None, max_val=None):
        z_axis = v_feat[:,:,2]
        occupied_mask=(z_axis>min_val)*(z_axis<max_val)
        #mask_1_ = torch.clamp(z_axis, min=min_val, max=max_val)
        mask_1_ = (z_axis>min_val)*(z_axis<max_val)
        mask_1 = mask_1_.unsqueeze(-1).repeat(1,1,5)
        voxel_features_bin1 = (v_feat * mask_1)
        voxel_num_points = mask_1_.sum(-1)
        points_mean = voxel_features_bin1[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(v_feat)
        points_mean = points_mean / normalizer
        occupied_mask = occupied_mask.sum(-1).bool()
        return points_mean, occupied_mask
    
    def z_coord(self,v_feat,min_val=None,max_val=None):
        num_pillars = v_feat.shape[0]
        z_coord = torch.zeros((num_pillars, 1)).to(device=v_feat.device)
        z_coord[:] = max_val-.5
        return z_coord
    
    def binning(self, x, num_bins, bin_idx):
        bin_size = float(8/num_bins)
        v_feat_list = [self.z_vfe(x, -5+(bin_size*i),-5+bin_size*(i+1)) for i in range(num_bins)]
        v_feat_bin = list(map(lambda x: x[0], v_feat_list))
        v_feat_mask = list(map(lambda x: x[1], v_feat_list))
        x = torch.stack((v_feat_bin),dim=1)
        if self.num_occupied_bin:
            mask = torch.stack((v_feat_mask),dim=1)
            mask = mask.sum(-1)
            occupied_mask = (mask>=self.num_occupied_bin)
        else:
            occupied_mask = None
        if self.use_emb:
            mask = torch.stack((v_feat_mask),dim=1)
            x[~mask] = self.emb1
        if self.bin_loss:
            attn = self.attention(x)
        else:
            attn = None
        # x = torch.cat((x,attn.unsqueeze(-1).contiguous()),dim=-1)
        N,P,C = x.shape
        x = x.view(N,P*C)
        if self.num_occupied_bin:
            x = x[occupied_mask]
        if self.use_hierachy:
            x = self.bin_shuffle[bin_idx](x)
        else:
            x = self.bin_shuffle(x)
        return x, attn, occupied_mask

    def forward(self, x, idx, batch_dict):
        x = x[:,1:]
        idx, batch_idx = torch.sort(idx)
        x = x[batch_idx]
        x = t_b(x, idx, max_num_nodes=self.max_num_nodes)[0]
        if self.bin_loss:
            z_coord = [self.z_coord(x, -5+i,-5+(i+1)) for i in range(self.num_bins)]
            z_coord = torch.stack((z_coord),dim=1)
        else:
            z_coord = None
        x_list = []
        if self.use_hierachy:
            for bin_idx in range(len(self.use_hierachy)):
                temp_x, temp_attn, temp_occupied_mask = self.binning(x, self.use_hierachy[bin_idx], bin_idx)
                if bin_idx == 0:
                    attn = temp_attn
                    occupied_mask = temp_occupied_mask
                x_list.append(temp_x)
            stacked_x = torch.stack(x_list,dim=0)
            x = torch.sum(stacked_x, dim=0)
        else:
            x, attn, occupied_mask = self.binning(x, self.num_bins, 0)
        return x, z_coord, attn, occupied_mask
 
def build(model_cfg, bin_loss, model_name='ZAXIS'):
    model_dict = {
        'ZAXIS': ZAXIS,
    }
    num_raw_point_features = model_cfg.num_raw_point_features
    num_filter = model_cfg.num_filter
    num_bins = model_cfg.num_bins
    model_class = model_dict[model_name]
    max_num_nodes = model_cfg.max_num_nodes
    num_occupied_bin = model_cfg.get("num_occupied_bins", False)
    use_emb = model_cfg.get("use_emb", False)
    use_hierachy = model_cfg.get("use_hierachy", False)

    model = model_class(num_raw_point_features,
                        num_filter,
                        num_bins,
                        max_num_nodes,
                        bin_loss,
                        num_occupied_bin,
                        use_emb,
                        use_hierachy
                        )
    return model
