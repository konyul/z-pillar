
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import time
from .cbam import ZBAM

class bin_shuffle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, in_channels//2, bias=False),
            nn.BatchNorm1d(in_channels//2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(in_channels//2, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU())

    def forward(self, x):
        return self.conv(x)

class Zconv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins
        self.bin_shuffle = bin_shuffle((self.in_channels)*num_bins, out_channels)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        return src, occupied_mask

class conv1d(nn.Module):
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

class CBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins
        self.zbam = ZBAM(out_channels)
        self.up_dimension = conv1d(input_dim = 5, hidden_dim = int(out_channels/2), output_dim = out_channels, num_layers = 2)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        src = src.max(2)[0]
        return src, occupied_mask


class ZcCBAM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bins = num_bins
        self.zbam = ZBAM(out_channels)
        self.bin_shuffle = bin_shuffle((self.in_channels)*num_bins, out_channels)
        self.up_dimension = conv1d(input_dim = 5, hidden_dim = int(out_channels/2), output_dim = out_channels, num_layers = 2)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        unq_coords, unq_inv, unq_cnt = data_dict['v_feat_unq_coords'], data_dict['v_feat_unq_inv'], data_dict['v_feat_unq_cnt']

        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask

    def forward(self, data_dict):
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        src = self.up_dimension(src)
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        return src, occupied_mask

def build_mlp(model_cfg, num_bins, model_name='ZCONV'):
    model_dict = {
        'Zconv': Zconv,
        'CBAM': CBAM,
        'ZcCBAM': ZcCBAM
}
    input_channel = model_cfg.input_channel
    output_channel = model_cfg.output_channel
    num_bins = num_bins
    model_class = model_dict[model_name]

    model = model_class(input_channel,
                        output_channel,
                        num_bins
                        )
    return model
