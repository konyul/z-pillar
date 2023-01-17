
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn

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

class MLP(nn.Module):
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
 
def build_mlp(model_cfg, num_bins, model_name='MLP'):
    model_dict = {
        'MLP': MLP,
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
