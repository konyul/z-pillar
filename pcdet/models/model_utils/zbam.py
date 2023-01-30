
import argparse
import copy
import math

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch_scatter

class conv(nn.Module):
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
                point_channel,
                feat_channel,
                output_channel,
                num_bins):
        super().__init__()
        self.point_channel = point_channel
        self.feat_channel = feat_channel
        self.output_channel = output_channel
        self.num_bins = num_bins
        self.intermediate_channel = point_channel + feat_channel
        self.conv = conv(self.intermediate_channel, point_channel)
        self.channel_attention = nn.Linear(num_bins, 1)
        self.spatial_attention = nn.Linear(point_channel, 1)
        self.bin_shuffle = conv((point_channel)*num_bins, output_channel)

    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        grid_size = data_dict['grid_size']
        scale_xy = grid_size[0] * grid_size[1] * grid_size[2]
        scale_y = grid_size[1] * grid_size[2]
        v_feat_coords = voxel_coords[:, 0] * scale_xy + voxel_coords[:, 3] * scale_y + voxel_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask
    
    def dyn_voxelization(self, data_dict):
        points = data_dict['points_sorted']
        points_data = data_dict['points_feature']
        point_cloud_range = data_dict['point_cloud_range']
        voxel_size = data_dict['voxel_size']
        grid_size = data_dict['grid_size']
        scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        scale_yz = grid_size[1] * grid_size[2]
        scale_z = grid_size[2]
        point_coords = torch.floor((points[:, 1:4] - point_cloud_range[0:3]) / voxel_size).int()
        merge_coords = points[:, 0].int() * scale_xyz + \
                        point_coords[:, 0] * scale_yz + \
                        point_coords[:, 1] * scale_z + \
                        point_coords[:, 2]
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        data_dict['voxel_features'] = points_mean.contiguous()
        data_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return data_dict

    def point_concat(self, data_dict):
        points = data_dict['points']
        pillar_merge_coords = data_dict['pillar_merge_coords']
        sparse_feat = data_dict['sparse_input']._features
        pillar_merge_coords_sorted, idx = torch.sort(pillar_merge_coords)
        points_sorted = points[idx]
        _, inv = torch.unique(pillar_merge_coords_sorted, return_inverse=True)
        sparse_feat = sparse_feat[inv]
        output_feature = torch.cat([points_sorted[:,1:], sparse_feat],dim=1)
        output = self.conv(output_feature)
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = points_sorted
        return data_dict

    def forward(self, sparse_input, data_dict):
        data_dict['sparse_input'] = sparse_input
        data_dict = self.point_concat(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        src = src * self.channel_attention(src.permute(0,2,1)).permute(0,2,1).contiguous()
        src = src * self.spatial_attention(src)
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        sparse_input._features[occupied_mask] = sparse_input._features[occupied_mask] + src
        return sparse_input
 
def build_zbam(model_cfg):
    point_channel = model_cfg.point_channel
    feat_channel = model_cfg.feat_channel
    output_channel = model_cfg.output_channel
    num_bins = model_cfg.num_bins
    model_name = model_cfg.zbam
    model_dict={
        'MLP': MLP
    }
    model_class = model_dict[model_name]
    model = model_class(point_channel,
                        feat_channel,
                        output_channel,
                        num_bins
                        )
    return model
