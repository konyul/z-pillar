
import argparse
import copy
import math
from easydict import EasyDict

import torch
import cv2
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch_scatter

from .cbam import ZBAM

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

class Conv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return self.relu(x)

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
        self.zbam = ZBAM(64)
        self.conv = Conv1d(8, output_channel)
        
    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        v_feat_coords = voxel_coords[:, 0] * data_dict['scale_xy'] + voxel_coords[:, 3] * data_dict['scale_y'] + voxel_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask
    
    def dyn_voxelization(self, data_dict):
        points = data_dict['points_sorted']
        points_data = data_dict['points_feature']
        point_coords = torch.floor((points[:, 1:4] - data_dict['point_cloud_range'][0:3]) / data_dict['voxel_size']).int()
        merge_coords = points[:, 0].int() * data_dict['scale_xyz'] + \
                        point_coords[:, 0] * data_dict['scale_yz'] + \
                        point_coords[:, 1] * data_dict['scale_z'] + \
                        point_coords[:, 2]
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        import pdb;pdb.set_trace()
        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // data_dict['scale_xyz'],
                                    (unq_coords % data_dict['scale_xyz']) // data_dict['scale_yz'],
                                    (unq_coords % data_dict['scale_yz']) // data_dict['scale_z'],
                                    unq_coords % data_dict['scale_z']), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        data_dict['voxel_features'] = points_mean.contiguous()
        data_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return data_dict

    def point_sum(self, data_dict):
        points = data_dict['pointsv2']
        points_coords_3d = torch.floor((points[:, 4:7] - data_dict['point_cloud_range'][0:3]) / data_dict['voxel_size']).int()
        points_coords = points_coords_3d[:,:2]
        pillar_merge_coords = points[:, 0].int() * data_dict['scale_xy'] + \
                       points_coords[:, 0] * data_dict['scale_y'] + \
                       points_coords[:, 1]
        sparse_feat = data_dict['sparse_input']._features
        pillar_merge_coords_sorted, idx = torch.sort(pillar_merge_coords)
        points_sorted = points[idx]
        _, inv = torch.unique(pillar_merge_coords_sorted, return_inverse=True)
        sparse_feat = sparse_feat[inv]
        output = sparse_feat + self.conv(points_sorted[:,1:])
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([points_sorted[:,:1],points_sorted[:,4:]],dim=1)
        return data_dict
    
    def downsampling(self, sparse_input, data_dict):
        downsample_rate = sparse_input.spatial_shape[0] / data_dict['grid_size'][0]
        data_dict['voxel_size'][:2] = data_dict['voxel_size'][:2]/downsample_rate
        data_dict['grid_size'][:2] = data_dict['grid_size'][:2] * downsample_rate
        data_dict['scale_xyz'] = data_dict['grid_size'][0]*data_dict['grid_size'][1]*data_dict['grid_size'][2]
        data_dict['scale_yz'] = data_dict['grid_size'][1]*data_dict['grid_size'][2]
        data_dict['scale_z'] = data_dict['grid_size'][2]
        data_dict['scale_xy'] = data_dict['grid_size'][0]*data_dict['grid_size'][1]
        data_dict['scale_y'] = data_dict['grid_size'][1]
        return data_dict

    def forward(self, sparse_input, data_dict):
        data_dict['sparse_input'] = sparse_input
        data_dict = self.downsampling(sparse_input, data_dict)
        data_dict = self.point_sum(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask = self.binning(data_dict)

        src = src[occupied_mask].permute(0, 2, 1)
        src = self.zbam(src)
        src = src.max(2)[0]
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
