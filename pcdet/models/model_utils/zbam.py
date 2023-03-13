
import argparse
import copy
import math
from easydict import EasyDict

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch_scatter
from torch_scatter import scatter_add
class Conv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn=bn
        if self.bn==True:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear(x)
        if self.bn==True:
            x = self.norm(x)
        return self.relu(x)


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
                 config):
        super().__init__()
        bin_shuffle_list = []
        conv_list = []
        linear_list = []
        self.encoder_levels = config.encoder_level
        self.input_channel = config.input_channel
        self.feat_channel = config.feat_channel
        self.num_bins = config.num_bins
        self.output_channel = config.output_channel
        for encoder_level in self.encoder_levels:
            out_channel = self.output_channel*(2**(encoder_level-1))
            in_channel = self.feat_channel*self.num_bins
            if encoder_level != 1:
                linear_list.append(Conv1d(self.input_channel*(2**(encoder_level-1)), self.feat_channel, bn=False))
            bin_shuffle_list.append(bin_shuffle(in_channel, out_channel))
            conv_list.append(Conv1d(8, out_channel))
        self.bin_shuffle_list = nn.Sequential(*bin_shuffle_list)
        self.conv_list = nn.Sequential(*conv_list)
        self.linear_list = nn.Sequential(*linear_list)
        self.grid_size = torch.tensor(config.grid_size).cuda()
        self.scale_xy = self.grid_size[0] * self.grid_size[1]
        self.scale_y = self.grid_size[1]
        self.voxel_size = torch.tensor(config.voxel_size).cuda()
        self.point_cloud_range = torch.tensor(config.point_cloud_range).cuda()
        self.scale_xyz = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.scale_yz = self.grid_size[1] * self.grid_size[2]
        self.scale_z = self.grid_size[2]
        self.nsamples = 1
        
    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        v_feat_coords = voxel_coords[:, 0] * self.scale_xy + voxel_coords[:, 3] * self.scale_y + voxel_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
        if 'unq_idx' in data_dict:
            unq_idx = data_dict['unq_idx']
            merged_coords = torch.cat([v_feat_coords[:,None],unq_idx[:,None]],dim=-1)
            unq_coords, unq_inv, unq_cnt = torch.unique(merged_coords, return_inverse=True, return_counts=True, dim=0)
            unq_idx = unq_coords[:,1]
        else:
            unq_idx = None
        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        return src, occupied_mask, unq_idx
    
    def dyn_voxelization(self, data_dict):
        points = data_dict['points_sorted']
        points_data = data_dict['points_feature']
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        if 'points_indices' in data_dict:
            points_indices = data_dict['points_indices']
            point_coords[:,0], point_coords[:,1] = points_indices[:, 2], points_indices[:, 1]    
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                    point_coords[:, 0] * self.scale_yz + \
                    point_coords[:, 1] * self.scale_z + \
                    point_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)
        if 'idx_inv' in data_dict:
            idx_inv = data_dict['idx_inv']
            idx = torch_scatter.scatter_mean(idx_inv.cuda(), unq_inv, dim=0)
            data_dict['unq_idx'] = idx
        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        data_dict['voxel_features'] = points_mean.contiguous()
        data_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return data_dict

    def point_sum_subm(self, data_dict):
        points = data_dict['points_with_f_center']
        pillar_merge_coords = data_dict['pillar_merge_coords']
        sparse_feat = data_dict['sparse_input']._features
        pillar_merge_coords_sorted, idx = torch.sort(pillar_merge_coords)
        points_sorted = points[idx]
        _, inv = torch.unique(pillar_merge_coords_sorted, return_inverse=True)
        sparse_feat = sparse_feat[inv]
        output = sparse_feat + self.conv_list[0](points_sorted[:,1:])
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([points_sorted[:,:1],points_sorted[:,4:]],dim=1)
        return data_dict
    
    def to_dense_batch(self, x, pillar_idx, max_num_nodes, max_pillar_idx):
        r"""
        Point sampling according to pillar index with constraint amount
        """
        
        # num_points in pillars (0 for empty pillar)
        num_nodes = scatter_add(pillar_idx.new_ones(x.size(0)), pillar_idx, dim=0,
                                dim_size=max_pillar_idx)
        cum_nodes = torch.cat([pillar_idx.new_zeros(1), num_nodes.cumsum(dim=0)])

        # check if num_points in pillars exceed the predefined num_points value
        filter_nodes = False
        if num_nodes.max() > max_num_nodes:
            filter_nodes = True
        tmp = torch.arange(pillar_idx.size(0), device=x.device) - cum_nodes[pillar_idx]
        if filter_nodes:
            mask = tmp < max_num_nodes
            x = x[mask]
            pillar_idx = pillar_idx[mask]
        return x, pillar_idx

    def point_sum_sparse(self, data_dict, downsample_level):    
        points = data_dict['points_with_f_center']
        unq_inv = data_dict['unq_inv'].long().cuda()
        sparse_feat = data_dict['sparse_input']._features
        sparse_indices = data_dict['sparse_input'].indices
        max_pillar_idx = sparse_indices.shape[0]
        points, unq_inv = self.to_dense_batch(points, unq_inv, max_num_nodes=20, max_pillar_idx=max_pillar_idx)
        expand_mask = unq_inv.clone()
        idx_inv = unq_inv.clone()
        n_sparse_feat = sparse_feat[expand_mask]
        points_indices = sparse_indices[expand_mask]
        output = n_sparse_feat + self.conv_list[self.encoder_levels.index(downsample_level)](points[:,1:])
        data_dict['unq_inv'] = idx_inv
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([points[:,:1],points[:,4:]],dim=1)
        data_dict['points_indices'] = points_indices
        data_dict['idx_inv'] = idx_inv
        return data_dict

    def forward(self, sparse_input, data_dict, downsample_level=False, zbam_config=None):
        data_dict['sparse_input'] = sparse_input
        if downsample_level > 1:
            data_dict = self.point_sum_sparse(data_dict, downsample_level)
        else:
            data_dict = self.point_sum_subm(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask, unq_idx = self.binning(data_dict)
        if occupied_mask.sum() == 0:
            return sparse_input
        src = src[occupied_mask]
        cur_level = self.encoder_levels.index(downsample_level)
        if downsample_level != 1:
            src = self.linear_list[cur_level-1](src)
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle_list[cur_level](src)
        if unq_idx is not None:
            unq_idx = unq_idx[occupied_mask]
            sparse_input._features[unq_idx] = sparse_input._features[unq_idx] + src
        else:
            sparse_input._features[occupied_mask] = sparse_input._features[occupied_mask] + src
        return sparse_input

def build_zbam(model_cfg):
    model_name = model_cfg.zbam
    model_dict={
        'Zconv': Zconv
    }
    model_class = model_dict[model_name]
    model = model_class(model_cfg
                        )
    return model