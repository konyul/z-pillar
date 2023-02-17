
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
import time


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
        self.encoder_levels = config.encoder_level
        self.input_channel = config.input_channel
        self.num_bins = config.num_bins
        self.output_channel = config.output_channel
        for encoder_level in self.encoder_levels:
            bin_shuffle_list.append(bin_shuffle(self.input_channel*encoder_level*self.num_bins, self.output_channel*encoder_level))
            conv_list.append(Conv1d(8, self.output_channel*encoder_level))
        self.bin_shuffle_list = nn.Sequential(*bin_shuffle_list)
        self.conv_list = nn.Sequential(*conv_list)
        self.grid_size = config.grid_size
        self.scale_xy = self.grid_size[0] * self.grid_size[1]
        self.scale_y = self.grid_size[1]
        self.voxel_size = config.voxel_size
        self.point_cloud_range = config.point_cloud_range
        self.scale_xyz = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.scale_yz = self.grid_size[1] * self.grid_size[2]
        self.scale_z = self.grid_size[2]
        
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
        points_indices_inv = data_dict['points_indices_inv']
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        if 'points_indices' in data_dict:
            points_indices = data_dict['points_indices']
            point_coords[:,0], point_coords[:,1] = points_indices[:, 2], points_indices[:, 1]    
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                    point_coords[:, 0] * self.scale_yz + \
                    point_coords[:, 1] * self.scale_z + \
                    point_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        idx = torch_scatter.scatter_mean(points_indices_inv.cuda(), unq_inv, dim=0)
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        data_dict['voxel_features'] = points_mean.contiguous()
        data_dict['voxel_features_coords'] = voxel_coords.contiguous()
        data_dict['unq_idx'] = idx
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
    
    def point_sum_sparse(self, data_dict, downsample_level):
        points = data_dict['points_with_f_center']
        unq_inv = data_dict['unq_inv']
        match_index = data_dict['sparse_input'].__dict__['indice_dict']['spconv'+str(downsample_level)].__dict__['pair_bwd']
        sparse_feat = data_dict['sparse_input']._features
        sparse_indices = data_dict['sparse_input'].indices
        converted_pillar_merge_coords =  match_index[:,unq_inv].permute(1,0).long()
        flatten_converted_pillar_merge_coords = converted_pillar_merge_coords.reshape(-1)
        mask = (flatten_converted_pillar_merge_coords!=-1)
        ori_cnt = (converted_pillar_merge_coords!=-1).sum(-1)
        new_sparse_feat = sparse_feat[flatten_converted_pillar_merge_coords][mask]
        points_indices_inv = torch.arange(0,sparse_feat.shape[0]).int()
        points_indices_inv = points_indices_inv[flatten_converted_pillar_merge_coords][mask]
        new_points = torch.repeat_interleave(points, ori_cnt, dim=0)
        points_indices = sparse_indices[flatten_converted_pillar_merge_coords][mask]
        output = new_sparse_feat + self.conv_list[self.encoder_levels.index(downsample_level)](new_points[:,1:])
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([new_points[:,:1],new_points[:,4:]],dim=1)
        data_dict['points_indices'] = points_indices
        data_dict['points_indices_inv'] = points_indices_inv
        return data_dict
    
    def point_sum_sparse(self, data_dict, downsample_level):
        points = data_dict['points_with_f_center']
        unq_inv = data_dict['unq_inv']
        sparse_indices = data_dict['sparse_input'].indices
        match_index = data_dict['sparse_input'].__dict__['indice_dict']['spconv'+str(downsample_level)].__dict__['pair_bwd']
        sparse_feat = data_dict['sparse_input']._features
        unq_inv =  match_index[:,unq_inv].permute(1,0).max(axis=-1)[0].long()
        output = sparse_feat[unq_inv, :] + self.conv_list[self.encoder_levels.index(downsample_level)](points[:,1:])
        points_indices_inv = torch.arange(0,sparse_feat.shape[0]).int()
        points_indices_inv = points_indices_inv[unq_inv]
        points_indices = sparse_indices[unq_inv]
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([points[:,:1],points[:,4:]],dim=1)
        data_dict['points_indices_inv'] = points_indices_inv
        data_dict['points_indices'] = points_indices
        return data_dict

    def forward(self, sparse_input, data_dict, downsample_level=False, zbam_config=None):
        data_dict['sparse_input'] = sparse_input
        if downsample_level > 1:
            data_dict = self.point_sum_sparse(data_dict, downsample_level)
        else:
            data_dict = self.point_sum_subm(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask, unq_idx = self.binning(data_dict)
        src = src[occupied_mask]
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle_list[self.encoder_levels.index(downsample_level)](src)
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