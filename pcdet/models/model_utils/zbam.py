
import argparse
import copy
import math
from easydict import EasyDict

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch_scatter
from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils

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
        self.encoder_levels = config.encoder_level
        self.input_channel = config.input_channel
        self.num_bins = config.num_bins
        self.output_channel = config.output_channel
        self.channel_ratio = config.sampling_cfg.get('channel_ratio',1)
        if self.channel_ratio != 1:
            channel = config.input_channel*(2**(config.encoder_level[-1]-1))
            self.linear = Conv1d(channel,int(channel*self.channel_ratio),bn=False)
        for encoder_level in self.encoder_levels:
            out_channel = self.output_channel*(2**(encoder_level-1))
            if encoder_level == 1:
                in_channel = int(self.input_channel*self.num_bins)    
            else:
                in_channel = int(self.input_channel*(2**(encoder_level-1))*self.num_bins*self.channel_ratio)
            bin_shuffle_list.append(bin_shuffle(in_channel, out_channel))
            conv_list.append(Conv1d(8, out_channel))
        self.bin_shuffle_list = nn.Sequential(*bin_shuffle_list)
        self.conv_list = nn.Sequential(*conv_list)
        self.grid_size = torch.tensor(config.grid_size).cuda()
        self.scale_xy = self.grid_size[0] * self.grid_size[1]
        self.scale_y = self.grid_size[1]
        self.voxel_size = torch.tensor(config.voxel_size).cuda()
        self.point_cloud_range = torch.tensor(config.point_cloud_range).cuda()
        self.scale_xyz = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.scale_yz = self.grid_size[1] * self.grid_size[2]
        self.scale_z = self.grid_size[2]
        self.nsamples = 1
        self.point_sample = config.get('sampling_type',False)
        if self.point_sample:
            self.sampling_type = config.sampling_type
            self.num_points = config.sampling_cfg.get('num_points',5000)
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
    
    def FPS(self, points, unq_inv):
        B = (points[:,0].max()+1).int()
        N = 200000
        input = torch.zeros(B,N,3).cuda()
        features = torch.zeros(B,N,10).cuda()
        for i in range(B):
            batch_mask = (points[:,0]==i)
            batch_input = points[batch_mask]
            batch_points = batch_input[:,4:7]
            batch_inv = unq_inv[batch_mask]
            batch_feature = torch.cat([batch_input,batch_inv[:,None]],dim=-1)
            num_points = batch_input.shape[0]
            if num_points > N:
                input[i] = batch_points[:N]
                features[i] = batch_feature[:N]
            else:
                input[i][:num_points] = batch_points
                features[i][:num_points] = batch_feature
        feat_flipped = features.transpose(1,2).contiguous()
        nsamples = self.num_points
        new_var = pointnet2_utils.gather_operation(
                feat_flipped,
                pointnet2_utils.farthest_point_sample(input, nsamples)
            ).transpose(1, 2).contiguous()
        new_var = new_var.reshape(-1,10).contiguous()
        new_points = new_var[:,:-1]
        new_inv = new_var[:,-1]
        return new_points, new_inv.long()
    
    def RS(self, points, unq_inv):
        B = (points[:,0].max()+1).int()
        N = 200000
        input = torch.zeros(B,N,10).cuda()
        for i in range(B):
            batch_mask = (points[:,0]==i)
            batch_input = points[batch_mask]
            batch_inv = unq_inv[batch_mask]
            num_points = batch_input.shape[0]
            if num_points > N:
                input[i] = torch.cat([batch_input[:N],batch_inv[:N,None]],dim=-1)
            else:
                input[i][:num_points] = torch.cat([batch_input,batch_inv[:,None]],dim=-1)
        rand_idx = torch.randint(0,input.shape[1],(self.num_points,))
        input = input[:,rand_idx,:].reshape(-1,10)
        mask = (input.sum(-1)!=0)
        input = input[mask]
        new_points = input[:,:-1]
        new_unq_inv = input[:,-1]
        return new_points, new_unq_inv.long()
    
    def point_gen(self, data_dict, downsample_level):
        points = data_dict['points_with_f_center']
        unq_inv = data_dict['unq_inv']
        unq_inv = unq_inv.long().cuda()
        pair_bwd = data_dict['sparse_input'].__dict__['indice_dict']['spconv'+str(downsample_level)].__dict__['pair_bwd']
        if self.point_sample:
            if self.sampling_type == 'FPS':
                points, unq_inv = self.FPS(points, unq_inv)
            elif self.sampling_type == 'RS':
                points, unq_inv = self.RS(points, unq_inv)
        L_pair_bwd =  pair_bwd[:,unq_inv].permute(1,0).long()
        L_pair_bwd_flat = L_pair_bwd.reshape(-1)
        cnt = (L_pair_bwd!=-1).sum(-1)
        mask = (L_pair_bwd_flat!=-1)
        expand_mask = L_pair_bwd_flat[mask]
        idx_inv = torch.arange(0, pair_bwd.max()+1).int()
        idx_inv = idx_inv[expand_mask]
        n_points = torch.repeat_interleave(points, cnt, dim=0)
        data_dict['expand_mask'] = expand_mask
        data_dict['points_with_f_center'] = n_points
        data_dict['unq_inv'] = idx_inv
        return data_dict
        
    def point_sum_sparse(self, data_dict, downsample_level):
        for level in range(2, downsample_level+1):
            data_dict = self.point_gen(data_dict, level)
        expand_mask = data_dict['expand_mask']
        n_points = data_dict['points_with_f_center']
        idx_inv = data_dict['unq_inv']
        sparse_feat = data_dict['sparse_input']._features
        sparse_indices = data_dict['sparse_input'].indices
        n_sparse_feat = sparse_feat[expand_mask]
        points_indices = sparse_indices[expand_mask]

        output = n_sparse_feat + self.conv_list[self.encoder_levels.index(downsample_level)](n_points[:,1:])
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([n_points[:,:1],n_points[:,4:]],dim=1)
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
        if self.channel_ratio != 1 and downsample_level != 1:
            src = self.linear(src)
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