
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

from .cbam import ZBAM
from .transformer import build_transformer
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

class CBAM(nn.Module):
    def __init__(self,
                 input_channel,
                 feat_channel,
                 output_channel,
                 num_bins):
        super().__init__()
        self.input_channel = input_channel
        self.feat_channel = feat_channel
        self.output_channel = output_channel
        self.num_bins = num_bins
        self.intermediate_channel = input_channel + feat_channel
        self.zbam = ZBAM(output_channel)
        self.conv = Conv1d(8, output_channel)

    def verify_positionv2(self, unq_coords):
        aa_pillar_coords = torch.stack((unq_coords // (1440**2),
                                     (unq_coords % (1440**2)) // (1440),
                                     unq_coords % (1440),
                                     ), dim=1)
        aa_pillar_coords = aa_pillar_coords[:, [0, 2, 1]]
    def binning(self, data_dict):
        voxels, voxel_coords = data_dict['voxel_features'], data_dict['voxel_features_coords'].to(torch.long)
        grid_size = data_dict['grid_size']
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        v_feat_coords = voxel_coords[:, 0] * scale_xy + voxel_coords[:, 3] * scale_y + voxel_coords[:, 2]
        unq_coords, unq_inv, unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
        src = voxels.new_zeros((unq_coords.shape[0], self.num_bins, voxels.shape[1]))
        src[unq_inv, voxel_coords[:, 1]] = voxels
        occupied_mask = unq_cnt >=2 
        if False:
            self.verify_positionv2(unq_coords)
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
        if 'points_indices' in data_dict:
            points_indices = data_dict['points_indices']
            merge_coords = points[:, 0].int() * scale_xyz + \
                        points_indices[:, 2] * scale_yz + \
                        points_indices[:, 1] * scale_z + \
                        point_coords[:, 2]
        else:
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

    def verify_position(self,pillar_merge_coords_sorted,data_dict,points_sorted):
        aa, inv = torch.unique(pillar_merge_coords_sorted, return_inverse=True)
        pillar_idx = torch.stack((aa//(1440**2),(aa%(1440**2))//(1440),aa%(1440)),dim=1)
        points_coords_3d = torch.floor((points_sorted[:, 4:7] - data_dict['point_cloud_range'][0:3]) / data_dict['voxel_size']).int()
        points_coords = points_coords_3d[:,:2]
        sorted_merge_coords = points_sorted[:, 0].int() * (1440**2) + \
                       points_coords[:, 0] * (1440) + \
                       points_coords[:, 1]
        aab, _, _ = torch.unique(sorted_merge_coords, return_inverse=True, return_counts=True, dim=0)
        aab = aab.int()
        aa_pillar_coords = torch.stack((aab // (1440**2),
                                     (aab % (1440**2)) // (1440),
                                     aab % (1440),
                                     ), dim=1)
        aa_pillar_coords = aa_pillar_coords[:, [0, 2, 1]]

    def point_sum_subm(self, data_dict):
        points = data_dict['points_with_f_center']
        pillar_merge_coords = data_dict['pillar_merge_coords']
        sparse_feat = data_dict['sparse_input']._features
        pillar_merge_coords_sorted, idx = torch.sort(pillar_merge_coords)
        points_sorted = points[idx]
        if False:
            self.verify_position(pillar_merge_coords_sorted,data_dict,points_sorted)
        _, inv = torch.unique(pillar_merge_coords_sorted, return_inverse=True)
        sparse_feat = sparse_feat[inv]
        output = sparse_feat + self.conv(points_sorted[:,1:])
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([points_sorted[:,:1],points_sorted[:,4:]],dim=1)
        return data_dict
    
    def point_sum_sparse(self, data_dict):
        points = data_dict['points_with_f_center']
        pillar_merge_coords = data_dict['pillar_merge_coords']
        unq_coords, unq_inv, unq_cnt = torch.unique(pillar_merge_coords, return_inverse=True, return_counts=True, dim=0)
        spconv2 = data_dict['sparse_input'].__dict__['indice_dict']['spconv2']
        ori_to_down = spconv2.__dict__['pair_bwd']
        converted_pillar_merge_coords =  ori_to_down[:,unq_inv].permute(1,0).long()
        sparse_feat = data_dict['sparse_input']._features
        flatten_converted_pillar_merge_coords = converted_pillar_merge_coords.reshape(-1)
        mask = (flatten_converted_pillar_merge_coords!=-1)
        ori_cnt = (converted_pillar_merge_coords!=-1).sum(-1)
        new_sparse_feat = sparse_feat[flatten_converted_pillar_merge_coords][mask]
        new_points = torch.repeat_interleave(points, ori_cnt, dim=0)
        new_converted_pillar_merge_coords = (flatten_converted_pillar_merge_coords)[mask]
        sparse_indices = data_dict['sparse_input'].indices
        points_indices = sparse_indices[new_converted_pillar_merge_coords]
        output = new_sparse_feat + self.conv(new_points[:,1:])
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([new_points[:,:1],new_points[:,4:]],dim=1)
        data_dict['points_indices'] = points_indices
        return data_dict
    
    def point_concat(self, data_dict):
        points = data_dict['points_with_f_center']
        pillar_merge_coords = data_dict['pillar_merge_coords']
        sparse_feat = data_dict['sparse_input']._features
        pillar_merge_coords_sorted, idx = torch.sort(pillar_merge_coords)
        points_sorted = points[idx]
        if False:
            self.verify_position(pillar_merge_coords_sorted,data_dict,points_sorted)
        _, inv = torch.unique(pillar_merge_coords_sorted, return_inverse=True)
        sparse_feat = sparse_feat[inv]
        output = self.conv(torch.cat([points_sorted[:,1:],sparse_feat],dim=-1))
        data_dict['points_feature'] = output
        data_dict['points_sorted'] = torch.cat([points_sorted[:,:1],points_sorted[:,4:]],dim=1)
        return data_dict

    def forward(self, sparse_input, data_dict):
        data_dict['sparse_input'] = sparse_input

        if 'spconv2' in data_dict['sparse_input'].__dict__['indice_dict'].keys():
            data_dict = self.point_sum_sparse(data_dict)
        else:
            data_dict = self.point_sum_subm(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask = self.binning(data_dict)

        src = src[occupied_mask].permute(0, 2, 1)
        src = self.zbam(src)
        src = src.max(2)[0]
        
        sparse_input._features[occupied_mask] = sparse_input._features[occupied_mask] + src
        return sparse_input
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

class feature_conv(nn.Module):
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

class Zconv(CBAM):
    def __init__(self,
                 input_channel,
                 feat_channel,
                 output_channel,
                 num_bins):
        super().__init__(input_channel,
                 feat_channel,
                 output_channel,
                 num_bins)
        self.bin_shuffle = bin_shuffle((input_channel)*num_bins, output_channel)
        self.bin_reduce = nn.Sequential(
            nn.Linear(output_channel*2, output_channel, bias=True),
            nn.ReLU())
        self.bin_shuffle_sparse = bin_shuffle((input_channel)*num_bins, output_channel*2)
        self.zbam = None
        self.feature_conv = feature_conv(input_channel, output_channel*2)

    def voxel_sum_sparse(self, data_dict):
        voxel_features = data_dict['voxel_features']
        voxel_coords = data_dict['voxel_features_coords']
        pillar_coords = voxel_coords[:,[0,2,3]]
        unq_pillar_coords, unq_inv = torch.unique(pillar_coords, dim=0, return_inverse=True)
        spconv2 = data_dict['sparse_input'].__dict__['indice_dict']['spconv2']
        ori_to_down = spconv2.__dict__['pair_bwd']
        converted_pillar_merge_coords =  ori_to_down[:,unq_inv].permute(1,0).long()
        sparse_feat = data_dict['sparse_input']._features
        flatten_converted_pillar_merge_coords = converted_pillar_merge_coords.reshape(-1)
        mask = (flatten_converted_pillar_merge_coords!=-1)
        ori_cnt = (converted_pillar_merge_coords!=-1).sum(-1)
        new_sparse_feat = sparse_feat[flatten_converted_pillar_merge_coords][mask]
        voxel = torch.cat([voxel_features,voxel_coords],dim=-1)
        new_voxel = torch.repeat_interleave(voxel, ori_cnt, dim=0)
        new_voxel_features = new_voxel[:,:voxel_features.shape[-1]]
        new_voxel_coords = new_voxel[:,voxel_features.shape[-1]:]
        new_converted_pillar_merge_coords = (flatten_converted_pillar_merge_coords)[mask]
        sparse_indices = data_dict['sparse_input'].indices
        voxel_indices = sparse_indices[new_converted_pillar_merge_coords]
        output = new_sparse_feat + self.feature_conv(new_voxel_features)
        new_voxel_coords = torch.cat([new_voxel_coords[:,[0,1]],voxel_indices[:,1:]],dim=-1)
        unq_voxel_coords, unq_inv = torch.unique(new_voxel_coords, dim=0, return_inverse=True)
        voxel_max = torch_scatter.scatter_max(output, unq_inv, dim=0)[0]
        data_dict['voxel_features'] = voxel_max.contiguous()
        data_dict['voxel_features_coords'] = unq_voxel_coords.contiguous()
        return data_dict
                 
    def forward(self, sparse_input, data_dict):
        data_dict['sparse_input'] = sparse_input

        if 'spconv2' in data_dict['sparse_input'].__dict__['indice_dict'].keys():
            #start_time = time.time()
            data_dict = self.voxel_sum_sparse(data_dict)
            #print("voxel_sum_sparse", time.time()-start_time)
        else:
            #start_time = time.time()
            data_dict = self.point_sum_subm(data_dict)
            data_dict = self.dyn_voxelization(data_dict)
            #print("point_sum_subm+dyn_voxelization", time.time()-start_time)
        src, occupied_mask = self.binning(data_dict)
        #start_time = time.time()
        if 'spconv2' in data_dict['sparse_input'].__dict__['indice_dict'].keys():
            src = src[occupied_mask]
            src = self.bin_reduce(src)
            N,P,C = src.shape
            src = src.view(N,P*C)
            src = self.bin_shuffle_sparse(src)
            
        else:
            src = src[occupied_mask]
            N,P,C = src.shape
            src = src.view(N,P*C)
            src = self.bin_shuffle(src)
        #print("bin_shuffle_time", time.time()-start_time)
        sparse_input._features[occupied_mask] = sparse_input._features[occupied_mask] + src
        return sparse_input

class ZcCBAM(CBAM):
    def __init__(self,
                 input_channel,
                 feat_channel,
                 output_channel,
                 num_bins):
        super().__init__(input_channel,
                 feat_channel,
                 output_channel,
                 num_bins)
        self.bin_shuffle = bin_shuffle((input_channel)*num_bins, output_channel)

    def forward(self, sparse_input, data_dict):
        data_dict['sparse_input'] = sparse_input

        if 'spconv2' in data_dict['sparse_input'].__dict__['indice_dict'].keys():
            data_dict = self.point_sum_sparse(data_dict)
        else:
            data_dict = self.point_sum_subm(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask = self.binning(data_dict)

        src = src[occupied_mask]
        src = src.permute(0,2,1).contiguous()
        src = self.zbam(src)
        N,P,C = src.shape
        src = src.view(N,P*C)
        src = self.bin_shuffle(src)
        
        sparse_input._features[occupied_mask] = sparse_input._features[occupied_mask] + src
        return sparse_input

class Transformer(CBAM):
    def __init__(self,
                 input_channel,
                 feat_channel,
                 output_channel,
                 num_bins):
        super().__init__(input_channel,
                 feat_channel,
                 output_channel,
                 num_bins)
        self.transformer_model = build_transformer(dict(),num_bins)
        self.query_embed = nn.Embedding(1, num_bins)
        self.zbam = None
    def forward(self, sparse_input, data_dict):
        data_dict['sparse_input'] = sparse_input

        if 'spconv2' in data_dict['sparse_input'].__dict__['indice_dict'].keys():
            data_dict = self.point_sum_sparse(data_dict)
        else:
            data_dict = self.point_sum_subm(data_dict)
        data_dict = self.dyn_voxelization(data_dict)
        src, occupied_mask = self.binning(data_dict)
        src = src[occupied_mask]
        src, _ = self.transformer_model(self.query_embed.weight,src)
        sparse_input._features[occupied_mask] = sparse_input._features[occupied_mask] + src
        return sparse_input


class VFE(CBAM):
    def __init__(self,
                 input_channel,
                 feat_channel,
                 output_channel,
                 num_bins):
        super().__init__(input_channel,
                 feat_channel,
                 output_channel,
                 num_bins)
        self.zbam = None
        self.bin_shuffle = bin_shuffle(feat_channel, output_channel)
        self.conv = Conv1d(input_channel, output_channel)

    def forward(self, sparse_input, data_dict):
        unq_coords, unq_inv, unq_cnt = torch.unique(data_dict['pillar_merge_coords'], return_inverse=True, return_counts=True, dim=0)
        features = data_dict['points_with_f_center'][:,1:]
        features = self.conv(features)
        sparse_feat = sparse_input._features
        input = features + sparse_feat[unq_inv,:]
        x = self.bin_shuffle(input)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]
        sparse_input = sparse_input.replace_feature(sparse_feat + x_max)
        return sparse_input

def build_zbam(model_cfg):
    input_channel = model_cfg.input_channel
    feat_channel = model_cfg.feat_channel
    output_channel = model_cfg.output_channel
    num_bins = model_cfg.num_bins
    model_name = model_cfg.zbam
    model_dict={
        'CBAM': CBAM,
        'Zconv': Zconv,
        "ZcCBAM": ZcCBAM,
        "Transformer": Transformer,
        "VFE": VFE
    }
    model_class = model_dict[model_name]
    model = model_class(input_channel,
                        feat_channel,
                        output_channel,
                        num_bins
                        )
    return model