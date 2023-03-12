import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate
from pcdet.models.model_utils.transformer import build_transformer
from pcdet.models.model_utils.mlp import build_mlp
import time
class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        self.zpillar = model_cfg.get("ZPILLAR", None)
        self.numbins = int(8 / voxel_size[2])
        if self.zpillar == 'Transformer':
            self.zpillar_model = build_transformer(model_cfg.ZPILLAR_CFG, self.numbins)
            self.query_embed = nn.Embedding(1, model_cfg.ZPILLAR_CFG.hidden_dim)
        elif self.zpillar == 'MLP':
            self.zpillar_model = build_mlp(model_cfg.ZPILLAR_CFG, self.numbins)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        self.scale_z = grid_size[2]
        
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def dyn_voxelization(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        points_coords = points_coords_3d[:,:2]
        mask = ((points_coords_3d >= 0) & (points_coords_3d < self.grid_size)).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_coords_3d = points_coords_3d[mask]
        if self.zpillar:
            batch_dict = self.dyn_voxelization(points, points_coords_3d, batch_dict)
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        if self.zpillar is not None:
            voxel_features, voxel_features_coords = batch_dict['voxel_features'], batch_dict['voxel_features_coords']
            v_feat_coords = voxel_features_coords[:, 0] * self.scale_xy + voxel_features_coords[:, 3] * self.scale_y + voxel_features_coords[:, 2]
            v_feat_unq_coords, v_feat_unq_inv, v_feat_unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
            batch_dict['v_feat_unq_coords'] = v_feat_unq_coords
            batch_dict['v_feat_unq_inv'] = v_feat_unq_inv
            batch_dict['voxel_features'] = voxel_features
            batch_dict['v_feat_unq_cnt'] = v_feat_unq_cnt
            if self.zpillar == 'Transformer':
                z_pillar_feat, occupied_mask = self.zpillar_model(self.query_embed.weight, batch_dict)
            else:
                z_pillar_feat, occupied_mask = self.zpillar_model(batch_dict)
        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
        # features = self.linear1(features)
        # features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        # features = torch.cat([features, features_max[unq_inv, :]], dim=1)
        # features = self.linear2(features)
        # features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   (unq_coords % self.scale_xy) // self.scale_y,
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        if self.zpillar is not None:
            features[occupied_mask] = features[occupied_mask] + z_pillar_feat
        batch_dict['voxel_features'] = batch_dict['pillar_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict

class DynamicScalePillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', True)
        if self.use_absolute_xyz:
            num_point_features += 3
        if self.use_cluster_xyz:
            num_point_features += 3
        if self.with_distance:
            num_point_features += 1
        self.num_filters = self.model_cfg.NUM_FILTERS
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = 8 / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size[:2]).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.AVFE_point_feature_fc = nn.Sequential(nn.Linear(11, 32, bias=False),
                                                    nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                                                    nn.ReLU())
        self.use_shift = self.model_cfg.get("use_shift", False)
        self.use_downsample = self.model_cfg.get("use_downsample", False)
        self.use_downsamplex2 = self.model_cfg.get("use_downsamplex2", False)
        self.use_downsample_shift = self.model_cfg.get("use_downsample_shift", False)
        self.fusion_method = self.model_cfg.get('fusion_method', False)
        in_channel = 32 * 2
        if self.use_shift:
            in_channel += 64
        if self.use_downsample:
            in_channel += 64
        if self.use_downsamplex2:
            in_channel += 64
        if self.use_downsample_shift:
            in_channel += 64
        if self.fusion_method == 'sum':
            in_channel = in_channel // 2
        self.AVFEO_point_feature_fc = nn.Sequential(
                                                nn.Linear(in_channel ,32, bias=False),
                                                nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                                                nn.ReLU())
        if self.fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(64, 1)
            self.fully_connected_layer_q = nn.Sequential(
                                                nn.Linear(64 ,64, bias=False),
                                                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                                                nn.ReLU())
            self.fully_connected_layer_k = nn.Sequential(
                                                nn.Linear(64 ,64, bias=False),
                                                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                                                nn.ReLU())
            self.fully_connected_layer_v = nn.Sequential(
                                                nn.Linear(64 ,64, bias=False),
                                                nn.BatchNorm1d(64, eps=1e-3, momentum=0.01),
                                                nn.ReLU())
        if self.fusion_method == 'gate':
            self.fully_connected_layer = nn.Sequential(
                                                nn.Linear(64 ,32, bias=False),
                                                nn.BatchNorm1d(32, eps=1e-3, momentum=0.01),
                                                nn.ReLU(),
                                                nn.Linear(32 ,1, bias=False))
    def get_output_feature_dim(self):
        return self.num_filters[-1]
    
    def downsample(self, points, downsample_level):
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        scale_xy = grid_size[0] * grid_size[1]
        scale_y = grid_size[1]
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / voxel_size).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size)).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * scale_xy + \
                       points_coords[:, 0] * scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)

        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
            point_mean = points_mean[unq_inv, :]

        features = self.gen_feat(points, f_center, point_mean, f_cluster, unq_inv)
        return features
    
    def gen_feat(self, points, f_center, point_mean, f_cluster, unq_inv):
        features = [points[:,1:], f_cluster, f_center]
        features = torch.cat(features,dim=-1).contiguous()
        scatter_feature = self.AVFE_point_feature_fc(features) 
        x_mean = torch_scatter.scatter_mean(scatter_feature, unq_inv, dim=0)
        features = torch.cat([scatter_feature, x_mean[unq_inv, :]], dim=1)
        return features
    
    def downsample_shift(self, points, downsample_level):
        voxel_size = self.voxel_size[[0, 1]] * downsample_level
        grid_size = (self.grid_size / downsample_level).long()
        voxel_x = voxel_size[0]
        voxel_y = voxel_size[1]
        x_offset = voxel_x / 2 + self.point_cloud_range[0]
        y_offset = voxel_y / 2 + self.point_cloud_range[1]
        shifted_point_cloud_range = self.point_cloud_range[[0,1]] + voxel_size[[0,1]] / 2
        points_coords = (torch.floor((points[:, [1, 2]] - shifted_point_cloud_range[[0, 1]]) / voxel_size[[0, 1]]) + 1).int()
        mask = ((points_coords >= 0) & (points_coords < grid_size[[0, 1]] + 1)).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        shifted_scale_xy = (grid_size[0] + 1) * (grid_size[1] + 1)
        shifted_scale_y = (grid_size[1] + 1)
        merge_coords = points[:, 0].int() * shifted_scale_xy + \
                       points_coords[:, 0] * shifted_scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * voxel_x + x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * voxel_y + y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        
        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
            point_mean = points_mean[unq_inv, :]
            
        features = self.gen_feat(points, f_center, point_mean, f_cluster, unq_inv)
        return features
    
    def shift(self, points):
        shifted_point_cloud_range = self.point_cloud_range[[0,1]] + self.voxel_size[[0,1]] / 2
        points_coords = (torch.floor((points[:, [1, 2]] - shifted_point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]) + 1).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]] + 1)).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()
        shifted_scale_xy = (self.grid_size[0] + 1) * (self.grid_size[1] + 1)
        shifted_scale_y = (self.grid_size[1] + 1)
        merge_coords = points[:, 0].int() * shifted_scale_xy + \
                       points_coords[:, 0] * shifted_scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset
        
        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
            point_mean = points_mean[unq_inv, :]
            
        features = self.gen_feat(points, f_center, point_mean, f_cluster, unq_inv)
        return features
    
    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        if self.use_shift:
            shifted_features = self.shift(points)
        if self.use_downsample:
            downsampled_features = self.downsample(points, 2)
        if self.use_downsamplex2:
            downsampled_featuresx2 = self.downsample(points, 4)
        if self.use_downsample_shift:
            shifted_downsampled_features = self.downsample_shift(points, 2)
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0, 1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_cluster_xyz:
            points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
            f_cluster = points_xyz - points_mean[unq_inv, :]
            point_mean = points_mean[unq_inv, :]

        features = self.gen_feat(points, f_center, point_mean, f_cluster, unq_inv)
        final_features = [features]
        if self.use_shift:
            final_features.append(shifted_features)
        if self.use_downsample:
            final_features.append(downsampled_features)
        if self.use_downsample_shift:
            final_features.append(shifted_downsampled_features)
        if self.use_downsamplex2:
            final_features.append(downsampled_featuresx2)
        if self.fusion_method == 'sum':
            stacked_final_features = torch.stack(final_features)
            final_features = torch.sum(stacked_final_features, dim=0).contiguous()
        elif self.fusion_method == 'concat':
            final_features = torch.cat(final_features, dim=-1).contiguous()
        elif self.fusion_method == 'attention':
            assert len(final_features) == 2
            q = final_features[0]
            k = final_features[1].clone()
            v = final_features[1].clone()
            q = self.fully_connected_layer_q(q)
            q = q[None,:,:]
            k = self.fully_connected_layer_k(k)
            k = k[None,:,:]
            v = self.fully_connected_layer_v(v)
            v = v[None,:,:]
            output, _ = self.attention(q,k,v)
            output = output.squeeze(0)
            final_features = [features, output]
            final_features = torch.cat(final_features, dim=-1).contiguous()
        elif self.fusion_method == 'gate':
            final_features = torch.cat(final_features, dim=-1).contiguous()
            attention_gate = self.fully_connected_layer(final_features)
            final_features = final_features * F.sigmoid(attention_gate)
        final_features_fc = self.AVFEO_point_feature_fc(final_features)
        features = torch_scatter.scatter_max(final_features_fc, unq_inv, dim=0)[0]
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        return batch_dict

class DynamicPillarVFESimple2D(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        if self.use_absolute_xyz:
            num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)
        self.zpillar = model_cfg.get("ZPILLAR", None)
        self.numbins = int(8 / voxel_size[2])
        if self.zpillar == 'Transformer':
            self.zpillar_model = build_transformer(model_cfg.ZPILLAR_CFG, self.numbins)
            self.query_embed = nn.Embedding(1, model_cfg.ZPILLAR_CFG.hidden_dim)
        elif self.zpillar == 'Zconv' or self.zpillar == 'CBAM' or self.zpillar == 'ZcCBAM':
            self.zpillar_model = build_mlp(model_cfg.ZPILLAR_CFG, self.numbins, model_name = self.zpillar)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.fuse_method = model_cfg.get("FUSION_METHOD", None)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def dyn_voxelization(self, points, point_coords, batch_dict):
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                        point_coords[:, 0] * self.scale_yz + \
                        point_coords[:, 1] * self.scale_z + \
                        point_coords[:, 2]
        points_data = points[:, 1:].contiguous()
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)
        
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xyz,
                                    (unq_coords % self.scale_xyz) // self.scale_yz,
                                    (unq_coords % self.scale_yz) // self.scale_z,
                                    unq_coords % self.scale_z), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]
        
        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_features_coords'] = voxel_coords.contiguous()
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)
        points_coords_3d = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        points_coords = points_coords_3d[:,:2]
        mask = ((points_coords_3d >= 0) & (points_coords_3d < self.grid_size)).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_coords_3d = points_coords_3d[mask]
        if self.zpillar:
            batch_dict = self.dyn_voxelization(points, points_coords_3d, batch_dict)
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)
        batch_dict['pillar_merge_coords'] = merge_coords
        batch_dict['unq_inv'] = unq_inv
        batch_dict['points'] = points
        batch_dict['point_cloud_range'] = self.point_cloud_range
        batch_dict['voxel_size'] = self.voxel_size
        batch_dict['grid_size'] = self.grid_size
        if self.zpillar is not None:
            voxel_features, voxel_features_coords = batch_dict['voxel_features'], batch_dict['voxel_features_coords']
            v_feat_coords = voxel_features_coords[:, 0] * self.scale_xy + voxel_features_coords[:, 3] * self.scale_y + voxel_features_coords[:, 2]
            v_feat_unq_coords, v_feat_unq_inv, v_feat_unq_cnt = torch.unique(v_feat_coords, return_inverse=True, return_counts=True, dim=0)
            batch_dict['v_feat_unq_coords'] = v_feat_unq_coords
            batch_dict['v_feat_unq_inv'] = v_feat_unq_inv
            batch_dict['voxel_features'] = voxel_features
            batch_dict['v_feat_unq_cnt'] = v_feat_unq_cnt
            if self.zpillar == 'Transformer':
                z_pillar_feat, occupied_mask = self.zpillar_model(self.query_embed.weight, batch_dict)
            else:
                z_pillar_feat, occupied_mask = self.zpillar_model(batch_dict)
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        batch_dict['points_with_f_center'] = torch.cat([points[:, :1],features],dim=1)
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords * 0),
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        voxel_coords = voxel_coords[:, [0, 1, 3, 2]]
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        pillar_coords = pillar_coords[:, [0, 2, 1]]
        if self.zpillar is not None:
            features[occupied_mask] = features[occupied_mask] + z_pillar_feat
        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict
