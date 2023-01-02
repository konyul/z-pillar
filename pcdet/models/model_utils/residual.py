from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils.spconv_exp import SparseConv2d, SparseSequential
try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass
import torch
import numpy as np
    
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

class PFNLayer_exp(nn.Module):
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
        self.out_channels = out_channels
        self.conv = Conv1d(8,out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, sparse_input, data_dict):
        ori_pillar_feature = data_dict['ori_pillar_features']
        ori_pillar_feature = self.conv(ori_pillar_feature)
        ori_unq_inv = data_dict['ori_unq_inv']
        sparse_input_feature = sparse_input._features
        inputs = ori_pillar_feature + sparse_input_feature[ori_unq_inv,:]
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, ori_unq_inv, dim=0)[0]
        sorted_ori_unq_inv = torch.unique(ori_unq_inv)
        sparse_input_feature[:x_max.shape[0]][sorted_ori_unq_inv] = x_max[sorted_ori_unq_inv]
        sparse_input = sparse_input.replace_feature(sparse_input_feature)
        return sparse_input

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, num_bins=8, stride=1, norm_fn=None, downsample=None, indice_key=None, reweight=False):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride
        self.reweight = reweight
        if self.reweight:
            #self.pfn = ZAXIS_res(planes, planes, num_bins, last_layer=True)
            self.pfn = PFNLayer_exp(planes, planes, num_bins, last_layer=True)

    def forward(self, x, data_dict=None):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        if self.reweight:
            out = self.pfn(out, data_dict)
        return out


class RESIDUAL(nn.Module):
    def __init__(self, grid_size, num_bins, *kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = np.array(grid_size)[[1, 0]]
        
        self.conv1 = SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res0'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res0', reweight=True, num_bins = num_bins),
        )

    
    def forward(self, x, idx, batch_dict):
        pillar_features, pillar_coords = x, idx
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor, batch_dict)
        return x_conv1
        
        
def build(model_cfg, model_name='RESIDUAL'):
    model_dict = {
        'RESIDUAL': RESIDUAL,
    }
    grid_size = model_cfg.grid_size
    num_filter = model_cfg.num_filter
    num_bins = model_cfg.num_bins
    model_class = model_dict[model_name]
    max_num_nodes = model_cfg.max_num_nodes

    model = model_class(grid_size,
                        num_bins,
                        )
    return model
