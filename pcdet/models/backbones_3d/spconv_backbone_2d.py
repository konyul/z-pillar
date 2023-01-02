from functools import partial

import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from ...utils.spconv_exp import SparseConv2d, SparseSequential
try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
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

    def forward(self, x):
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

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out
    
    
class PillarBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
            dense_block(256, 256, 3, norm_fn=norm_fn, padding=1),
        )
        
        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }


    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })

        return batch_dict


class PillarRes18BackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4)

        # batch_dict.update({
        #     'encoded_spconv_tensor': out,
        #     'encoded_spconv_tensor_stride': 8
        # })
        batch_dict.update({
            'multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return batch_dict


class PillarResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()

        batch_dict.update({
            'spatial_features': x_conv4,
            'spatial_features_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict
    
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
        self.conv = Conv1d(8,out_channels)
        self.relu = nn.ReLU()
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
        sparse_input = sparse_input.replace_feature(x_max)
        return sparse_input
    
class SparseBasicBlock_exp(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, num_bins=8, stride=1, norm_fn=None, downsample=None, indice_key=None, reweight=False):
        super(SparseBasicBlock_exp, self).__init__()

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

def post_act_block_exp(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'spconv':
        conv = SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class ZAXIS_res(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bins,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.num_bins = num_bins
        if self.use_norm:
            self.linear = nn.Linear(in_channels*self.num_bins, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels*self.num_bins, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, sparse_input, data_dict):
        ori_pillar_feature = data_dict['ori_pillar_features']
        sparse_input_feature = sparse_input._features
        N,P,C = ori_pillar_feature.shape
        inputs = ori_pillar_feature + sparse_input_feature.unsqueeze(1).repeat(1,P,1)
        inputs = inputs.view(N,P*C)
        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        sparse_input = sparse_input.replace_feature(x)
        return sparse_input

class PillarResBackBone8x_ZAXIS(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block_exp

        self.conv1 = SparseSequential(
            SparseBasicBlock_exp(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock_exp(32, 32, norm_fn=norm_fn, indice_key='res1', reweight=True, num_bins = model_cfg.NUM_BINS),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock_exp(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock_exp(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock_exp(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock_exp(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock_exp(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock_exp(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['pillar_features'], batch_dict['pillar_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x_conv1 = self.conv1(input_sp_tensor, batch_dict)
        x_conv2 = self.conv2(x_conv1, batch_dict)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = x_conv4.dense()

        batch_dict.update({
            'spatial_features': x_conv4,
            'spatial_features_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        return batch_dict