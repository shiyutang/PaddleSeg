# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file refers to https://github.com/hustvl/TopFormer and https://github.com/fudan-zvg/SeaFormer
"""

import paddle
from paddle import ParamAttr
from paddle.regularizer import L2Decay
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg import utils
from paddleseg.models.backbones.transformer_utils import Identity, DropPath


def _make_divisible(val, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(val + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * val:
        new_v += divisor
    return new_v


class HSigmoid(nn.Layer):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6()

    def forward(self, x):
        return self.relu(x + 3) / 6


class Conv2DBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 lr_mult=1.0):
        super().__init__()
        conv_weight_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.c = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ks,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            weight_attr=conv_weight_attr,
            bias_attr=False)
        bn_weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(bn_weight_init),
            learning_rate=lr_mult)
        bn_bias_attr = paddle.ParamAttr(
            initializer=nn.initializer.Constant(0), learning_rate=lr_mult)
        self.bn = nn.BatchNorm2D(
            out_channels, weight_attr=bn_weight_attr, bias_attr=bn_bias_attr)

    def forward(self, inputs):
        out = self.c(inputs)
        out = self.bn(out)
        return out


class ConvBNAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 norm=nn.BatchNorm2D,
                 act=None,
                 bias_attr=False,
                 lr_mult=1.0,
                 use_conv=True):
        super(ConvBNAct, self).__init__()
        param_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                weight_attr=param_attr,
                bias_attr=param_attr if bias_attr else False)
        self.act = act() if act is not None else Identity()
        self.bn = norm(out_channels, weight_attr=param_attr, bias_attr=param_attr) \
            if norm is not None else Identity()

    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Hardsigmoid(nn.Layer):
    def __init__(self, slope=0.2, offset=0.5):
        super().__init__()
        self.slope = slope
        self.offset = offset

    def forward(self, x):
        return nn.functional.hardsigmoid(
            x, slope=self.slope, offset=self.offset)


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = Hardsigmoid(slope=0.2, offset=0.5)

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 dilation=1):
        super().__init__()

        def _create_act(act):
            if act == "hardswish":
                return nn.Hardswish()
            elif act == "relu":
                return nn.ReLU()
            elif act is None:
                return None
            else:
                raise RuntimeError(
                    "The activation function is not supported: {}".format(act))

        self.c = nn.Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False,
            dilation=dilation)
        self.bn = nn.BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        self.act = _create_act(act)

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 dilation=1):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2) * dilation,
            num_groups=mid_c,
            if_act=True,
            act=act,
            dilation=dilation)
        if self.if_se:
            self.mid_se = SEModule(mid_c)
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


class StackedMV3Block(nn.Layer):
    """
    MobileNetV3
    Args:
        config: list. MobileNetV3 depthwise blocks config.
        in_channels (int, optional): The channels of input image. Default: 3.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
    Returns:
        model: nn.Layer. Specific MobileNetV3 model depends on args.
    """

    def __init__(self,
                 cfgs,
                 stem,
                 inp_channel,
                 in_channels=3,
                 scale=1.0,
                 lr_mult=1.0,
                 pretrained=None):
        super().__init__()

        self.scale = scale
        self.stem = stem

        if self.stem:
            self.conv = ConvBNLayer(
                in_c=3,
                out_c=_make_divisible(inp_channel * self.scale),
                filter_size=3,
                stride=2,
                padding=1,
                num_groups=1,
                if_act=True,
                act="hardswish")
        self.blocks = nn.LayerList()
        for i, (k, exp, c, se, act, s) in enumerate(cfgs):
            self.blocks.append(
                ResidualUnit(
                    in_c=_make_divisible(inp_channel * self.scale),
                    mid_c=_make_divisible(self.scale * exp),
                    out_c=_make_divisible(self.scale * c),
                    filter_size=k,
                    stride=s,
                    use_se=se,
                    act=act,
                    dilation=1))
            inp_channel = _make_divisible(self.scale * c)

    def forward(self, x):
        if self.stem:
            x = self.conv(x)

        for i, block in enumerate(self.blocks):
            x = block(x)

        return x


class SqueezeAxialPositionalEmbedding(nn.Layer):
    def __init__(self, dim, shape):
        super().__init__()
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal())
        self.pos_embed = paddle.create_parameter(
            shape=[1, dim, shape], attr=weight_attr, dtype='float32')

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(
            self.pos_embed,
            size=(N, ),
            mode='linear',
            align_corners=False,
            data_format='NCW')
        return x


class Sea_Attention(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 attn_ratio=4,
                 activation=None,
                 lr_mult=1.0,
                 stride_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_k = Conv2DBN(dim, nh_kd, 1, lr_mult=lr_mult)
        self.to_v = Conv2DBN(dim, self.dh, 1, lr_mult=lr_mult)

        self.stride_attention = stride_attention

        if self.stride_attention:
            self.stride_conv = nn.Sequential(
                nn.Conv2D(
                    dim, dim, kernel_size=3, stride=2, padding=1, groups=dim),
                nn.BatchNorm2D(dim), )

        self.proj = paddle.nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, dim, bn_weight_init=0, lr_mult=lr_mult))
        self.proj_encode_row = paddle.nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, self.dh, bn_weight_init=0, lr_mult=lr_mult))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = paddle.nn.Sequential(
            activation(),
            Conv2DBN(
                self.dh, self.dh, bn_weight_init=0, lr_mult=lr_mult))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2DBN(
            2 * self.dh,
            2 * self.dh,
            ks=3,
            stride=1,
            pad=1,
            dilation=1,
            groups=2 * self.dh,
            lr_mult=lr_mult)
        self.act = activation()
        self.pwconv = Conv2DBN(2 * self.dh, dim, ks=1, lr_mult=lr_mult)
        self.sigmoid = HSigmoid()

    def forward(self, x):  # x (B,N,C)
        B, C, H_ori, W_ori = x.shape
        if self.stride_attention:
            x = self.stride_conv(x)
        B, C, H, W = x.shape

        q = self.to_q(x)  # [B, nhead*dim, H, W]
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = paddle.concat([q, k, v], axis=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(
            [B, self.num_heads, -1, H]).transpose(
                [0, 1, 3, 2])  # [B, nhead, H, dim] 
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(
            [B, self.num_heads, -1, H])  # [B, nhead, dim, H]
        vrow = v.mean(-1).reshape([B, self.num_heads, -1, H]).transpose(
            [0, 1, 3, 2])  # [B, nhead, H, dim*attn_ratio]

        attn_row = paddle.matmul(qrow, krow) * self.scale  # [B, nhead, H, H]
        attn_row = F.softmax(attn_row, axis=-1)

        xx_row = paddle.matmul(attn_row, vrow)  # [B, nhead, H, dim*attn_ratio]
        xx_row = self.proj_encode_row(
            xx_row.transpose([0, 1, 3, 2]).reshape([B, self.dh, H, 1]))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(
            [B, self.num_heads, -1, W]).transpose([0, 1, 3, 2])
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(
            [B, self.num_heads, -1, W])
        vcolumn = paddle.mean(v, -2).reshape(
            [B, self.num_heads, -1, W]).transpose([0, 1, 3, 2])

        attn_column = paddle.matmul(qcolumn, kcolumn) * self.scale
        attn_column = F.softmax(attn_column, axis=-1)

        xx_column = paddle.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(
            xx_column.transpose([0, 1, 3, 2]).reshape([B, self.dh, 1, W]))

        xx = paddle.add(xx_row, xx_column)  # [B, self.dh, H, W] 
        xx = paddle.add(v, xx)

        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        if self.stride_attention:
            xx = F.interpolate(xx, size=(H_ori, W_ori), mode='bilinear')

        return xx


class MLP(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.,
                 lr_mult=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2DBN(in_features, hidden_features, lr_mult=lr_mult)
        param_attr = paddle.ParamAttr(learning_rate=lr_mult)
        self.dwconv = nn.Conv2D(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            groups=hidden_features,
            weight_attr=param_attr,
            bias_attr=param_attr)
        self.act = act_layer()
        self.fc2 = Conv2DBN(hidden_features, out_features, lr_mult=lr_mult)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads,
                 mlp_ratio=4.,
                 attn_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.ReLU,
                 lr_mult=1.0,
                 stride_attention=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Sea_Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            lr_mult=lr_mult,
            stride_attention=stride_attention)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity(
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       lr_mult=lr_mult)

    def forward(self, x1):
        # 多头注意力
        x1 = x1 + self.drop_path(self.attn(x1))
        # 线性组合多头注意力
        x1 = x1 + self.drop_path(self.mlp(x1))

        return x1


class BasicLayer(nn.Layer):
    def __init__(self,
                 block_num,
                 embedding_dim,
                 key_dim,
                 num_heads,
                 mlp_ratio=4.,
                 attn_ratio=2.,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 lr_mult=1.0,
                 act_layer=None,
                 stride_attention=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.LayerList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    lr_mult=lr_mult,
                    stride_attention=stride_attention))

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class Fusion_block(nn.Layer):
    def __init__(self, inp, oup, embed_dim, activations=None,
                 lr_mult=1.0) -> None:
        super(Fusion_block, self).__init__()
        self.local_embedding = ConvBNAct(
            inp, embed_dim, kernel_size=1, lr_mult=lr_mult)
        self.global_act = ConvBNAct(
            oup, embed_dim, kernel_size=1, lr_mult=lr_mult)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        B, C_c, H_c, W_c = x_g.shape

        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            self.act(global_act),
            size=(H, W),
            mode='bilinear',
            align_corners=False)

        out = local_feat * sig_act

        return out


class InjectionMultiSumallmultiallsum(nn.Layer):
    def __init__(self,
                 in_channels=(64, 128, 256, 384),
                 activations=None,
                 out_channels=160,
                 lr_mult=1.0):
        super(InjectionMultiSumallmultiallsum, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.act_list = nn.LayerList()
        self.out_channels = out_channels
        for i in range(len(in_channels)):
            self.embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            self.act_embedding_list.append(
                ConvBNAct(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    lr_mult=lr_mult))
            self.act_list.append(nn.Sigmoid())

    def forward(
            self, inputs
    ):  # x_x8, x_x16, x_x32, x_x64 3 outputs: [4, 64, 64, 64] [4, 192, 16, 16] [4, 256, 8, 8]
        low_feat1 = F.interpolate(
            inputs[0], scale_factor=0.5, mode="bilinear")  # x16
        low_feat1_act = self.act_list[0](self.act_embedding_list[0](low_feat1))
        low_feat1 = self.embedding_list[0](low_feat1)

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode="bilinear")  # x16
        low_feat2_act = self.act_list[1](
            self.act_embedding_list[1](low_feat2))  # x16
        low_feat2 = self.embedding_list[1](low_feat2)

        # low_feat3_act = F.interpolate(
        #     self.act_list[2](self.act_embedding_list[2](inputs[2])),
        #     size=low_feat2.shape[2:],
        #     mode="bilinear")
        # low_feat3 = F.interpolate(
        #     self.embedding_list[2](inputs[2]),
        #     size=low_feat2.shape[2:],
        #     mode="bilinear")

        high_feat_act = F.interpolate(
            self.act_list[2](self.act_embedding_list[2](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[2](inputs[2]),
            size=low_feat2.shape[2:],
            mode="bilinear")

        res = low_feat1_act * low_feat2_act * high_feat_act * (
            low_feat1 + low_feat2) + high_feat

        return res


class InjectionMultiSumallmultiallsumSimpx8(nn.Layer):
    def __init__(self,
                 in_channels=(64, 128, 256, 384),
                 activations=None,
                 out_channels=160,
                 lr_mult=1.0):
        super(InjectionMultiSumallmultiallsumSimpx8, self).__init__()
        self.embedding_list = nn.LayerList()
        self.act_embedding_list = nn.LayerList()
        self.act_list = nn.LayerList()
        self.out_channels = out_channels
        for i in range(len(in_channels)):
            if i != 1:
                self.embedding_list.append(
                    ConvBNAct(
                        in_channels[i],
                        out_channels,
                        kernel_size=1,
                        lr_mult=lr_mult))
            if i != 0:
                self.act_embedding_list.append(
                    ConvBNAct(
                        in_channels[i],
                        out_channels,
                        kernel_size=1,
                        lr_mult=lr_mult))
                self.act_list.append(nn.Sigmoid())

    def forward(self, inputs):
        # x_x8, x_x16, x_x32
        low_feat1 = self.embedding_list[0](inputs[0])

        low_feat2 = F.interpolate(
            inputs[1], size=low_feat1.shape[-2:], mode="bilinear")
        low_feat2_act = self.act_list[0](self.act_embedding_list[0](low_feat2))

        high_feat_act = F.interpolate(
            self.act_list[1](self.act_embedding_list[1](inputs[2])),
            size=low_feat2.shape[2:],
            mode="bilinear")
        high_feat = F.interpolate(
            self.embedding_list[1](inputs[2]),
            size=low_feat2.shape[2:],
            mode="bilinear")

        res = low_feat2_act * high_feat_act * low_feat1 + high_feat

        return res


class MobileSegNet(nn.Layer):
    def __init__(self,
                 cfgs,
                 channels,
                 embed_dims,
                 key_dims,
                 depths=[2, 2],
                 num_heads=8,
                 attn_ratios=2,
                 mlp_ratios=[2, 4],
                 drop_path_rate=0.,
                 act_layer=nn.ReLU6,
                 lr_mult=1.0,
                 in_channels=3,
                 inj_type='AAM',
                 pretrained=None,
                 out_channels=160,
                 dims=(128, 160),
                 out_feat_chs=None,
                 stride_attention=False):
        super().__init__()
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.dims = dims

        for i in range(len(cfgs)):
            smb = StackedMV3Block(
                cfgs=cfgs[i],
                stem=True if i == 0 else False,
                inp_channel=channels[i],
                lr_mult=lr_mult)
            setattr(self, f'smb{i+1}', smb)

        for i in range(len(depths)):
            dpr = [
                x.item() for x in paddle.linspace(0, drop_path_rate, depths[i])
            ]
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=embed_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0,
                attn_drop=0.0,
                drop_path=dpr,
                act_layer=act_layer,
                lr_mult=lr_mult,
                stride_attention=stride_attention)
            setattr(self, f"trans{i+1}", trans)

        self.inj_type = inj_type
        if self.inj_type == "AAM":
            self.inj_module = InjectionMultiSumallmultiallsum(
                in_channels=out_feat_chs,
                activations=act_layer,
                out_channels=out_channels,
                lr_mult=lr_mult)
            self.injection_out_channels = [self.inj_module.out_channels] * 3
        elif self.inj_type == "AAMSx8":
            self.inj_module = InjectionMultiSumallmultiallsumSimpx8(
                in_channels=out_feat_chs,
                activations=act_layer,
                out_channels=out_channels,
                lr_mult=lr_mult)
            self.injection_out_channels = [self.inj_module.out_channels] * 3
        elif self.inj_type == 'origin':
            for i in range(len(dims)):
                fuse = Fusion_block(
                    out_feat_chs[0] if i == 0 else dims[i - 1],
                    out_feat_chs[i + 1],
                    embed_dim=dims[i],
                    lr_mult=lr_mult)
                setattr(self, f"fuse{i + 1}", fuse)
            self.injection_out_channels = [dims[i]] * 3
        else:
            raise NotImplementedError(self.inj_module + ' is not implemented')

        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        outputs = []
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)

        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)

            # 1/8 shared feat
            if i == 1:
                outputs.append(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(
                    self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)
                outputs.append(x)

        if self.inj_type == "origin":
            x_detail = outputs[0]
            for i in range(len(self.dims)):
                fuse = getattr(self, f'fuse{i+1}')

                x_detail = fuse(x_detail, outputs[i + 1])
            output = x_detail
        else:
            output = self.inj_module(outputs)

        return output


@manager.BACKBONES.add_component
def MobileSeg_Base(**kwargs):
    cfg1 = [
        # k t c, s
        [3, 16, 16, True, "relu", 1],
        [3, 64, 32, False, "relu", 2],
        [3, 96, 32, False, "relu", 1]
    ]
    cfg2 = [[5, 128, 64, True, "hardswish", 2],
            [5, 240, 64, True, "hardswish", 1]]
    cfg3 = [[5, 384, 128, True, "hardswish", 2],
            [5, 384, 128, True, "hardswish", 1]]
    cfg4 = [[5, 768, 192, True, "hardswish", 2],
            [5, 768, 192, True, "hardswish", 1]]

    channels = [16, 32, 64, 128, 192]
    depths = [3, 3]
    key_dims = [16, 24]
    emb_dims = [128, 192]
    num_heads = 8
    drop_path_rate = 0.1

    model = MobileSegNet(
        cfgs=[cfg1, cfg2, cfg3, cfg4],
        channels=channels,
        embed_dims=emb_dims,
        key_dims=key_dims,
        depths=depths,
        num_heads=num_heads,
        drop_path_rate=drop_path_rate,
        act_layer=nn.ReLU6,
        inj_type='AAMSx8',
        **kwargs)

    return model
