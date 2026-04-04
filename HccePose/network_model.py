# Author: Yulin Wang (yulinwang@seu.edu.cn)
# School of Mechanical Engineering, Southeast University, China

"""HccePose network definitions.

This file keeps the lightweight HccePose source-file style and documents the
upstream origin of the segmentation/backbone modules adapted from ZebraPose.
The referenced ZebraPose repository is released under the MIT License:
https://github.com/suyz526/ZebraPose

Relevant upstream files:
- zebrapose/model/aspp.py
- zebrapose/model/resnet.py
- zebrapose/model/efficientnet.py
- zebrapose/model/BinaryCodeNet.py

---

本文件定义 HccePose 所用的分割头、主干与包装网络，并保持与上游 ZebraPose
实现一致的轻量组织方式。分割与编码解码相关模块在 MIT 许可下改编自上述
仓库路径中的对应文件。
"""

import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from typing import List
import copy
from torchvision.models.efficientnet import MBConvConfig,MBConv
from torchvision.ops.misc import ConvNormActivation

# Adapted from ZebraPose `zebrapose/model/aspp.py` (MIT License).
class ASPP(nn.Module):
    '''
    ---
    ---
    ASPP decoder head for ResNet-OS8 high-level features.
    ---
    ---
    It fuses multi-scale atrous convolutions and global pooling, then upsamples
    and fuses with skip features `x_64` and `x_128` to predict dense logits.

    Args:
        - num_classes: Output channels (mask + binary code channels).
        - concat: Reserved flag from upstream (kept for compatibility).
    ---
    ---
    面向 ResNet-OS8 高层特征的 ASPP 解码头。
    ---
    ---
    融合多尺度空洞卷积与全局池化特征，再上采样并与 `x_64`、`x_128` 跳跃连接
    拼接，输出密集预测 logits。

    参数:
        - num_classes: 输出通道数（1 路 mask + 多路编码）。
        - concat: 上游保留参数（为兼容而保留）。
    '''
    def __init__(self, num_classes, concat=True):
        super(ASPP, self).__init__()
        self.concat = concat

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        padding = 1
        output_padding = 1

        self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding) 
        self.upsample_2 = self.upsample(256+64, 256, 3, padding, output_padding) 

        self.conv_1x1_4 = nn.Conv2d(256 + 64, num_classes, kernel_size=1, padding=0)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        '''
        ---
        ---
        Build one transposed-convolution upsampling block with BN and conv refinements.
        ---
        ---
        Args:
            - in_channels: Input channel count.
            - num_filters: Output channel count after upsampling.
            - kernel_size: Kernel size of the transposed convolution.
            - padding / output_padding: ConvTranspose2d padding settings.

        Returns:
            - upsample_layer: Sequential module (stride-2 upsample + conv stack).
        ---
        ---
        构造一组转置卷积上采样子模块（含 BN 与卷积细化）。
        ---
        ---
        参数:
            - in_channels: 输入通道数。
            - num_filters: 上采样后的通道数。
            - kernel_size: 转置卷积核大小。
            - padding / output_padding: 转置卷积的填充参数。

        返回:
            - upsample_layer: 顺序模块（2 倍上采样 + 卷积堆叠）。
        '''
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, x_128=None, x_64=None, x_32=None, x_16=None):
        '''
        ---
        ---
        Run ASPP on the deepest ResNet feature and fuse skip connections.
        ---
        ---
        Args:
            - x_high_feature: Deepest feature map from the backbone.
            - x_128, x_64, x_32, x_16: Optional skip maps (`x_128`/`x_64` are used).

        Returns:
            - Dense prediction tensor before the final split into mask and codes.
        ---
        ---
        在最深层 ResNet 特征上执行 ASPP，并与跳跃特征融合。
        ---
        ---
        参数:
            - x_high_feature: 主干最深层特征图。
            - x_128, x_64, x_32, x_16: 多尺度跳跃特征（实际使用 `x_128` 与 `x_64`）。

        返回:
            - 尚未拆分为 mask 与编码通道前的密集预测张量。
        '''

        feature_map_h = x_high_feature.size()[2]
        feature_map_w = x_high_feature.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature))) 
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) 
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) 

        out_img = self.avg_pool(x_high_feature) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 

        x = self.upsample_1(out)

        x = torch.cat([x, x_64], 1)
        x = self.upsample_2(x)
    
        x = self.conv_1x1_4(torch.cat([x, x_128], 1)) 

        return x

# Adapted from ZebraPose `zebrapose/model/aspp.py` (MIT License).
class ASPP_Efficientnet_upsampled(nn.Module):
    '''
    ---
    ---
    ASPP head tailored to EfficientNet-B4 feature channels (448-dim input).
    ---
    ---
    Same multi-scale pooling and upsampling idea as `ASPP`, but fuses EfficientNet
    stage tensors `l3` and `l2` instead of ResNet skips.

    Args:
        - num_classes: Output channels (mask + binary code channels).
    ---
    ---
    针对 EfficientNet-B4 特征通道（448 维输入）定制的 ASPP 头。
    ---
    ---
    与 `ASPP` 相同的多尺度池化与上采样思路，但融合 EfficientNet 的 `l3`、`l2`
    阶段特征而非 ResNet 跳跃连接。

    参数:
        - num_classes: 输出通道数（mask + 编码）。
    '''
    def __init__(self, num_classes):
        super(ASPP_Efficientnet_upsampled, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(448, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)
        self.conv_3x3_1 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)
        self.conv_3x3_2 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)
        self.conv_3x3_3 = nn.Conv2d(448, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(448, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)
        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)
        padding = 1
        output_padding = 1
        self.upsample_1 = self.upsample(256, 256, 3, padding, output_padding)
        self.upsample_2 = self.upsample(256+32, 256, 3, padding, output_padding)
        self.conv_1x1_4 = nn.Conv2d(256 + 24, num_classes, kernel_size=1, padding=0)

    def upsample(self, in_channels, num_filters, kernel_size, padding, output_padding):
        '''
        ---
        ---
        Same upsampling builder as in `ASPP.upsample` (transposed conv + conv stack).
        ---
        ---
        Args / Returns: See `ASPP.upsample`.
        ---
        ---
        与 `ASPP.upsample` 相同的上采样构建器（转置卷积 + 卷积堆叠）。
        ---
        ---
        参数 / 返回: 同 `ASPP.upsample`。
        '''
        upsample_layer = nn.Sequential(
                            nn.ConvTranspose2d(
                                in_channels,
                                num_filters,
                                kernel_size=kernel_size,
                                stride=2,
                                padding=padding,
                                output_padding=output_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_filters),
                            nn.ReLU(inplace=True)
                        )
        return upsample_layer


    def forward(self, x_high_feature, l3=None,l2=None):
        '''
        ---
        ---
        Fuse EfficientNet deep features with `l3` and `l2` skip tensors.
        ---
        ---
        Args:
            - x_high_feature: Deepest EfficientNet feature map.
            - l3, l2: Mid-level features concatenated after each upsampling step.

        Returns:
            - Dense logits before splitting mask and code channels.
        ---
        ---
        将 EfficientNet 深层特征与 `l3`、`l2` 跳跃张量融合。
        ---
        ---
        参数:
            - x_high_feature: EfficientNet 最深层特征。
            - l3, l2: 两次上采样后分别拼接的中层特征。

        返回:
            - 拆分 mask 与编码前的密集 logits。
        '''
        feature_map_h = x_high_feature.size()[2]
        feature_map_w = x_high_feature.size()[3]
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature))) 
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature))) 

        out_img = self.avg_pool(x_high_feature) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        x = self.upsample_1(out)
        x = torch.cat([x, l3], 1)
        x = self.upsample_2(x)
            
        x = self.conv_1x1_4(torch.cat([x, l2], 1)) 
        return x

# Adapted from ZebraPose `zebrapose/model/efficientnet.py` (MIT License).
class efficientnet_upsampled(nn.Module):
    '''
    ---
    ---
    EfficientNet-B4 backbone with adjustable input channels and extra MBConv stages.
    ---
    ---
    Loads torchvision EfficientNet-B4, replaces stem conv for `input_channels`,
    and appends inverted-residual stages for richer features used by the ASPP head.

    Args:
        - input_channels: Number of input image channels (default 3).
    ---
    ---
    基于 EfficientNet-B4 的主干，可改输入通道并附加 MBConv 阶段。
    ---
    ---
    加载 torchvision 的 EfficientNet-B4，将首层卷积替换为 `input_channels`，
    并追加倒残差阶段以供 ASPP 头使用。

    参数:
        - input_channels: 输入图像通道数（默认 3）。
    '''
    def __init__(self, input_channels=3):
        super(efficientnet_upsampled,self).__init__()
        print("efficientnet_b4")
        efficientnet = models.efficientnet_b4()
        old_conv1 = efficientnet.features[0][0]
        new_conv1 = nn.Conv2d(
            in_channels=input_channels,  
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=True if old_conv1.bias else False,
        )
        new_conv1.weight[:, :old_conv1.in_channels, :, :].data.copy_(old_conv1.weight.clone())
        efficientnet.features[0][0] = new_conv1
        self.efficientnet = nn.Sequential(*list(efficientnet.children())[0])
        block = MBConv
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        layers: List[nn.Module] = []
        width_mult = 1.4
        depth_mult=1.8
        stochastic_depth_prob = 0.2
        bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
        inverted_residual_setting = [
                            bneck_conf(6, 3, 1, 40, 80, 3),
                            bneck_conf(6, 5, 1, 80, 112, 3),
                            bneck_conf(6, 5, 1, 112, 192, 4),
                            bneck_conf(6, 3, 1, 192, 320, 1),
                            ]
        self.eff_layer_2 = nn.Sequential(*list(self.efficientnet.children())[:2])
        self.eff_layer_3 = nn.Sequential(*list(self.efficientnet.children())[2:3])
        self.eff_layer_4 = nn.Sequential(*list(self.efficientnet.children())[3:4])
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels =  lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                    norm_layer=norm_layer, activation_layer=nn.SiLU))
        self.final_layer = nn.Sequential(*layers)


    def forward(self,x):
        '''
        ---
        ---
        Forward pass returning deep feature and two skip levels.
        ---
        ---
        Args:
            - x: Input image tensor (B, C, H, W).

        Returns:
            - final: Deepest feature map for ASPP.
            - l3, l2: Skip features fused inside `ASPP_Efficientnet_upsampled`.
        ---
        ---
        前向传播，返回最深层特征与两级跳跃特征。
        ---
        ---
        参数:
            - x: 输入图像张量 (B, C, H, W)。

        返回:
            - final: 供 ASPP 使用的最深层特征。
            - l3, l2: 在 `ASPP_Efficientnet_upsampled` 中融合的跳跃特征。
        '''
        l2 = self.eff_layer_2(x)
        l3 = self.eff_layer_3(l2)
        l4 = self.eff_layer_4(l3)
        final = self.final_layer(l4)
        return final,l3,l2

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    '''
    ---
    ---
    Build a sequential residual stage from repeated blocks.
    ---
    ---
    The first block may change stride or dilation, while the following blocks keep
    `stride=1`. It is used as a compact builder for the current ResNet-OS8 backbone.

    Args:
        - block: Residual block class (e.g. `BasicBlock`).
        - in_channels: Input channels to the first block.
        - channels: Base channel width inside each block.
        - num_blocks: Number of stacked blocks.
        - stride: Stride of the first block.
        - dilation: Dilation rate shared by the stage.

    Returns:
        - layer: `nn.Sequential` of stacked blocks.
    ---
    ---
    按照给定 block 类型、通道数和步长，快速构造一整层残差块。
    ---
    ---
    首个 block 可以带步长或膨胀参数，后续 block 统一使用 stride=1，
    作为当前 ResNet-OS8 主干的搭建工具。

    参数:
        - block: 残差块类（如 `BasicBlock`）。
        - in_channels: 第一个 block 的输入通道数。
        - channels: 块内基础通道宽度。
        - num_blocks: 堆叠的 block 数量。
        - stride: 第一个 block 的步长。
        - dilation: 该阶段共享的空洞率。

    返回:
        - layer: 堆叠后的 `nn.Sequential`。
    '''
    strides = [stride] + [1]*(num_blocks - 1) 

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks)

    return layer

# Adapted from ZebraPose `zebrapose/model/resnet.py` (MIT License).
class BasicBlock(nn.Module):
    '''
    ---
    ---
    Standard two-convolution ResNet basic block with optional downsampling.
    ---
    ---
    Two 3x3 convolutions with BN; when stride or channel width changes, a 1x1
    shortcut adjusts the residual path.

    Args (constructor):
        - in_channels, channels, stride, dilation: See conv and shortcut layout.
    ---
    ---
    标准双层卷积 ResNet BasicBlock，可选下采样。
    ---
    ---
    两个 3x3 卷积 + BN；当步长或通道变化时，用 1x1 捷径对齐残差路径。

    参数（构造函数）:
        - in_channels, channels, stride, dilation: 卷积与捷径的布局参数。
    '''
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        '''
        ---
        ---
        Apply two convolutions and add the residual shortcut.
        ---
        ---
        Args:
            - x: Input feature map.

        Returns:
            - Activated output tensor.
        ---
        ---
        两次卷积并与残差捷径相加。
        ---
        ---
        参数:
            - x: 输入特征图。

        返回:
            - 激活后的输出张量。
        '''

        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out) 

        return out

# Adapted from ZebraPose `zebrapose/model/resnet.py` (MIT License).
class ResNet_BasicBlock_OS8(nn.Module):
    '''
    ---
    ---
    ResNet-34 trunk modified to output stride 8 with dilated stage 4/5.
    ---
    ---
    Replaces the original stride-32 tail with dilated `BasicBlock` stages so that
    `x_high_feature` keeps higher resolution for dense prediction.

    Args (constructor):
        - input_channels: Stem input channels (RGB default 3).
    ---
    ---
    修改为输出步长 8 的 ResNet-34 主干（第 4/5 阶段使用空洞卷积）。
    ---
    ---
    用带空洞的 `BasicBlock` 阶段替换原 stride-32 尾部，使 `x_high_feature`
    保持更高分辨率以利于密集预测。

    参数（构造函数）:
        - input_channels: 首层输入通道数（RGB 默认为 3）。
    '''
    def __init__(self, input_channels = 3):
        super(ResNet_BasicBlock_OS8, self).__init__()
        resnet = models.resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(resnet.children())[:-4])
        # first conv, bn, relu
        self.resnet_layer_1 = nn.Sequential(*list(resnet.children())[:-7]) 
        # max pooling, resnet block
        self.resnet_layer_2 = nn.Sequential(*list(resnet.children())[-7:-5]) 
        # resnet block
        self.resnet_layer_3 = nn.Sequential(*list(resnet.children())[-5:-4])
        num_blocks_layer_4 = 6
        num_blocks_layer_5 = 3
        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)
        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)
        print ("resnet 34")

    def forward(self, x):
        '''
        ---
        ---
        Extract multi-scale features for ASPP and skip fusion.
        ---
        ---
        Args:
            - x: Input image tensor.

        Returns:
            - x_high_feature: Deepest map for ASPP input.
            - x_128, x_64, x_32, x_16: Skip maps at decreasing resolutions.
        ---
        ---
        提取多尺度特征供 ASPP 与跳跃连接使用。
        ---
        ---
        参数:
            - x: 输入图像张量。

        返回:
            - x_high_feature: 供 ASPP 的最深层特征。
            - x_128, x_64, x_32, x_16: 由粗到细的跳跃特征。
        '''
        x_128 = self.resnet_layer_1(x)
        x_64 = self.resnet_layer_2(x_128)
        x_32 = self.resnet_layer_3(x_64)
        x_16 = self.layer4(x_32)
        x_high_feature = self.layer5(x_16)
        return x_high_feature, x_128, x_64, x_32, x_16

# Adapted from ZebraPose `zebrapose/model/BinaryCodeNet.py` together with
# its ASPP/ResNet/EfficientNet modules (MIT License).
class DeepLabV3(nn.Module):
    '''
    ---
    ---
    DeepLab-style encoder-decoder: backbone + ASPP, split into mask and codes.
    ---
    ---
    Chooses ResNet-OS8 + `ASPP` when `efficientnet_key` is None, otherwise
    EfficientNet + `ASPP_Efficientnet_upsampled`. The final tensor is split into
    one mask channel and remaining binary-code channels.

    Args:
        - num_classes: Total output channels (1 mask + code channels).
        - efficientnet_key: If set, selects EfficientNet backbone branch.
        - input_channels: Input image channels.
    ---
    ---
    DeepLab 风格编解码：主干 + ASPP，输出再拆分为 mask 与编码。
    ---
    ---
    当 `efficientnet_key` 为 None 时使用 ResNet-OS8 + `ASPP`，否则使用
    EfficientNet + `ASPP_Efficientnet_upsampled`。最终张量拆成 1 路 mask
    与剩余二进制编码通道。

    参数:
        - num_classes: 总输出通道（1 + 编码通道数）。
        - efficientnet_key: 非空时走 EfficientNet 分支。
        - input_channels: 输入通道数。
    '''
    def __init__(self, num_classes, efficientnet_key=None, input_channels=3):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes
        self.efficientnet_key = efficientnet_key

        if efficientnet_key == None:
            self.resnet = ResNet_BasicBlock_OS8(input_channels=input_channels) 
            self.aspp = ASPP(num_classes=self.num_classes) 
        else:
            self.efficientnet = efficientnet_upsampled(input_channels=input_channels)
            self.aspp = ASPP_Efficientnet_upsampled(num_classes=self.num_classes) 

    def forward(self, x):
        '''
        ---
        ---
        Forward: backbone -> ASPP -> split mask and hierarchical code logits.
        ---
        ---
        Args:
            - x: Input tensor (batch_size, C, H, W).

        Returns:
            - mask: Single-channel mask logits.
            - binary_code: Remaining channels for front/back code prediction.
        ---
        ---
        前向：主干 -> ASPP -> 拆分为 mask 与分层编码 logits。
        ---
        ---
        参数:
            - x: 输入张量 (batch_size, C, H, W)。

        返回:
            - mask: 单通道 mask logits。
            - binary_code: 其余通道，用于正背面编码预测。
        '''
        # (x has shape (batch_size, 3, h, w))
        if self.efficientnet_key == None:
            x_high_feature, x_128, x_64, x_32, x_16 = self.resnet(x)
            output = self.aspp(x_high_feature, x_128, x_64, x_32, x_16)
        else:
            l9,l3,l2 = self.efficientnet(x)
            output = self.aspp(l9,l3,l2)
        mask,binary_code = torch.split(output,[1,self.num_classes-1],1)
        return mask, binary_code

class FixedSizeList:
    """A tiny fixed-length queue used for recent loss statistics.

    训练损失中需要维护最近若干次 front/back 编码误差的滑动窗口。本类只保留
    固定长度的数据，超出长度时自动丢弃最旧元素，用来支持后续的动态加权。

    This small helper keeps a fixed-length history of recent values. It is used
    by the loss module to maintain moving windows of front/back code errors for
    dynamic weighting.
    """
    def __init__(self, size):
        '''
        ---
        ---
        Create a fixed-capacity FIFO list.
        ---
        ---
        Args:
            - size: Maximum number of elements to retain.
        ---
        ---
        创建定长 FIFO 列表。
        ---
        ---
        参数:
            - size: 最多保留的元素个数。
        '''
        self.size = size
        self.data = []
    
    def append(self, item):
        '''
        ---
        ---
        Append one value, dropping the oldest element when full.
        ---
        ---
        Args:
            - item: Scalar or value to store.
        ---
        ---
        追加一个元素；若已满则丢弃最旧元素。
        ---
        ---
        参数:
            - item: 要保存的标量或数值。
        '''
        if len(self.data) >= self.size:
            self.data.pop(0)  
        self.data.append(item)
    
    def get_list(self):
        '''
        ---
        ---
        Return the current stored sequence as a Python list.
        ---
        ---
        Returns:
            - data: List of stored items.
        ---
        ---
        以 Python list 形式返回当前保存的序列。
        ---
        ---
        返回:
            - data: 已保存元素的列表。
        '''
        return self.data
    
    def __repr__(self):
        '''
        ---
        ---
        Debug string mirroring the internal list representation.
        ---
        ---
        调试字符串，与内部 list 表示一致。
        '''
        return repr(self.data)
    
class HccePose_Loss(nn.Module):
    """Loss definition for joint mask and hierarchical code prediction.

    HccePose 同时预测前/后两套分层编码与前景 mask，因此损失函数既包含
    mask 回归，也包含 front/back 编码的分段 L1 损失。这里还维护了一个
    误差历史窗口，用于动态平衡不同编码位的权重。

    HccePose predicts front/back hierarchical codes together with a foreground
    mask. The loss therefore combines mask regression and segmented L1 losses
    for front/back codes, while also tracking recent error histories to adapt
    per-bit weighting over time.
    """
    def __init__(self, ):
        '''
        ---
        ---
        Initialize L1 losses, mask loss, and per-axis error history containers.
        ---
        ---
        初始化 L1 损失、mask 损失以及各坐标轴误差历史容器。
        '''
        
        super().__init__()
        
        self.Front_error_list = [[], [], []]
        self.Back_error_list = [[], [], []]
        
        self.current_front_error_ratio = [None, None, None]
        self.current_back_error_ratio = [None, None, None]
        
        self.weight_front_error_ratio = [None, None, None]
        self.weight_back_error_ratio = [None, None, None]
        
        self.Front_L1Loss = nn.L1Loss(reduction='none')
        self.Back_L1Loss = nn.L1Loss(reduction='none')
        
        self.mask_loss = nn.L1Loss(reduction="mean")
        
        self.activation_function = torch.nn.Sigmoid()
        
        pass
    
    def cal_error_ratio(self, pred_code, gt_code, pred_mask):
        '''
        ---
        ---
        Per-channel mean absolute error ratio inside the foreground mask.
        ---
        ---
        Binarizes predictions and ground truth, weights errors by `pred_mask`,
        and normalizes by foreground pixel count.

        Args:
            - pred_code: Predicted code logits or activations.
            - gt_code: Ground-truth binary codes.
            - pred_mask: Foreground mask tensor.

        Returns:
            - error_ratio: Per-channel error ratios (summed over spatial dims).
        ---
        ---
        在前景 mask 内按通道计算平均绝对误差比例。
        ---
        ---
        对预测与真值做二值化，用 `pred_mask` 加权误差，并按前景像素数归一化。

        参数:
            - pred_code: 预测编码 logits 或激活。
            - gt_code: 真值二进制编码。
            - pred_mask: 前景 mask 张量。

        返回:
            - error_ratio: 各通道误差比例（空间维已求和）。
        '''
        pred_mask  = pred_mask.clone().detach().round().clamp(0,1) 
        pred_code = torch.sigmoid(pred_code).clone().detach().round().clamp(0,1)
        gt_code = gt_code.clone().detach().round().clamp(0,1)
        error = torch.abs(pred_code-gt_code)*pred_mask
        error_ratio = error.sum([0,2,3])/(pred_mask.sum()+1)
        return error_ratio
    
    def print_error_ratio(self, ):
        '''
        ---
        ---
        Print moving-average front/back error ratios for each axis (debug helper).
        ---
        ---
        打印各轴 front/back 误差的滑动平均（调试辅助）。
        '''
        if self.weight_front_error_ratio[0] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('front(x) error: {}'.format(self.weight_front_error_ratio[0].detach().cpu().numpy()))
        if self.weight_front_error_ratio[1] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('front(y) error:{}'.format(self.weight_front_error_ratio[1].detach().cpu().numpy()))
        if self.weight_front_error_ratio[2] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('front(z) error:{}'.format(self.weight_front_error_ratio[2].detach().cpu().numpy()))

        if self.weight_back_error_ratio[0] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('back(x) error:{}'.format(self.weight_back_error_ratio[0].detach().cpu().numpy()))
        if self.weight_back_error_ratio[1] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('back(y) error:{}'.format(self.weight_back_error_ratio[1].detach().cpu().numpy()))
        if self.weight_back_error_ratio[2] is not None:
            np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
            print('back(z) error:{}'.format(self.weight_back_error_ratio[2].detach().cpu().numpy()))
        return
    
    def forward(self, pred_front, pred_back, pred_mask, gt_front, gt_back, gt_mask,):
        '''
        ---
        ---
        Compute mask L1 plus dynamically weighted front/back code L1 losses.
        ---
        ---
        Updates per-bit error histories, builds exponential weights, and returns
        a dict with `mask_loss`, `Front_L1Losses`, and `Back_L1Losses`.

        Args:
            - pred_front / pred_back: Network code predictions.
            - pred_mask: Predicted mask logits.
            - gt_front / gt_back / gt_mask: Ground-truth codes and mask.

        Returns:
            - dict with scalar / vector losses for the trainer.
        ---
        ---
        计算 mask 的 L1 以及按历史动态加权的 front/back 编码 L1 损失。
        ---
        ---
        更新逐位误差历史、构造指数权重，并返回包含 `mask_loss`、
        `Front_L1Losses`、`Back_L1Losses` 的字典。

        参数:
            - pred_front / pred_back: 网络编码预测。
            - pred_mask: 预测 mask logits。
            - gt_front / gt_back / gt_mask: 真值编码与 mask。

        返回:
            - 供训练器使用的标量/向量损失字典。
        '''
        
        pred_mask_for_loss = pred_mask[:, 0, :, :]
        pred_mask_for_loss = torch.sigmoid(pred_mask_for_loss)
        mask_loss_v = self.mask_loss(pred_mask_for_loss, gt_mask)
        
        pred_mask_prob = self.activation_function(pred_mask)
        pred_mask_prob = pred_mask_prob.detach().clone().round().clamp(0,1)
        pred_mask = pred_mask_prob
          
        Front_L1Loss_v_l = []
        Back_L1Loss_v_l = []
        
        for k in range(3):
            front_error_ratio = self.cal_error_ratio(pred_front[:, k*8:(k+1)*8], gt_front[:, k*8:(k+1)*8], pred_mask)
            self.current_front_error_ratio[k] = front_error_ratio.clone().detach()
            if self.weight_front_error_ratio[k] is None:
                self.weight_front_error_ratio[k]  = front_error_ratio.clone().detach()
                for i in range(pred_front[:, k*8:(k+1)*8].shape[1]):
                    self.Front_error_list[k].append(FixedSizeList(100))
                    self.Front_error_list[k][i].append(self.current_front_error_ratio[k][i].cpu().numpy())
            else:
                for i in range(pred_front[:, k*8:(k+1)*8].shape[1]):
                    self.Front_error_list[k][i].append(self.current_front_error_ratio[k][i].cpu().numpy())
                    self.weight_front_error_ratio[k][i] = np.mean(self.Front_error_list[k][i].data)
                
            back_error_ratio = self.cal_error_ratio(pred_back[:, k*8:(k+1)*8], gt_back[:, k*8:(k+1)*8], pred_mask)
            self.current_back_error_ratio[k] = back_error_ratio.clone().detach()
            if self.weight_back_error_ratio[k] is None:
                self.weight_back_error_ratio[k]  = back_error_ratio.clone().detach()
                for i in range(pred_back[:, k*8:(k+1)*8].shape[1]):
                    self.Back_error_list[k].append(FixedSizeList(100))
                    self.Back_error_list[k][i].append(self.current_back_error_ratio[k][i].cpu().numpy())
            else:
                for i in range(pred_back[:, k*8:(k+1)*8].shape[1]):
                    self.Back_error_list[k][i].append(self.current_back_error_ratio[k][i].cpu().numpy())
                    self.weight_back_error_ratio[k][i] = np.mean(self.Back_error_list[k][i].data)
        
        
            weight_front_error_ratio = torch.exp(torch.minimum(self.weight_front_error_ratio[k],0.51-self.weight_front_error_ratio[k]) * 3).detach().clone()
            weight_back_error_ratio = torch.exp(torch.minimum(self.weight_back_error_ratio[k],0.51-self.weight_back_error_ratio[k]) * 3).detach().clone()
        
            Front_L1Loss_v = self.Front_L1Loss(pred_front[:, k*8:(k+1)*8]*pred_mask.detach().clone(),(gt_front[:, k*8:(k+1)*8] *2 -1)*pred_mask.detach().clone())
            Front_L1Loss_v = Front_L1Loss_v.mean([0,2,3])
            Front_L1Loss_v = torch.sum(Front_L1Loss_v*weight_front_error_ratio)/torch.sum(weight_front_error_ratio)
        
            Back_L1Loss_v = self.Back_L1Loss(pred_back[:, k*8:(k+1)*8]*pred_mask.detach().clone(),(gt_back[:, k*8:(k+1)*8] *2 -1)*pred_mask.detach().clone())
            Back_L1Loss_v = Back_L1Loss_v.mean([0,2,3])
            Back_L1Loss_v = torch.sum(Back_L1Loss_v*weight_back_error_ratio)/torch.sum(weight_back_error_ratio)

            Front_L1Loss_v_l.append(Front_L1Loss_v[None])
            Back_L1Loss_v_l.append(Back_L1Loss_v[None])
        
        Front_L1Loss_v_l = torch.cat(Front_L1Loss_v_l, dim = 0).view(-1)
        Back_L1Loss_v_l = torch.cat(Back_L1Loss_v_l, dim = 0).view(-1)
        
        return {
            'mask_loss' : mask_loss_v, 
            'Front_L1Losses' : Front_L1Loss_v_l,
            'Back_L1Losses' : Back_L1Loss_v_l,
        }


class HccePose_BF_Net(nn.Module):
    """Main HccePose network wrapper with decode and batch inference helpers.

    该类在 `DeepLabV3` 主网络外层封装了 front/back 编码解码、二值 mask
    后处理、2D 坐标图恢复，以及与物体尺寸绑定的三维编码恢复，方便测试
    阶段直接调用 `inference_batch(...)` 获取后处理后的结果。

    This wrapper extends the `DeepLabV3` backbone with HccePose-specific
    decoding utilities, including front/back code decoding, binary mask
    post-processing, 2D coordinate recovery and object-scale 3D code recovery,
    so test-time code can directly call `inference_batch(...)`.
    """
    def __init__(
        self, 
        efficientnet_key = None, 
        input_channels = 3,
        min_xyz = None,
        size_xyz = None,
    ):
        '''
        ---
        ---
        Build `DeepLabV3` with 48+1 output channels and store object 3D bounds.
        ---
        ---
        Args:
            - efficientnet_key: Backbone selector passed to `DeepLabV3`.
            - input_channels: Network input channels.
            - min_xyz / size_xyz: Object AABB in model coordinates for decoding.
        ---
        ---
        构建输出通道为 48+1 的 `DeepLabV3`，并保存物体三维包围盒参数。
        ---
        ---
        参数:
            - efficientnet_key: 传给 `DeepLabV3` 的主干选择项。
            - input_channels: 网络输入通道数。
            - min_xyz / size_xyz: 解码用的物体坐标系 AABB。
        '''
        super(HccePose_BF_Net, self).__init__()
        self.net = DeepLabV3(48 + 1,  efficientnet_key=efficientnet_key, input_channels=input_channels)
        
        self.min_xyz = min_xyz
        self.size_xyz = size_xyz
        self.powers = None
        self.coord_image = None
        self.activation_function = torch.nn.Sigmoid()

    def forward(self, inputs):
        '''
        ---
        ---
        Delegate to the inner `DeepLabV3` network.
        ---
        ---
        Args:
            - inputs: Normalized crop tensor.

        Returns:
            - mask, binary_code: Same as `DeepLabV3.forward`.
        ---
        ---
        转调内部 `DeepLabV3` 网络。
        ---
        ---
        参数:
            - inputs: 归一化后的裁剪张量。

        返回:
            - mask, binary_code: 与 `DeepLabV3.forward` 相同。
        '''
        return self.net(inputs)
    
    def hcce_decode_v0(self, class_code_images_pytorch, class_base=2):
        '''
        ---
        ---
        NumPy-based hierarchical code decoding (legacy path, per-axis chain).
        ---
        ---
        Converts binarized code maps into 3-channel class indices on CPU, then
        moves the result back to the input device.

        Args:
            - class_code_images_pytorch: Code tensor (moved to CPU internally).
            - class_base: Binary base for bit weights (default 2).

        Returns:
            - class_id_image_2: Decoded 3-channel index map as a torch tensor.
        ---
        ---
        基于 NumPy 的分层编码解码（旧路径，按轴链式差分）。
        ---
        ---
        在 CPU 上将二值化编码图转为 3 通道类别索引，再搬回输入设备。

        参数:
            - class_code_images_pytorch: 编码张量（内部会搬到 CPU）。
            - class_base: 位权重的进制（默认 2）。

        返回:
            - class_id_image_2: 解码后的 3 通道索引图（torch）。
        '''
        
        class_code_images = class_code_images_pytorch.detach().cpu().numpy()
        class_id_image_2 = np.zeros((class_code_images.shape[0], class_code_images.shape[1],class_code_images.shape[2], 3))
        codes_length = int(class_code_images.shape[3]/3) 
        
        class_id_image_2[...,0] = class_id_image_2[...,0] + class_code_images[...,0] * (class_base**(codes_length - 1 - 0))
        temp2 = class_code_images[...,0]
        for i in range(codes_length-1):
            temp2 = class_code_images[...,i+1] - temp2
            temp2 = np.abs(temp2)
            class_id_image_2[...,0] = class_id_image_2[...,0] + temp2 * (class_base**(codes_length - 1 - i - 1))
        
        class_id_image_2[...,1] = class_id_image_2[...,1] + class_code_images[...,0+codes_length] * (class_base**(codes_length - 1 - 0))
        temp2 = class_code_images[...,0+codes_length]
        for i in range(codes_length - 1):
            temp2 = class_code_images[...,i+codes_length+1] - temp2
            temp2 = np.abs(temp2)
            class_id_image_2[...,1] = class_id_image_2[...,1] + temp2 * (class_base**(codes_length - 1 - i - 1))

        class_id_image_2[...,2] = class_id_image_2[...,2] + class_code_images[...,0+codes_length*2] * (class_base**(codes_length - 1 - 0))
        temp2 = class_code_images[...,0+codes_length*2]
        for i in range(codes_length-1):
            temp2 = class_code_images[...,i+codes_length*2+1] - temp2
            temp2 = np.abs(temp2)
            class_id_image_2[...,2] = class_id_image_2[...,2] + temp2 * (class_base**(codes_length - 1 - i - 1))

        class_id_image_2 = torch.from_numpy(class_id_image_2).to(class_code_images_pytorch.device)
        return class_id_image_2

    def hcce_decode(self, class_code_images):
        '''
        ---
        ---
        Fully torch hierarchical code decoding using absolute successive differences.
        ---
        ---
        For each RGB axis segment, accumulates weighted absolute differences between
        neighboring bit planes to recover an integer code per pixel.

        Args:
            - class_code_images: Tensor shaped (B, H, W, C) with C divisible by 3.

        Returns:
            - class_id_image: Tensor (B, H, W, 3) of decoded scalar codes per axis.
        ---
        ---
        全 PyTorch 分层编码解码：相邻比特位绝对差分加权求和。
        ---
        ---
        对每个 RGB 轴对应的码段，对相邻位平面做加权绝对差分，恢复每像素整数码。

        参数:
            - class_code_images: 形状 (B, H, W, C) 且 C 可被 3 整除。

        返回:
            - class_id_image: 每轴解码标量码，形状 (B, H, W, 3)。
        '''
        class_base = 2
        
        batch_size, height, width, channels = class_code_images.shape
        codes_length = channels // 3 

        class_id_image = torch.zeros_like(class_code_images[..., :3])

        if self.powers is None:
            device = class_code_images.device
            powers = torch.pow(
                torch.tensor(class_base, device=device, dtype=torch.float32),
                torch.arange(codes_length-1, -1, -1, device=device)
            )
        for c in range(3):
            start_idx = c * codes_length
            end_idx = start_idx + codes_length
            codes = class_code_images[..., start_idx:end_idx]
            diffs = torch.zeros_like(codes)
            diffs[..., 0] = codes[..., 0]
            for k in range(1, codes_length):
                diffs[..., k] = torch.abs(codes[..., k] - diffs[..., k-1])
            class_id_image[..., c] = torch.sum(diffs * powers, dim=-1)
        
        return class_id_image
    
    # @torch.inference_mode()
    def inference_batch(self, inputs, Bbox, thershold=0.5):
        '''
        ---
        ---
        Run one batched HccePose forward pass and decode all pose-related tensors.
        ---
        ---
        Receives normalized crops and image-space boxes, thresholds mask and codes,
        decodes hierarchical bits, maps normalized UV to image coordinates, and
        scales codes into object-space 3D using `min_xyz` and `size_xyz`.

        Args:
            - inputs: Normalized crop batch tensor.
            - Bbox: Tensor of shape (B, 4) as [x, y, w, h] in the original image.
            - thershold: Sigmoid threshold for mask and code binarization.

        Returns:
            - dict: Keys include `pred_mask`, `coord_2d_image`, object-space
              front/back codes, raw logits, etc.
        ---
        ---
        执行一批 HccePose 前向并完成与位姿相关的全部解码。
        ---
        ---
        输入归一化裁剪与原图框，对 mask 与编码阈值化、分层解码，将归一化 UV
        映射到图像坐标，并用 `min_xyz`、`size_xyz` 缩放到物体坐标系 3D。

        参数:
            - inputs: 归一化裁剪 batch 张量。
            - Bbox: 形状 (B, 4)，原图空间 [x, y, w, h]。
            - thershold: mask 与编码二值化的 Sigmoid 阈值。

        返回:
            - dict: 含 `pred_mask`、`coord_2d_image`、物体空间正背面坐标、
              原始 logits 等键。
        '''

        pred_mask, pred_front_back_code = self.net(inputs)
        pred_mask_logits = pred_mask
        pred_mask = self.activation_function(pred_mask)
        pred_mask[pred_mask > thershold] = 1.0
        pred_mask[pred_mask <= thershold] = 0.0
        pred_mask = pred_mask[:, 0, ...]
        
        pred_front_back_code_logits = pred_front_back_code
        pred_front_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,:24]
        pred_back_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,24:]
        
        pred_front_back_code = self.activation_function(pred_front_back_code)
        pred_front_back_code[pred_front_back_code > thershold] = 1.0
        pred_front_back_code[pred_front_back_code <= thershold] = 0.0
        
        pred_front_back_code = pred_front_back_code.permute(0, 2, 3, 1)
        pred_front_code = pred_front_back_code[...,:24]
        pred_back_code = pred_front_back_code[...,24:]
        pred_front_code = self.hcce_decode(pred_front_code) / 255
        pred_back_code = self.hcce_decode(pred_back_code) / 255
        if self.coord_image is None:
            x = torch.arange(pred_front_code.shape[2] , device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[2] 
            y = torch.arange(pred_front_code.shape[1] , device=pred_front_code.device).to(torch.float32) / pred_front_code.shape[1] 
            X, Y = torch.meshgrid(x, y, indexing='xy')  
            self.coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1) 
        coord_image = self.coord_image[None,...].repeat(pred_front_code.shape[0],1,1,1)
        coord_image[..., 0] = coord_image[..., 0] * Bbox[:, None, None, 2] + Bbox[:, None, None, 0]
        coord_image[..., 1] = coord_image[..., 1] * Bbox[:, None, None, 3] + Bbox[:, None, None, 1]
        pred_front_code_0 = pred_front_code * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        pred_back_code_0 = pred_back_code * self.size_xyz[None,None,None] + self.min_xyz[None,None,None]
        
        return {
            'pred_mask' : pred_mask,
            'coord_2d_image' : coord_image,
            'pred_front_code_obj' : pred_front_code_0,
            'pred_back_code_obj' : pred_back_code_0,
            'pred_front_code' : pred_front_code,
            'pred_back_code' : pred_back_code,
            'pred_front_code_raw' : pred_front_code_raw,
            'pred_back_code_raw' : pred_back_code_raw,
            'pred_mask_logits' : pred_mask_logits,
            'pred_front_back_code_logits' : pred_front_back_code_logits,
        }


def save_checkpoint(path, net, iteration_step, best_score, optimizer, max_to_keep, keypoints_ = None, w_optimizer = True):
    '''
    ---
    ---
    Save a regular training checkpoint and prune old numeric-named files.
    ---
    ---
    Creates `path` if needed, removes the oldest checkpoint when the count exceeds
    `max_to_keep`, and stores `model_state_dict` plus optional optimizer and
    metadata under filename `iteration_step`.

    Args:
        - path: Directory to store checkpoints.
        - net: Model (handles DataParallel / DDP wrappers).
        - iteration_step: Integer used as the checkpoint filename.
        - best_score: Scalar logged into the checkpoint dict.
        - optimizer: Optimizer whose state is saved when `w_optimizer` is True.
        - max_to_keep: Maximum checkpoints to retain in `path`.
        - keypoints_: Optional array stored when provided.
        - w_optimizer: Whether to include `optimizer_state_dict`.

    Returns:
        - None (writes files to disk).
    ---
    ---
    保存常规训练 checkpoint，并按数量删除最旧的数字命名文件。
    ---
    ---
    若目录不存在则创建；当文件数超过 `max_to_keep` 时删除最旧 checkpoint；
    以训练步数 `iteration_step` 为文件名写入 `model_state_dict` 及可选优化器
    与元数据。

    参数:
        - path: checkpoint 保存目录。
        - net: 模型（兼容 DataParallel / DDP）。
        - iteration_step: 用作文件名的整数步数。
        - best_score: 写入字典的标量分数。
        - optimizer: `w_optimizer` 为 True 时保存其状态。
        - max_to_keep: 目录内最多保留的 checkpoint 数量。
        - keypoints_: 可选，提供时一并保存。
        - w_optimizer: 是否写入 `optimizer_state_dict`。

    返回:
        - 无（仅写磁盘）。
    '''
    
    if not os.path.isdir(path):
        os.makedirs(path)
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt = [int(i) for i in saved_ckpt]
    saved_ckpt.sort()
    
    num_saved_ckpt = len(saved_ckpt)
    if num_saved_ckpt >= max_to_keep:
        os.remove(os.path.join(path, str(saved_ckpt[0])))

    if isinstance(net, torch.nn.parallel.DataParallel):
        state_dict = net.module.state_dict()
    elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    if w_optimizer:
        if keypoints_ is None:
            torch.save(
                        {
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration_step': iteration_step,
                        'best_score': best_score
                        }, 
                        os.path.join(path, str(iteration_step))
                    )
        else:
            torch.save(
                        {
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iteration_step': iteration_step,
                        'best_score': best_score,
                        'keypoints_' : keypoints_.tolist(),
                        }, 
                        os.path.join(path, str(iteration_step))
                    )
    else:
        if keypoints_ is None:
            torch.save(
                        {
                        'model_state_dict': state_dict,
                        'iteration_step': iteration_step,
                        'best_score': best_score
                        }, 
                        os.path.join(path, str(iteration_step))
                    )
        else:
            torch.save(
                        {
                        'model_state_dict': state_dict,
                        'iteration_step': iteration_step,
                        'best_score': best_score,
                        'keypoints_' : keypoints_.tolist(),
                        }, 
                        os.path.join(path, str(iteration_step))
                    )
    
def get_checkpoint(path):
    '''
    ---
    ---
    Pick the checkpoint file whose filename encodes the highest score.
    ---
    ---
    Parses numeric prefixes before the `step` substring in each filename and
    returns the full path of the argmax file.

    Args:
        - path: Directory containing best-score checkpoints.

    Returns:
        - Full path to the selected checkpoint file.
    ---
    ---
    选择文件名中所编码分数最高的 checkpoint 文件路径。
    ---
    ---
    解析每个文件名中 `step` 子串前的数值前缀，返回分数最大文件的完整路径。

    参数:
        - path: 存放 best checkpoint 的目录。

    返回:
        - 被选中的 checkpoint 文件的完整路径。
    '''
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt_s = [float(i.split('step')[0].replace('_', '.')) for i in saved_ckpt]
    saved_ckpt_id = np.argmax(saved_ckpt_s)
    return os.path.join(path, saved_ckpt[saved_ckpt_id])

def save_best_checkpoint(best_score_path, net, optimizer, best_score, iteration_step, keypoints_ = None, w_optimizer = True):
    '''
    ---
    ---
    Overwrite the single best checkpoint with score-embedded filename.
    ---
    ---
    Deletes any existing file in `best_score_path`, then saves a new checkpoint
    whose name concatenates formatted `best_score` and `iteration_step`.

    Args:
        - best_score_path: Directory for the lone best file.
        - net, optimizer: Same semantics as `save_checkpoint`.
        - best_score, iteration_step: Encoded into the output filename.
        - keypoints_, w_optimizer: Optional extras mirroring `save_checkpoint`.

    Returns:
        - None.
    ---
    ---
    用带分数的文件名覆盖保存唯一的 best checkpoint。
    ---
    ---
    若目录中已有文件则先删除，再保存新 checkpoint，文件名由格式化后的
    `best_score` 与 `iteration_step` 拼接而成。

    参数:
        - best_score_path: 仅保存一个 best 文件的目录。
        - net, optimizer: 与 `save_checkpoint` 含义相同。
        - best_score, iteration_step: 写入文件名。
        - keypoints_, w_optimizer: 与 `save_checkpoint` 一致的可选项。

    返回:
        - 无。
    '''
    saved_ckpt = [f for f in os.listdir(best_score_path) if os.path.isfile(os.path.join(best_score_path, f))]
    if saved_ckpt != []:
        os.remove(os.path.join(best_score_path, saved_ckpt[0]))

    best_score_file_name = '{:.4f}'.format(best_score)
    best_score_file_name = best_score_file_name.replace('.', '_')
    best_score_file_name = best_score_file_name + 'step'
    best_score_file_name = best_score_file_name + str(iteration_step)
    if isinstance(net, torch.nn.parallel.DataParallel):
        state_dict = net.module.state_dict()
    elif isinstance(net, torch.nn.parallel.DistributedDataParallel):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    if w_optimizer:
        if keypoints_ is None:
            torch.save(
                {
                    'model_state_dict': state_dict, #net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'iteration_step': iteration_step
                }, 
                os.path.join(best_score_path, best_score_file_name)
            )
        else:
            torch.save(
                {
                    'model_state_dict': state_dict, #net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'iteration_step': iteration_step,
                    'keypoints_' : keypoints_.tolist(),
                }, 
                os.path.join(best_score_path, best_score_file_name)
            )
    else:
        if keypoints_ is None:
            torch.save(
                {
                    'model_state_dict': state_dict, #net.state_dict(),
                    'best_score': best_score,
                    'iteration_step': iteration_step
                }, 
                os.path.join(best_score_path, best_score_file_name)
            )
        else:
            torch.save(
                {
                    'model_state_dict': state_dict, #net.state_dict(),
                    'best_score': best_score,
                    'iteration_step': iteration_step,
                    'keypoints_' : keypoints_.tolist(),
                }, 
                os.path.join(best_score_path, best_score_file_name)
            )

    print("best check point saved in ", os.path.join(best_score_path, best_score_file_name))

def load_checkpoint(check_point_path, net : HccePose_BF_Net, optimizer=None, local_rank=0, CUDA_DEVICE='0'):
    '''
    ---
    ---
    Load the highest-scoring checkpoint from a directory into `net`.
    ---
    ---
    Uses `get_checkpoint`, loads onto `cuda:CUDA_DEVICE`, restores optional
    optimizer state, and returns metadata dict. Prints a notice when loading fails.

    Args:
        - check_point_path: Directory searched by `get_checkpoint`.
        - net: Target `HccePose_BF_Net` instance.
        - optimizer: If not None, load `optimizer_state_dict` when present.
        - local_rank: Rank used to gate print on failure.
        - CUDA_DEVICE: CUDA device index string.

    Returns:
        - dict with `best_score`, `iteration_step`, `keypoints_` (possibly defaults).
    ---
    ---
    将目录中得分最高的 checkpoint 加载到 `net`。
    ---
    ---
    通过 `get_checkpoint` 解析路径，在 `cuda:CUDA_DEVICE` 上加载，可选恢复
    优化器状态，并返回元信息字典；失败时在指定 rank 打印提示。

    参数:
        - check_point_path: `get_checkpoint` 所搜索的目录。
        - net: 目标 `HccePose_BF_Net`。
        - optimizer: 非 None 时尝试加载 `optimizer_state_dict`。
        - local_rank: 控制失败时是否打印。
        - CUDA_DEVICE: CUDA 设备编号字符串。

    返回:
        - 含 `best_score`、`iteration_step`、`keypoints_` 的字典（可能为默认）。
    '''
    best_score = 0
    iteration_step = 0
    keypoints_ = []
    try:
        checkpoint = torch.load( get_checkpoint(check_point_path), map_location='cuda:'+CUDA_DEVICE, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_score = checkpoint['best_score']
        iteration_step = checkpoint['iteration_step']
        keypoints_ = checkpoint['keypoints_']
    except:
        if local_rank == 0:
            print('no checkpoint !')
    return {
        'best_score' : best_score,
        'iteration_step' : iteration_step,
        'keypoints_' : keypoints_,
    }
