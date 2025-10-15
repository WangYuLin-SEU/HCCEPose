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

class ASPP(nn.Module):
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

class ASPP_Efficientnet_upsampled(nn.Module):
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

class efficientnet_upsampled(nn.Module):
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
        l2 = self.eff_layer_2(x)
        l3 = self.eff_layer_3(l2)
        l4 = self.eff_layer_4(l3)
        final = self.final_layer(l4)
        return final,l3,l2

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) 

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks)

    return layer

class BasicBlock(nn.Module):
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

        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out))

        out = out + self.downsample(x)

        out = F.relu(out) 

        return out

class ResNet_BasicBlock_OS8(nn.Module):
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
        x_128 = self.resnet_layer_1(x)
        x_64 = self.resnet_layer_2(x_128)
        x_32 = self.resnet_layer_3(x_64)
        x_16 = self.layer4(x_32)
        x_high_feature = self.layer5(x_16)
        return x_high_feature, x_128, x_64, x_32, x_16

class DeepLabV3(nn.Module):
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
    def __init__(self, size):
        self.size = size
        self.data = []
    
    def append(self, item):
        if len(self.data) >= self.size:
            self.data.pop(0)  
        self.data.append(item)
    
    def get_list(self):
        return self.data
    
    def __repr__(self):
        return repr(self.data)
    
class HccePose_Loss(nn.Module):
    def __init__(self, ):
        
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
        pred_mask  = pred_mask.clone().detach().round().clamp(0,1) 
        pred_code = torch.sigmoid(pred_code).clone().detach().round().clamp(0,1)
        gt_code = gt_code.clone().detach().round().clamp(0,1)
        error = torch.abs(pred_code-gt_code)*pred_mask
        error_ratio = error.sum([0,2,3])/(pred_mask.sum()+1)
        return error_ratio
    
    def print_error_ratio(self, ):
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
    def __init__(
        self, 
        efficientnet_key = None, 
        input_channels = 3,
        min_xyz = None,
        size_xyz = None,
    ):
        super(HccePose_BF_Net, self).__init__()
        self.net = DeepLabV3(48 + 1,  efficientnet_key=efficientnet_key, input_channels=input_channels)
        
        self.min_xyz = min_xyz
        self.size_xyz = size_xyz
        self.powers = None
        self.coord_image = None
        self.activation_function = torch.nn.Sigmoid()

    def forward(self, inputs):
        return self.net(inputs)
    
    def hcce_decode_v0(self, class_code_images_pytorch, class_base=2):
        
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
    
    @torch.inference_mode()
    def inference_batch(self, inputs, Bbox, thershold=0.5):

        pred_mask, pred_front_back_code = self.net(inputs)
        pred_mask = self.activation_function(pred_mask).round().clamp(0,1)
        pred_mask = pred_mask[:, 0, ...]
        
        pred_front_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,:24]
        pred_back_code_raw = ((pred_front_back_code.permute(0, 2, 3, 1)+1)/2).clone().clamp(0,1)[...,24:]
        
        pred_front_back_code = self.activation_function(pred_front_back_code).round().clamp(0,1)
        
        pred_front_back_code = pred_front_back_code.permute(0, 2, 3, 1)
        pred_front_code = pred_front_back_code[...,:24]
        pred_back_code = pred_front_back_code[...,24:]
        pred_front_code = self.hcce_decode(pred_front_code) / 255
        pred_back_code = self.hcce_decode(pred_back_code) / 255
        if self.coord_image is None:
            x = torch.arange(pred_front_code.shape[2] , device=pred_front_code.device) / pred_front_code.shape[2] 
            y = torch.arange(pred_front_code.shape[1] , device=pred_front_code.device) / pred_front_code.shape[1] 
            X, Y = torch.meshgrid(x, y, indexing='xy')  
            self.coord_image = torch.cat([X[..., None], Y[..., None]], dim=-1) 
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
        }


def save_checkpoint(path, net, iteration_step, best_score, optimizer, max_to_keep, keypoints_ = None, w_optimizer = True):
    
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
    saved_ckpt = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    saved_ckpt_s = [float(i.split('step')[0].replace('_', '.')) for i in saved_ckpt]
    saved_ckpt_id = np.argmax(saved_ckpt_s)
    return os.path.join(path, saved_ckpt[saved_ckpt_id])

def save_best_checkpoint(best_score_path, net, optimizer, best_score, iteration_step, keypoints_ = None, w_optimizer = True):
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
    best_score = 0
    iteration_step = 0
    keypoints_ = []
    try:
        checkpoint = torch.load( get_checkpoint(check_point_path), map_location='cuda:'+CUDA_DEVICE)
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
