from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
from .abBlock import *


__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock', model_struct=None):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        if model_struct is not None:
            if model_struct == "senet":
                block = ABBlock_se
            elif model_struct == "single_B":
                block = ABBlock_B
            elif model_struct == "single_A":
                block = ABBlock_A
            elif model_struct =="AB":
                block = ABBlock_AB
            elif model_struct =="AB_BN":
                block = ABBlock_AB_BN
            elif model_struct =="A_se":
                block = ABBlock_Ase
            elif model_struct == "A1":
                block = ABBlock_A1
            elif model_struct == "A1_se":
                block = ABBlock_A1se
            elif model_struct == "A1B":
                block = ABBlock_A1B
            elif model_struct == "Awh":
                block = ABBlock_Awh
            elif model_struct == "AwhB":
                block = ABBlock_AwhB
            elif model_struct == "Awh_se":
                block = ABBlock_Awh_se

            elif model_struct == "AB_no_conv2":
                block = ABBlock_AB_no_conv2

            elif model_struct == "no_conv2":
                block = ABBlock_no_conv2
            elif model_struct == "AB_rand_conv2":
                block = ABBlock_AB_rand_conv2

            elif model_struct == "A_relu":
                block = ABBlock_A_relu
            elif model_struct == "A_reluB":
                block = ABBlock_A_reluB

            elif model_struct == "A_tanh":
                block = ABBlock_A_tanh

            elif model_struct == "A_tanhB":
                block = ABBlock_A_tanhB

            elif model_struct == "A1.1":
                block = ABBlock_A1_1


            elif model_struct == "B1":
                block = ABBlock_B1


            elif model_struct == "dcn":
                block = ABBlock_dcn

            elif model_struct == "DR1B":
                block = ABBlock_DR1B
            elif model_struct == "DR1_full":
                block = ABBlock_DR1_full
            elif model_struct == "DR1_full_rand":
                block = ABBlock_DR1_full_rand
            elif model_struct == "DR1_strict":
                block = ABBlock_DR1_strict
            elif model_struct == "DR1_strict_rand":
                block = ABBlock_DR1_strict_rand
            elif model_struct == "DR1_ABconv_randB":
                block = ABBlock_DR1_ABconv_randB
            elif model_struct == "DR1_rand_B":
                block = ABBlock_DR1_rand_B
            elif model_struct == "DR1_v1B":
                block = ABBlock_DR1_v1B
            elif model_struct == "DR1_v1_lightB":
                block = ABBlock_DR1_v1_lightB
            elif model_struct == "DR1_v1_light_randB":
                block = ABBlock_DR1_v1_light_randB
            elif model_struct == "DR1_v1_randB":
                block = ABBlock_DR1_v1_randB

            elif model_struct == "ABconv":
                block = ABBlock_ABconv
            elif model_struct == "rand_conv":
                block = ABBlock_rand_conv
            elif model_struct == "AB_as_conv":
                block = ABBlock_AB_as_conv
            elif model_struct == "AB_as_conv_res":
                block = ABBlock_AB_as_conv_res

            elif model_struct == "AB_as_conv3":
                block = ABBlock_AB_as_conv3
            elif model_struct == "ABconv_rand":
                block = ABBlock_ABconv_rand

            elif model_struct == "ABconv_res_rand":
                block = ABBlock_ABconv_res_rand
            elif model_struct == "ABconv3_rand":
                block = ABBlock_ABconv3_rand
            elif model_struct == "ABconv_rand_binary":
                block = ABBlock_ABconv_rand_binary
            else:
                assert 0, f"block '{model_struct}' is not supported!"

        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if m.weight.requires_grad:
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, model_struct=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
