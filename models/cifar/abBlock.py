import torch.nn as nn
from .se_module import *

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_rand(in_planes, out_planes, stride=1):
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    conv.weight.requires_grad = False

    return conv

class ABBlock_se(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_se, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_b = SEBLayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out2, out2)  # se

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ABBlock_B(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_B, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_b = SEBLayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # B only

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_B1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_B1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_b = SEBLayer_v1(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # B only

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ABBlock_A(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_A_relu(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A_relu, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = A_relu_Layer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_A_tanh(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A_tanh, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = A_tanh_Layer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ABBlock_A1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_v1(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_A1_1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A1_1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_v1_1(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ABBlock_Awh(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_Awh, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_wh(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out = self.bn2(out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ABBlock_AB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_AB, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_A_reluB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A_reluB, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = A_relu_Layer(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_A_tanhB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A_tanhB, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = A_tanh_Layer(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ABBlock_AB_rand_conv2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_AB_rand_conv2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_rand(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # print("conv1:", self.conv1.weight[0][0])
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)  # fixed_rand_weight
        # print("conv2:", self.conv2.weight[0][0])

        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ABBlock_AB_no_conv2(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_AB_no_conv2, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = out1  # no conv2
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ABBlock_A1B(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A1B, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_v1(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)
        out1 = self.se_a(out1, out1)  # a
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ABBlock_A1se(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_A1se, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_v1(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)
        out1 = self.se_a(out1, out1)  # a
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)

        out = self.se_b(out2, out2)  # se

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ABBlock_AwhB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_AwhB, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_wh(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)
        out1 = self.se_a(out1, out1)  # a
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out1, out2)  # se_b
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ABBlock_Awh_se(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_Awh_se, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer_wh(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)
        out1 = self.se_a(out1, out1)  # a
        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out2, out2)  # se
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ABBlock_Ase(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, ):
        super(ABBlock_Ase, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se_a = ALayer(planes, reduction)
        self.se_b = SEBLayer(planes, reduction)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu(out)

        out1 = self.se_a(out1, out1)  # a

        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out = self.se_b(out2, out2)  # se


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out