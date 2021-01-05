from torch import nn
import torch


class SEBLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        print("SE_B_Layer")

    def forward(self, x_in, x_out):
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x_out * y.expand_as(x_out)

class SEBLayer_v1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBLayer_v1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())

        print("SE_B_Layer_v1")

    def forward(self, x_in, x_out):
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y_a = self.se_a(x_in)
        return x_out * y.expand_as(x_out) * y_a.expand_as(x_out)

class ALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        print("A_Layer")

    def forward(self, x_in, x_out):
        y = self.se_a(x_in)
        return x_out * y.expand_as(x_out)


class A_relu_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(A_relu_Layer, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.act = nn.ReLU(inplace=True)
        print("A_relu_Layer")

    def forward(self, x_in, x_out):
        A = self.se_a(x_in)
        y = x_out * A.expand_as(x_out)
        y = self.act(y)
        return y

class A_tanh_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(A_tanh_Layer, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.act = nn.Tanh()
        print("A_tanh_Layer")

    def forward(self, x_in, x_out):
        A = self.se_a(x_in)
        y = x_out * A.expand_as(x_out)
        y = self.act(y)
        return y



class ALayer_v1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_v1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())

        print("A_Layer_v1")

    def forward(self, x_in, x_out):
        # A generated from x_in
        A = self.se_a(x_in)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        fold = nn.Fold(x_in.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x_out)
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        out = fold(out_unfold)

        return out

class ALayer_v1_1(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_v1_1, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        # self.weights = torch.
        print("A_Layer_v1.1")

    def forward(self, x, weights):
        # A generated from x_in
        A = self.se_a(x)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        fold = nn.Fold(x.size()[2:], **fold_params)

        # input_ones = torch.ones(x.shape, dtype=x.dtype, device='cuda')
        # divisor = fold(unfold(input_ones))

        inp = torch.randn(128, 64, 8, 8)
        w = torch.randn(64, 64, 3, 3)

        # apply A to x_out
        out_unfold = unfold(x)
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        out = fold(out_unfold)

        return out


class ALayer_v2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_v2, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        print("A_Layer_v2")

    def forward(self, x_in, x_out):
        # A generated from x_in
        A = self.se_a(x_in)
        # B
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # fold preparations
        fold_params = dict(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        unfold = nn.Unfold(**fold_params)
        fold = nn.Fold(x_in.size()[2:], **fold_params)
        # apply A to x_out
        out_unfold = unfold(x_out)
        out_unfold = out_unfold * A.flatten(2, -1).expand(out_unfold.shape)
        out = fold(out_unfold)

        return out


class ALayer_wh(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ALayer_wh, self).__init__()
        self.se_a = nn.Sequential(nn.Conv2d(channel, 16, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.Sigmoid())
        input_w = 8

        self.conv2_wh = nn.Sequential(
            nn.Linear(input_w * channel, input_w * channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_w * channel // 16, input_w, bias=False),
            nn.Sigmoid()
        )
        print("A_Layer_wh-share fc")

    def forward(self, x_in, x_out):
        x_w = torch.mean(x_in, dim=2).unsqueeze(2)
        x_h = torch.mean(x_in, dim=3).unsqueeze(3)

        # compress channels
        x_w = x_w.view(x_w.size(0), -1)
        x_h = x_h.view(x_h.size(0), -1)

        A_w = self.conv2_wh(x_w)
        A_h = self.conv2_wh(x_h)

        A = A_w.unsqueeze(1).unsqueeze(2) * A_h.unsqueeze(1).unsqueeze(3)

        return x_out * A.expand_as(x_out)


class GCN(nn.Module):
    def __init__(self, c, out_c, k=3):  # out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k, 1), padding=((k - 1) / 2, 0))
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k), padding=(0, (k - 1) / 2))
        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1, k), padding=((k - 1) / 2, 0))
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k, 1), padding=(0, (k - 1) / 2))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x