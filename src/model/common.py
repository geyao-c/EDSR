import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from model import ecb
# from model.ecb import ECB
from model.modifyecb import ECB

class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
        # (batch_size, num_groups, chs_per_group, h, w)
        x = x.transpose(1, 2)  # dim_1 and dim_2
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class GroupResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, groups=2):

        super(GroupResBlock, self).__init__()
        m = []
        for i in range(2):
            # m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias, groups=groups))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(Channel_Shuffle(groups))
                m.append(act)
            elif i == 1:
                m.append(Channel_Shuffle(groups))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1,
                 dilation: int = 1):
        padding = (kernel_size - 1) // 2 * dilation
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_planes


class DWConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super(DWConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.out_channels = out_planes


class MobileV1ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MobileV1ResBlock, self).__init__()
        m = []

        # -------------------------------
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias, groups=n_feats))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)

        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=bias, groups=1))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)

        # -------------------------------
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias, groups=n_feats))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)

        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=bias, groups=1))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class GroupMobileV1ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(GroupMobileV1ResBlock, self).__init__()
        m = []

        # -------------------------------
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias, groups=2))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)

        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=bias, groups=1))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)

        # -------------------------------
        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=bias, groups=2))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))
        m.append(act)

        m.append(nn.Conv2d(n_feats, n_feats, kernel_size=1, padding=0, bias=bias, groups=1))
        if bn:
            m.append(nn.BatchNorm2d(n_feats))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ECBResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1, with_idt=False, bias_type=True, groups=1):

        super(ECBResBlock, self).__init__()
        m = []
        for i in range(2):
            # m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            m.append(ECB(n_feats, n_feats, depth_multiplier=2.0, act_type='linear', with_idt=with_idt, bias_type=bias_type,
                         groups=groups))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                if groups != 1:
                    m.append(Channel_Shuffle(groups))
                m.append(act)
            elif i == 1:
                if groups != 1:
                    m.append(Channel_Shuffle(groups))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ECBUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, with_idt=False, bias_type=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                # m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(ECB(n_feats, 4 * n_feats, depth_multiplier=2.0, act_type='linear', with_idt=with_idt, bias_type=bias_type))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            # m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(ECB(n_feats, 9 * n_feats, depth_multiplier=2.0, act_type='linear', with_idt=with_idt, bias_type=bias_type))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(ECBUpsampler, self).__init__(*m)

