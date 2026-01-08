import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import numpy as np
from einops import rearrange, reduce

from ..backbone.pkinet import DropPath
from ..modules.conv import Conv, DWConv,autopad
from ..modules.block import *
from .attention import *
from .rep_block import *
from .orepa import *
from .RFAconv import *
from .wtconv2d import *
from .metaformer import *


__all__ = [ 'Enhance','MANet','MANet_PD', 'Enhance_AFCA']

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p


######################################### Ehance_AFCA start ########################################
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out


class AFCA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(AFCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)  # (1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)  # (1,1,64)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)  # (1,64,1,1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input * out


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='silu')
        nn.init.constant_(self.bn.weight, 1.0)
        nn.init.constant_(self.bn.bias, 0.0)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# Enhance_AFCA
class Enhance_AFCA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c1 = int(c1)
        c2 = int(c2)

        try:
            n = max(1, round(float(n)))
        except (ValueError, TypeError):
            n = 1
        try:
            e = float(e)
            if not (0 < e <= 1):
                e = 0.55
        except (ValueError, TypeError):
            e = 0.55

        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # self.m = nn.ModuleList(
        #     Enhance(self.c, [2,4,6]) for _ in range(n)
        # )

        self.m = nn.ModuleList(
            Enhance(self.c, [1,3,5]) for _ in range(n)
        )

        self.attention = AFCA(c2, b=1, gamma=2) if c2 >= 32 else None

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        output = self.cv2(torch.cat(y, 1))

        if self.attention is not None:
            output = self.attention(output)

        return output

######################################### Ehance_AFCA end ##############


class EdgeEnhancer(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, 1, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class Enhance(nn.Module):

    def __init__(self, inc, bins):
        super().__init__()
        num_bins = len(bins)
        split_channels = max(8, inc // num_bins) 

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, split_channels, 1),
                Conv(split_channels, split_channels, 3, g=split_channels)
            ) for bin in bins
        ])

        self.ees = nn.ModuleList([EdgeEnhancer(split_channels) for _ in bins])

        self.local_conv = Conv(inc, inc, 3)

        concat_channels = inc + num_bins * split_channels

        self.final_conv = Conv(concat_channels, inc, 1)

        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]

        for idx, f in enumerate(self.features):
            feat = f(x)
            feat = self.ees[idx](feat)
            feat_up = F.interpolate(feat, x_size[2:], mode='bilinear', align_corners=True)
            out.append(feat_up)

        concat_feat = torch.cat(out, 1)
        result = self.final_conv(concat_feat)

        result = x + self.alpha * result

        return result
######################################### Ehance_AFCA end ##############


# ###################################################################
class MANet(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv_first = Conv(c1, 2 * self.c, 1, 1)
        self.cv_final = Conv((4 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv_block_1 = Conv(2 * self.c, self.c, 1, 1)
        dim_hid = int(p * 2 * self.c)
        self.cv_block_2 = nn.Sequential(Conv(2 * self.c, dim_hid, 1, 1), DWConv(dim_hid, dim_hid, kernel_size, 1),
                                      Conv(dim_hid, self.c, 1, 1))
    def forward(self, x):
        y = self.cv_first(x)
        y0 = self.cv_block_1(y)
        y1 = self.cv_block_2(y)
        y2, y3 = y.chunk(2, 1)
        y = list((y0, y1, y2, y3))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv_final(torch.cat(y, 1))

class MANet_PD(MANet):
    def __init__(self, c1, c2, n=1, shortcut=False, p=1, kernel_size=3, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, p, kernel_size, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))



##############################################################################
# Star_Block
class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x
