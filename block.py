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
        nn.init.constant_(self.w, m)

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
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='linear')
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0.0)


    def forward(self, x):
        x_pool = self.avg_pool(x)
        x1 = self.conv1(x_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x_pool).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1, x2), dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2), x1.transpose(-1, -2)), dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out = self.mix(out1, out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return x * out


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


#多尺度增强模块
class Enhance(nn.Module):
    def __init__(self, inc, bins=None):
        super().__init__()
        self.inc = inc
        self.bins = bins if (isinstance(bins, (list, tuple)) and len(bins) > 0) else [4, 6, 8]
        self.split_dim = max(self.inc // (len(self.bins) + 1), 8)
        
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                Conv(self.inc, self.split_dim, 1, g=1),
                Conv(self.split_dim, self.split_dim, 3, g=self.split_dim)
            ) for bin_size in self.bins
        ])
        self.local_conv = Conv(self.inc, self.inc // 2, 1, g=1)
        concat_dim = self.inc // 2 + len(self.bins) * self.split_dim
        self.final_conv = Conv(concat_dim, self.inc, 1, g=1)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for feat in self.features:
            scale_feat = feat(x)
            scale_feat_up = F.interpolate(scale_feat, x_size[2:], mode='bilinear', align_corners=False)
            out.append(scale_feat_up)
        return self.final_conv(torch.cat(out, dim=1))


#Enhance_AFCA
class Enhance_AFCA(nn.Module):
    def __init__(self, c1=None, c2=None, shortcut=True, n=1, g=1, e=0.375, bins=None, resume_compat=False):
        super().__init__()
        self.config_c2 = c2
        self.config_shortcut = shortcut
        self.config_n = n
        self.config_g = g
        self.config_e = e
        self.config_bins = bins
        self.config_resume_compat = resume_compat
        
        self.initialized = False
        self.cv1 = None
        self.m = None
        self.cv2 = None
        self.attention = None
        self.c1 = None
        self.c2 = None
        self.c = None

    def _initialize_layers(self, actual_c1, device):
        self.c1 = max(actual_c1, 8)
        if isinstance(self.config_c2, bool):
            self.c2 = self.c1
        elif self.config_c2 is not None and isinstance(self.config_c2, (int, float)) and self.config_c2 > 0:
            self.c2 = max(int(self.config_c2), 8)
        else:
            self.c2 = self.c1
        n = max(int(self.config_n), 1) if isinstance(self.config_n, (int, float)) else 1
        g = max(int(self.config_g), 1) if isinstance(self.config_g, (int, float)) else 1
        e = self.config_e if (isinstance(self.config_e, (int, float)) and 0 < self.config_e <= 1) else 0.375
        bins = self.config_bins if (isinstance(self.config_bins, (list, tuple)) and len(self.config_bins) > 0) else [4, 6, 8]
        resume_compat = self.config_resume_compat if isinstance(self.config_resume_compat, bool) else False
        self.c = max(int(self.c2 * e), 8)
        self.cv1 = Conv(self.c1, 2 * self.c, 1, g=g).to(device)
        self.m = nn.ModuleList([Enhance(self.c, bins) for _ in range(n)]).to(device)
        cv2_in_ch = 2 * self.c if resume_compat else (2 + n) * self.c
        self.cv2 = Conv(cv2_in_ch, self.c2, 1, g=g).to(device)
        self.attention = AFCA(self.c2).to(device) if self.c2 >= 32 else None
        
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._initialize_layers(x.shape[1], x.device)
        if x.shape[1] != self.c1:
            self._initialize_layers(x.shape[1], x.device)
        cv1_out = self.cv1(x)
        if cv1_out.shape[1] % 2 != 0:
            pad = torch.zeros_like(cv1_out[:, :1, :, :], device=x.device)
            cv1_out = torch.cat([cv1_out, pad], dim=1)
        y = list(cv1_out.chunk(2, dim=1))
        
        current = y[-1]
        for module in self.m:
            enhanced = module(current)
            if self.config_shortcut:
                enhanced = enhanced + current
            if not self.config_resume_compat:
                y.append(enhanced)
            current = enhanced
        
        concat_features = torch.cat(y, dim=1) if not self.config_resume_compat else torch.cat([y[0], current], dim=1)
        output = self.cv2(concat_features)
        
        if self.attention is not None:
            output = self.attention(output)
        
        return output

    def fuseforward(self, x):
        if not self.initialized:
            self._initialize_layers(x.shape[1], x.device)
        
        cv1_out = self.cv1.conv(x)
        cv1_out = self.cv1.act(cv1_out)
        if cv1_out.shape[1] % 2 != 0:
            pad = torch.zeros_like(cv1_out[:, :1, :, :], device=x.device)
            cv1_out = torch.cat([cv1_out, pad], dim=1)
        y = list(cv1_out.chunk(2, dim=1))

        current = y[-1]
        for module in self.m:
            scale_feat = current
            for submodule in module.features[0]:
                if hasattr(submodule, 'conv'):
                    scale_feat = submodule.conv(scale_feat)
                    if hasattr(submodule, 'act'):
                        scale_feat = submodule.act(scale_feat)
            scale_feat_up = F.interpolate(scale_feat, current.size()[2:], mode='bilinear', align_corners=False)
            
            local_feat = module.local_conv.conv(current)
            local_feat = module.local_conv.act(local_feat)
            
            concat_feat = torch.cat([local_feat, scale_feat_up], dim=1)
            enhanced = module.final_conv.conv(concat_feat)
            enhanced = module.final_conv.act(enhanced)
            
            if self.config_shortcut:
                enhanced = enhanced + current
            if not self.config_resume_compat:
                y.append(enhanced)
            current = enhanced

        concat_features = torch.cat(y, dim=1) if not self.config_resume_compat else torch.cat([y[0], current], dim=1)
        output = self.cv2.conv(concat_features)
        output = self.cv2.act(output)

        if self.attention is not None:
            output = self.attention(output)

        return output
######################################### Ehance_AFCA end ##############



############################Enhance########################################
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class CSP_Enhance(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))

class Enhance(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CSP_Enhance(self.c, [3, 6, 9, 12]) for _ in range(n))


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
