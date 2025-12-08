import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from einops import rearrange
from ..modules.conv import Conv, autopad
from ..backbone.TransNext import  get_relative_position_cpb
from timm.models.layers import trunc_normal_

__all__ = [ 'SimAM', 'MLCA', 'CAA', 'AFCA']



class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)
    
    
    
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_weight=local_weight
        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)
        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)
        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x



# CVPR2024 PKINet
class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x
    
class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
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
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels,channels,kernel_size=5,padding=2,groups=channels)
        self.dconv1_7 = nn.Conv2d(channels,channels,kernel_size=(1,7),padding=(0,3),groups=channels)
        self.dconv7_1 = nn.Conv2d(channels,channels,kernel_size=(7,1),padding=(3,0),groups=channels)
        self.dconv1_11 = nn.Conv2d(channels,channels,kernel_size=(1,11),padding=(0,5),groups=channels)
        self.dconv11_1 = nn.Conv2d(channels,channels,kernel_size=(11,1),padding=(5,0),groups=channels)
        self.dconv1_21 = nn.Conv2d(channels,channels,kernel_size=(1,21),padding=(0,10),groups=channels)
        self.dconv21_1 = nn.Conv2d(channels,channels,kernel_size=(21,1),padding=(10,0),groups=channels)
        self.conv = nn.Conv2d(channels,channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class CAFM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=(1,1,1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=(3,3,3), stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1,1,1), bias=bias)
        self.fc = nn.Conv3d(3*self.num_heads, 9, kernel_size=(1,1,1), bias=True)

        self.dep_conv = nn.Conv3d(9*dim//self.num_heads, dim, kernel_size=(3,3,3), bias=True, groups=dim//self.num_heads, padding=1)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0,2,3,1) 
        f_all = qkv.reshape(f_conv.shape[0], h*w, 3*self.num_heads, -1).permute(0, 2, 1, 3) 
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)
        #local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9*x.shape[1]//self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv) # B, C, H, W
        out_conv = out_conv.squeeze(2)
        # global SA
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output

    

