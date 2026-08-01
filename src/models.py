import torch

import torch.nn as nn
from .layers import RConv2d, ConvBlock, ConvNormBlock
from .utils import setKaiming_, setXavier_

class Critic(nn.Module):
    def __init__(self, depth=5, init_ch=16, max_ch=512):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(6,init_ch,5,1,2), # 256->256
        ])
        for i in range(depth):
            chin = min([init_ch*2**i,max_ch])
            chout = min([init_ch*2**(i+1),max_ch])
            self.conv_blocks.append(ConvBlock(chin,chout,4,2,1))
        self.toMap = (setXavier_(nn.Conv2d(chout,1,3,1,1,padding_mode='reflect')))
    def forward(self, x:torch.Tensor):
        for layer in self.conv_blocks:
            x = layer(x)
        return self.toMap(x)

class Generator(nn.Module):
    def __init__(self, lenght=5, init_ch=16):
        super().__init__()
        self.encoder = nn.Sequential(
            setKaiming_(nn.Conv2d(3,init_ch,4,2,1,padding_mode='reflect')),
            nn.InstanceNorm2d(init_ch),
            nn.ReLU(),
            setKaiming_(nn.Conv2d(init_ch,init_ch*2,4,2,1,padding_mode='reflect')),
            nn.InstanceNorm2d(init_ch*2),
            nn.ReLU(),
            setKaiming_(nn.Conv2d(init_ch*2,init_ch*4,4,2,1,padding_mode='reflect')),
            nn.InstanceNorm2d(init_ch*4),
            nn.ReLU(),
            setKaiming_(nn.Conv2d(init_ch*4,init_ch*8,4,2,1,padding_mode='reflect')),
            nn.InstanceNorm2d(init_ch*8),
            nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            *[RConv2d(init_ch*8) for _ in range(lenght)]
        )
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvNormBlock(8*init_ch,4*init_ch,3,1,1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvNormBlock(4*init_ch,2*init_ch,3,1,1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvNormBlock(2*init_ch,init_ch,3,1,1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvNormBlock(init_ch,init_ch//2,3,1,1),
        )
        self.toRGB = setXavier_(nn.Conv2d(init_ch//2,3,1))
    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.toRGB(x)
        x = torch.tanh(x)
        return x