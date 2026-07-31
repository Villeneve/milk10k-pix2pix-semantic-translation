import torch

import torch.nn as nn
from .layers import RConv2d, ConvBlock, ConvNormBlock
from .utils import setKaiming_, setXavier_

class Critic(nn.Module):
    def __init__(self, depth=5, init_ch=32, max_ch=512):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            ConvBlock(6,32,5,1,2), # 256->256
        ])
        for i in range(depth):
            chin = min([init_ch*2**i,max_ch])
            chout = min([init_ch*2**(i+1),max_ch])
            self.conv_blocks.append(ConvBlock(chin,chout,4,2,1))
        self.toMap = torch.nn.utils.spectral_norm(setXavier_(nn.Conv2d(chout,1,3,1,1,padding_mode='reflect')))
    def forward(self, x:torch.Tensor):
        for layer in self.conv_blocks:
            x = layer(x)
        return self.toMap(x)

class Generator(nn.Module):
    def __init__(self, lenght=5):
        super().__init__()
        self.encoder = nn.Sequential(
            # 256 -> 128 RF=4 jmp=2
            setKaiming_(nn.Conv2d(3,32,4,2,1,padding_mode='reflect')),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 128 -> 64 RF=4+(4-1)*2=10 jmp=4
            setKaiming_(nn.Conv2d(32,64,4,2,1,padding_mode='reflect')),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64 -> 32 RF=10+(4-1)*4=22 jmp=8
            # setKaiming_(nn.Conv2d(64,128,4,2,1,padding_mode='reflect')),
            # nn.InstanceNorm2d(128),
            # nn.ReLU(),
            # # 32 -> 16 RF=22+(4-1)*8=46 jmp=16
            # setKaiming_(nn.Conv2d(128,256,4,2,1,padding_mode='reflect')),
            # nn.InstanceNorm2d(256),
            # nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            # 32 -> 32 RF=22+(3-1)*8=38 jmp=8
            *[RConv2d(64) for _ in range(lenght)]
        )
        self.decoder = nn.Sequential(
            # nn.UpsamplingNearest2d(scale_factor=2),
            # ConvNormBlock(256,128,3,1,1),
            # nn.UpsamplingNearest2d(scale_factor=2),
            # ConvNormBlock(128,64,3,1,1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvNormBlock(64,32,3,1,1),
            nn.UpsamplingNearest2d(scale_factor=2),
            ConvNormBlock(32,16,3,1,1),
        )
        self.toRGB = setXavier_(nn.Conv2d(16,3,1))
    def forward(self, x:torch.Tensor):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.toRGB(x)
        x = torch.tanh(x)
        return x