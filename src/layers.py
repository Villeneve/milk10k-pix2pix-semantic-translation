import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .utils import setKaiming_, setXavier_

class RConv2d(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.residual = nn.Sequential(
            setKaiming_(nn.Conv2d(channels,channels,3,1,1,padding_mode='reflect')),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            setXavier_(nn.Conv2d(channels,channels,3,1,1,padding_mode='reflect')),
            nn.InstanceNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1,channels,1,1))
    def forward(self, x:torch.Tensor):
        return x + self.gamma*self.residual(x)

class ConvBlock(nn.Module):
    def __init__(
            self,
            inCh:int,
            outCh:int,
            k_size=3,
            stride=1,
            padding=0,
            padding_mode='reflect',
            **kwargs
    ):
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(setKaiming_(nn.Conv2d(inCh,outCh,k_size,stride=stride,padding=padding,padding_mode=padding_mode,**kwargs),.2)),
            nn.LeakyReLU(.2),
        )
    def forward(self, x:torch.Tensor):
        return self.block(x)

class ConvNormBlock(nn.Module):
    def __init__(
            self,
            inCh:int,
            outCh:int,
            k_size=3,
            stride=1,
            padding=0,
            padding_mode='reflect',
            **kwargs
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inCh,outCh,k_size,stride=stride,padding=padding,padding_mode=padding_mode,**kwargs),
            nn.InstanceNorm2d(outCh),
            nn.ReLU(),
        )
    def forward(self, x:torch.Tensor):
        return self.block(x)