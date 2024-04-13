import torch
from torch import nn
from torch.nn import functional as F
from block import VAE_AttentionBlock, VAE_ResidualBlock


'''
'''
class VAE_Encoder(nn.Sequential):
    def __init__ (self):
        super().__init__(
        # (B,3,H,W) -> (B,128,H,W)
        nn.Conv2d(3,128,kernel_size = 3,padding=0),
        # (B,128,H,W) -> (B,128,H,W)
        VAE_ResidualBlock(128,128),
        # (B,128,H,W) -> (B,128,H,W)
        VAE_ResidualBlock(128,128),
        # (B,128,H,W) -> (B,128,H,W)
        nn.Conv2d(128,128,kernel_size = 3,stride=2,padding=0),
        # (B,128,H,W) -> (B,128,H,W)
        VAE_ResidualBlock(128,256),
        # (B,128,H,W) -> (B,128,H,W)
        VAE_ResidualBlock(256,256),
        nn.Conv2d(256,256,kernel_size = 3,stride=2,padding=0),
        VAE_ResidualBlock(256,512),
        VAE_ResidualBlock(512,512),
        nn.Conv2d(512,512,kernel_size = 3,stride=2,padding=0),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        VAE_AttentionBlock(512),
        VAE_ResidualBlock(512,512),
        nn.GroupNorm(32,512),
        nn.SiLU(),
        nn.Conv2d(512,8,kernel_size = 3,padding=1),
        nn.Conv2d(8,8,kernel_size = 1,padding=0),
        )
    def forward(self,x: torch.Tensor, noise:torch.Tensor)-> torch.Tensor:
        # x: (B,Cin,H,W)
        # noise: (B,Cout,H/8,H/8)
        for module in self:
            if getattr(module,'stride',None) == (2,2):
                x = F.pad(x,(0,1,0,1))
            x = module(x)
        mean,log_variance = torch.chunk(x,2,dim=1)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        # Z ~ N(0,I) -> N(mean,variance)=x?
        # X = mean + stdev * Z
        x = mean + stdev*noise
        x *= 0.18215 # some scaling factor
        return x


