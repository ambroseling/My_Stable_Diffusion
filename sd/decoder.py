import torch
from torch import nn
from torch.nn import functional as F
from block import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Decoder(nn.Sequential):
    def __init__ (self):
        super().__init__(
        nn.Conv2d(4,4,kernel_size=1,padding=0),
        nn.Conv2d(4,512,kernel_size=3,padding=1),
        VAE_ResidualBlock(512,512),
        VAE_AttentionBlock(512),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        VAE_ResidualBlock(512,512),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(512,512,kernel_size=3,padding=1),
        VAE_ResidualBlock(512,256),
        VAE_ResidualBlock(256,256),
        VAE_ResidualBlock(256,256),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(256,256,kernel_size=3,padding=1),
        VAE_ResidualBlock(256,128),
        VAE_ResidualBlock(128,128),
        VAE_ResidualBlock(128,128),
        nn.GroupNorm(32,128),
        nn.SiLU(),
        nn.Conv2d(128,3,kernel_size=3,padding=1)

        )
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # x: (B,4,H/8,W/8)
        x /= 0.18215
        for module in self:
            x = module(x)
        return x
    

if __name__ == "__main__":
    decoder = VAE_Decoder()
    print(decoder.state_dict()['1.weight'].shape)