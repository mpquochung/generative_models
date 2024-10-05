import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(resid)
    
class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x

        n, c, h, w = x.shape

        x = x.view(n,c,h*w)

        x = x.transpose(-1, -2)

        x = self.attention(x)

        x = x.transpose(-1, -2)

        x = x.view(n,c,h,w)

        x+= resid

        return x
