import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__ (self, n_heads: int, d_embed: int, in_proj_bias = True, out_proj_bias = True) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask =  False) -> torch.Tensor:
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        
        q,k,v = self.in_proj(x).chunk(3, dim = -1)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2) 

        # masking
        if casual_mask:
            mask = torch.ones_like(weight, dtype = torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight = F.softmax(weight / math.sqrt(self.d_head), dim = -1)
        
        output = weight @ v

        output = output.transpose(1,2).reshape(input_shape)

        output = self.out_proj(output)

        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias = True, out_proj_bias = True) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Seq_Len_Q, Dim_Q)
        # context: (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Define q,k,v
        q = self.q_proj(x).view(interim_shape).transpose(1,2)
        k = self.k_proj(context).view(interim_shape).transpose(1,2)
        v = self.v_proj(context).view(interim_shape).transpose(1,2)

        # Attention Calculation
        weight = q @ k.transpose(-1,-2)
        weight = F.softmax(weight/math.sqrt(self.d_head), dim = -1)
        output = weight @ v
        output = output.transpose(1,2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output


