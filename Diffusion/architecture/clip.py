import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, context_length: int) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, embed_dim))

    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        return x
    

class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, d_embed: int) -> None:
        super().__init__()

        self.attention = SelfAttention(n_heads, d_embed)
        self.layernorm1 = nn.LayerNorm(d_embed)
        self.layernorm2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, d_embed*4)
        self.linear_2 = nn.Linear(d_embed*4, d_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        residue = x

        x = self.layernorm1(x)
        x = self.attention(x, casual_mask = True)
        x += residue

        residue = x

        x = self.layernorm2(x)
        x = self.linear_1(x)
        x = x*torch.sigmoid(1.702 * x) #GeLU
        x= self.linear_2(x)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList(
            [
                CLIPLayer(12, 768) for i in range(12)
            ]
        )

        self.layernorm = nn.LayerNorm(768) 

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        return output