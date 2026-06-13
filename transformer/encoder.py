import torch
from torch import nn

from transformer.multihead_attention import MultiHeadAttention
from transformer.feedforward_network import FeedForward

class EncoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, drop_out=0.1):
    super().__init__()

    self.MultiHeadAttn = MultiHeadAttention(d_model, num_heads)
    self.FFN = FeedForward(d_model, d_ff)

    self.LayerNorm1 = nn.LayerNorm(d_model)
    self.LayerNorm2 = nn.LayerNorm(d_model)

    self.Dropout = nn.Dropout(drop_out)

  def forward(self,x):
    attn_out = self.MultiHeadAttn(x,x,x,mask=None)
    x = self.LayerNorm1(x + self.Dropout(attn_out))

    ffn_out = self.FFN(x)
    x = self.LayerNorm2(x + self.Dropout(ffn_out))

    return x

class Encoder(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, drop_out=0.1, N=6):
      super().__init__()

      self.N = N
      self.EncBlcockList = nn.ModuleList([
          EncoderBlock(d_model, num_heads, d_ff, drop_out)
          for i in range(N)
      ])

  def forward(self,x):
    for EncBlock in self.EncBlcockList:
      x = EncBlock(x)

    return x
