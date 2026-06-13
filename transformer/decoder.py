import torch
from torch import nn

from transformer.multihead_attention import MultiHeadAttention
from transformer.feedforward_network import FeedForward

class DecoderBlock(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, drop_out=0.1):
    super().__init__()

    self.num_heads = num_heads

    self.Masked_MultiHeadAttn = MultiHeadAttention(d_model, num_heads)
    self.MultiHeadAttn = MultiHeadAttention(d_model, num_heads)
    self.FFN = FeedForward(d_model, d_ff)

    self.LayerNorm1 = nn.LayerNorm(d_model)
    self.LayerNorm2 = nn.LayerNorm(d_model)
    self.LayerNorm3 = nn.LayerNorm(d_model)

    self.Dropout = nn.Dropout(drop_out)

  def forward(self, x, E):
    device = x.device

    batch_size = x.size(0)
    x_seq_len = x.size(1)

    mask = torch.ones(batch_size,self.num_heads,x_seq_len,x_seq_len).to(device)
    mask = torch.tril(mask)

    mask_attn_out = self.Masked_MultiHeadAttn(x,x,x,mask)
    x = self.LayerNorm1(x + self.Dropout(mask_attn_out))

    attn_out = self.MultiHeadAttn(x,E,E,mask=None)
    x = self.LayerNorm2(x + self.Dropout(attn_out))

    ffn_out = self.FFN(x)
    x = self.LayerNorm3(x + self.Dropout(ffn_out))

    return x

class Decoder(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, drop_out=0.1, N=6):
      super().__init__()

      self.N = N
      self.DecBlcockList = nn.ModuleList([
          DecoderBlock(d_model, num_heads, d_ff, drop_out)
          for i in range(N)
      ])

  def forward(self, x, E):
    for DecBlock in self.DecBlcockList:
      x = DecBlock(x, E)

    return x
