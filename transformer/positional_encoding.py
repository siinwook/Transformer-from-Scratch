import torch
from torch import nn

class PositionalEncoding(nn.Module):
  def __init__(self,max_len,d_model):
    super().__init__()

    pos = torch.arange(0,max_len).unsqueeze(1) # (max_len,) -> (max_len,1)
    div_term = 1.0 / 10000**(torch.arange(0,d_model,2) / d_model)

    pe = torch.zeros((max_len,d_model))
    pe[:,0::2] = torch.sin(pos * div_term) # (max_len,d_model)
    pe[:,1::2] = torch.cos(pos * div_term)
    pe = pe.unsqueeze(0) # (1,max_len,d_model)

    self.register_buffer("pe", pe) # Non-trainable params, but included in model
    #self.pe = pe // not params, not buffer

  def forward(self,x):
    seq_len = x.size(1)

    return x + self.pe[:,:seq_len,:]
