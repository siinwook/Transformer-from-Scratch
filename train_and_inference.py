import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # HF
from tokenizers import ByteLevelBPETokenizer # HF
import math

from dataset import multi30k_en2de
from transformer.positional_encoding import PositionalEncoding
from transformer.multihead_attention import MultiHeadAttention
from transformer.feedforward_network import FeedForward
from transformer.encoder import EncoderBlock
from transformer.encoder import Encoder
from transformer.decoder import DecoderBlock
from transformer.decoder import Decoder
from transformer.transformer import Transformer

# =========================
# Constants
# =========================
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
VOCAB_SIZE = 5000

# =========================
# Cuda
# =========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Get HF dataset
# =========================
multi30k = load_dataset("bentrevett/multi30k")
multi30k_train = multi30k['train']

# =========================
# Get HF tokenizer
# =========================
tokenizer = ByteLevelBPETokenizer()

# =========================
# Padding
# =========================
def collate_fn(batch):
  src = [item['src'] for item in batch]
  tgt = [item['tgt'] for item in batch]
  label = [item['label'] for item in batch]

  padded_src = pad_sequence(src, batch_first = True, padding_value = PAD_IDX)
  padded_tgt = pad_sequence(tgt, batch_first = True, padding_value = PAD_IDX)
  padded_label = pad_sequence(label, batch_first = True, padding_value = PAD_IDX)

  return {'src':padded_src, 'tgt':padded_tgt, 'label':padded_label}

# =========================
# Get dataset and dataloader
# =========================
dataset = multi30k_en2de(data = multi30k_train, num_samples = len(multi30k_train), tokenizer = tokenizer, vocab_size = VOCAB_SIZE)
dataloader = DataLoader(dataset, batch_size = 64, shuffle = True,collate_fn = collate_fn)

# =========================
# Model
# =========================
model = Transformer(src_vocab_size = VOCAB_SIZE, tgt_vocab_size= VOCAB_SIZE, max_len = 75, d_model = 192, num_heads = 8, d_ff = 784, drop_out = 0.1, N = 6).to(device)

# =========================
# Training conditions
# =========================
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
epochs=[25,25]
lrs=[1e-4,5e-5]

# =========================
# Train
# =========================
train_loss = []
for i in range(len(epochs)):
  optimizer = torch.optim.Adam(model.parameters(), lr = lrs[i])
  for epoch in range(epochs[i]):
    model.train()
    epoch_loss = 0

    for batch, sentences in enumerate(dataloader):
      src = sentences['src']
      tgt_input = sentences['tgt']
      tgt_label = sentences['label']

      src = src.to(device)
      tgt_input = tgt_input.to(device)
      tgt_label = tgt_label.to(device)

      logits = model(src, tgt_input)

      logits = logits.reshape(-1, logits.size(-1))
      tgt_label = tgt_label.reshape(-1)

      optimizer.zero_grad()
      loss = loss_fn(logits,tgt_label)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()

    print(f"epoch {sum(epochs[:i])+epoch+1} loss: {epoch_loss / len(dataloader)}")
    train_loss.append(epoch_loss / len(dataloader))

# =========================
# Inference
# =========================
tokenizer = dataset.tokenizer

test_src = "A trendy girl talking on her cellphone while gliding slowly down the street." # multi30k['test']

src_token = tokenizer.encode(test_src).ids
src_token = [BOS_IDX] + src_token + [EOS_IDX]

model.eval()
with torch.no_grad():
  src_token = torch.tensor([src_token]).to(device)
  src_embed = model.src_Embedding(src_token) * math.sqrt(model.d_model)
  src_embed = model.Pos_Encoding(src_embed)

  E = model.Encoder(src_embed)

  test_tgt = torch.tensor([[BOS_IDX]]).to(device)

  for i in range(20):
    tgt_embed = model.tgt_Embedding(test_tgt) * math.sqrt(model.d_model)
    tgt_embed = model.Pos_Encoding(tgt_embed)

    out = model.Decoder(tgt_embed, E)
    out = model.FC(out)

    out = out[:,-1,:].argmax(dim=-1).item()

    test_tgt= torch.cat((test_tgt,torch.tensor([[out]]).to(device)),dim=-1)

    if out == EOS_IDX:
      break

test_tgt = test_tgt.cpu()
out = tokenizer.decode(test_tgt[0].tolist())

print(out)
