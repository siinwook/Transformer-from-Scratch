# Transformer-from-Scratch

This project reproduces the structure of Transformer, *"Attention is all you need"*, and operates simple train & inference.

## What I implemented

- Positional Encoding
- Multi-Head Attention
- Encoder & Decoder
- Transformer
- Multi30k training pipeline joint Byte-Pair Encoding
- English to German Neural Machine Translation inference
- Train loss curve

## Key Idea

- Transformer is based on solely attention mechanism, unlike recurrent or convolutional models.
- Embedded tokens are added with positional encoding which represents the position of tokens in sequence.
- Multi head attention allows the model to learn different representations simultaneously.

## Experiments

| Experiment | Description |
|---|---|
| Memory complexity | Model summary comparison |
| ResNet44 vs PlainNet44 | Accuracy and training difficulty |
| ResNet20/56 vs PlainNet20/56 | *The degradation problem* |

## How to Run

```bash
git clone https://github.com/siinwook/ResNet-CIFAR10-from-Scratch.git
cd ResNet-CIFAR10-from-Scratch

pip install -r requirements.txt
python -m src.train
```
## Tech Stack

PyTorch, Hugging Face, Cuda, Matplotlib, Jupyter Notebook
