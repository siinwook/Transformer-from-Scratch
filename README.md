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

- Transformer is based on solely attention mechanism unlike recurrent or convolutional models.
- Embedded tokens are added with positional encoding which represents the position of tokens in sequence.
- Multi head attention allows the model to learn different representations simultaneously.

## Tech Stack

PyTorch, Hugging Face, Cuda, Matplotlib, Jupyter Notebook
