# Transformer Architecture

## Data Flow

Input Tokens → Embedding → Positional Encoding → Encoder * N → E

Output Tokens → Embedding → Positional Encoding → Decoder * N (with E) → Linear Projection → Softmax

## Components

### 1. Positional Encoding

- PE(pos,2i) = sin(pos/10000^(2i/dmodel))
- PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))

Let θ = pos/10000^(2i/dmodel), it is acceptable to understand as (sin(θ), cos(θ)) pair of different frequency clocks.

![positional_encoding_intuition](./positional_encoding_intuition.png)

In this view, high frequency clock (low i) represents local position and low frequency clock (high i) represents global position in sequence.


### 2. Multi-Head Attention

Q,K,V projection...

Shape:
(batch, seq_len, d_model)
→
(batch, heads, seq_len, d_k)

...
