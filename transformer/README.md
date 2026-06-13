# Transformer Architecture

## Data Flow

Input Tokens → Embedding → Positional Encoding → Encoder * N → Encoder Output (Memory)

Output Tokens → Embedding → Positional Encoding → Decoder * N (with E) → Linear Projection → Softmax

## Components

### 1. Positional Encoding

- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Let θ = pos/10000^(2i/dmodel), it is acceptable to understand as (sin(θ), cos(θ)) pair of different frequency clocks.

![positional_encoding_intuition](./positional_encoding_intuition.png)

In this intuition, high frequency clock (low i) represents local position and low frequency clock (high i) represents global position in sequence.


### 2. Multi-Head Attention

- Attention(Q,K,V) = softmax(QKT / √dk)V
- MultiHead(Q,K,V) = Concat(head1, ..., headh) where headi = Attention(Q,K,V)

Tensor Shape:
| Operation | Before | After |
|---|---|---|
| Input | (batch, seq_len, d_model) | (batch, seq_len, d_model) |
| Q projection | (batch, seq_len, d_model) | (batch, num_heads, seq_len, d_k) |
| K projection | (batch, seq_len, d_model) | (batch, num_heads, seq_len, d_k) |
| V projection | (batch, seq_len, d_model) | (batch, num_heads, seq_len, d_v) |
| Q @ K.T | (batch, num_heads, seq_len, d_k) | (batch, num_heads, seq_len, seq_len) |
| softmax(QKT / √dk)V | (batch, num_heads, seq_len, seq_len) | (batch, num_heads, seq_len, d_v) |
| Concatenate | (batch, num_heads, seq_len, d_v) | (batch, seq_len, d_model) |

Multi-head attention allows the model to capture different representation subspaces.

### 3. Encoder

- Encoder consists of multi-head attention of Q, K, V derived from same inputs and feed forward network.
- Each sublayers are added with input(residual connection) and layer-normalized, namely LayerNorm(x + Sublayer(x))
- Repeats N times.

### 4. Decoder

- Decoder has same structure except additional masked multi-head attention in first sublayer, which prevents model from predicting with unknown tokens.
- In second attention layer, Q comes from output of first attention, wheras K and V come from output of encoder (memory).
- Each sublayers are added with input(residual connection) and layer-normalized, namely LayerNorm(x + Sublayer(x))
- Repeats N times.

### 5. Linear Projection and Softmax

- Returns embedded tokens to original vocabulary.
- Apply weight tying to reduce memory complexity.
