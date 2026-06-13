# Transformer Architecture

## Data Flow

Input Tokens → Embedding → Positional Encoding → Encoder * N → E

Output Tokens → Embedding → Positional Encoding → Decoder * N (with E) → Linear Projection → Softmax

## Components

### Multi-Head Attention

Q,K,V projection...

Shape:
(batch, seq_len, d_model)
→
(batch, heads, seq_len, d_k)

...
