# Dataset Pipeline

This module prepares the Multi30k English-German translation dataset for training the Transformer model.

## Pipeline

Raw Text  

↓  

Lowercase Normalization  

↓  

Joint BPE Tokenization  

↓  

Add Special Tokens (BOS, EOS)  
↓  

Tensor Conversion


## Token Format

### Encoder Input
```
<BOS> + English Tokens + <EOS>
```

### Decoder Input (Teaching Force)
```
<BOS> + German Tokens
```

### Decoder Target Label
```
German Tokens + <EOS>
```

## Tokenizer

A shared vocabulary is trained using Hugging Face `ByteLevelBPETokenizer` on both English and German sentences.

This creates a joint subword vocabulary for the Encoder and Decoder.
