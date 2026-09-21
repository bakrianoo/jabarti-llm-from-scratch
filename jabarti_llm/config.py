"""
config.py -- the single source of truth for every model dimension
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32_000    
    d_model: int = 768          # Embedding Dim width
    d_ff: int = 3072            # defaults to 4 * d_model, GPT-2's ratio
    max_seq_len: int = 1024
    dropout: float = 0.1
    n_layers: int = 12          # how many Transformer blocks to stack

    n_heads: int = 12
    qkv_bias: bool = False


    @property
    def d_k(self):
        """The width of one head's Query, Key and Value vectors."""
        return self.d_model // self.n_heads

