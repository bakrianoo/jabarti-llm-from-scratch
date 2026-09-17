"""
config.py -- the single source of truth for every model dimension
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 32_000    
    d_model: int = 768          # Embedding Dim width
    max_seq_len: int = 1024
    dropout: float = 0.1

    n_heads: int = 12
    qkv_bias: bool = False


    @property
    def d_k(self):
        """The width of one head's Query, Key and Value vectors."""
        return self.d_model // self.n_heads

    

    