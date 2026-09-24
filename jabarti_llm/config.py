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
    pad_id: int = 0

    tie_weights: bool = True


    @property
    def d_k(self):
        """The width of one head's Query, Key and Value vectors."""
        return self.d_model // self.n_heads

    @classmethod
    def jabarti(cls):
        return cls(
            d_model=512, n_heads=8, n_layers=8
        )

    @classmethod
    def tiny(cls):
        return cls(
            d_model=128, n_heads=4, n_layers=2, max_seq_len=256
        )
    
@dataclass
class GenerationConfig:
    """How to turn logits into text.

    Sampling is where a model stops being deterministic. Every field here
    trades coherence against variety, and there is no universally right
    setting -- ch11 exists to give a feel for the trade.
    """

    max_new_tokens: int = 100
    temperature: float = 0.8

    top_k: int | None = 50     
    top_p: float | None = 0.95   
                                  
    use_cache: bool = True      
    seed: int | None = None      

    @classmethod
    def greedy(cls):
        """Always take the likeliest token. Deterministic, and repetitive."""
        return cls(temperature=0.0, top_k=None, top_p=None)