"""
block.py -- one Transformer block
"""

import torch.nn as nn

from jabarti_llm.model.attention import MultiHeadAttention
from jabarti_llm.model.feedforward import FeedForward

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)



    def forward(self, x, kv_cache=None, use_cache=False):

        attention_out, updated_cache = self.attn(
            self.ln1(x), kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + self.dropout(attention_out)

        ff_out = self.ffn( self.ln2(x) )
        x = x + self.dropout(ff_out)

        return x, updated_cache

