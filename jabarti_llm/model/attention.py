"""
attention.py -- multi-head causal self-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    """Every token looks at every earlier token, through several lenses at once.

    One head learns one way of relating tokens. Two independently initialised
    heads produce completely different attention patterns on the same sentence
    (proven in ch06), so a single head is a compromise rather than a
    simplification. n_heads of them run in parallel, each free to specialise.

    Input:  (B, seq_length, d_model)
    Output: (B, seq_length, d_model)
    """

    def __init__(self, config):
        super().__init__()

        self.d_k = config.d_k
        self.n_heads = config.n_heads

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)

        self.W_o = nn.Linear(config.d_model, config.d_model)

        mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )

        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, kv_cache=None, use_cache=False):

        """kv_cache is the (key, value) pair from every previous step.

        During training it is always None: the whole sequence arrives at once.
        During generation (ch11) it holds everything computed so far, so this
        step only has to project the one new token and append to it.
        """
    
        batch_size, seq_len, d_model = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            past_key, past_value = kv_cache
            past_len = past_key.size(2)

            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        updated_cache = (k, v) if use_cache else None
        total_len = past_len + seq_len

        scores = q @ k.transpose(-2, -1)
        scores = scores / self.d_k**0.5

        causal = self.mask[past_len:total_len, :total_len]
        scores = scores.masked_fill(~causal, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(out), updated_cache
    