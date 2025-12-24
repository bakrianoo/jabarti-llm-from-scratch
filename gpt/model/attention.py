"""
Multi-Head Attention mechanism for GPT model.

Implements scaled dot-product attention with causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism with causal masking.
    
    Each position can only attend to positions at or before it (autoregressive).
    Multiple attention heads allow the model to learn different types of
    relationships (syntax, semantics, morphology, etc.).
    
    Args:
        d_model: Model dimension (512)
        n_heads: Number of attention heads (8)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads # Dimension per head (64 for 512/8)

        # Dropout for attention weights
        self.attention_dropout = nn.Dropout(dropout)

        # Dropout for output
        self.output_dropout = nn.Dropout(dropout)

        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)

        # Output projection after concatenating heads
        self.output_projection = nn.Linear(d_model, d_model)

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.d_head)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask to prevent attending to future positions.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Lower triangular mask of shape [seq_len, seq_len]
            Values are 1 for allowed positions, 0 for masked positions
        """
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device)
        )

        return mask

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None
                ):

        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            padding_mask: Optional padding mask [batch_size, seq_len]
        
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, d_model = x.shape

        # 1. Project input to Q, K, V: [batch_size, seq_len, d_model]
        Q = self.query_projection(x)
        K = self.key_projection(x)
        V = self.value_projection(x)

        # [batch_size, seq_len, d_model] -> [4, 128, 512]
        # 2. Reshape Q, K, V for multi-head attention
        # [4, 128, 512] -> .view -> [4, 128, 8, 64]
        # [4, 128, 8, 64] -> transpose(1, 2) -> [4, 8, 128, 64] [batch_size, n_heads, seq_len, d_head]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # 3. Compute attention scores
        # Q @ K^T:
        # [batch_size, n_heads, seq_len, d_head] @ [batch_size, n_heads, d_head, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        # attention_scores.shape = [batch_size, n_heads, seq_len, seq_len]

        # 4. Scale attention scores by √d_head to prevent softmax saturation
        attention_scores = attention_scores / self.scale

        # 5. Generate causal mask (prevent attending to future positions)
        # causal_mask.shape = [seq_len, seq_len]
        causal_mask = self._create_causal_mask(seq_len, x.device)

        # Expand causal mask for batch and heads: [seq_len, seq_len] → [1, 1, seq_len, seq_len]
        # [seq_len, seq_len] 
        #                   -> unsqueeze(0) -> [1, seq_len, seq_len] 
        #                   -> unsqueeze(0) -> [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply mask: set masked positions to -inf before softmax
        attention_scores = attention_scores.masked_fill(
            causal_mask == 0, float('-inf')
        )

        # 6. Apply padding mask if provided
        if padding_mask is not None:
            # attention score: [batch_size, n_heads, seq_len, seq_len]
            # padding_mask   : [batch_size, 1,       1,       seq_len]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(
                padding_mask == 0, float('-inf')
            )

        # 7. Apply softmax to get attention weights
        # -> [batch_size, n_heads, seq_len, seq_len]
        attention_weights = F.softmax(
            attention_scores, dim=-1
        )

        # Handle case where entire row is -inf (results in nan after softmax)
        attention_weights = torch.nan_to_num(
            attention_weights, nan=0.0
        )

        # 8. Apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)

        # 9. Apply attention weights to values
        # [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_head]
        attended_values = torch.matmul(attention_weights, V)

        # 10. Concatenate heads
        # we have attended_values: [batch_size, n_heads, seq_len, d_head] -> [batch_size, seq_len, d_model]
        #                          # Example: [32, 8, 10, 64] -> [32, 10, 512]

        # Before: [batch, n_heads, seq_len, d_head]
        #         [32,    8,       10,      64]
        attended_values = attended_values.transpose(1, 2).contiguous()
        # After:  [batch, seq_len, n_heads, d_head]
        #         [32,    10,      8,       64]


        # Before: [batch, seq_len, n_heads, d_head]
        #         [32,    10,      8,       64]
        attended_values = attended_values.view(batch_size, seq_len, self.d_model)
        # After:  [batch, seq_len, d_model]
        #         [32,    10,      512]

        # 11. Apply output projection
        output = self.output_projection(attended_values)
        output = self.output_dropout(output)

        return output
