"""
Embedding layers for GPT model.
Combines token embeddings with learned positional embeddings.
"""

import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):

    """
    Combined token and positional embeddings for GPT model.
    
    Converts token IDs to dense vectors and adds learned positional information.
    Uses learned positional embeddings (not sinusoidal) which is better for
    Arabic's flexible word order.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pad_token_id: ID of padding token (for proper masking)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.dropout = nn.Dropout(dropout)

        # Token embeddings: maps token IDs to dense vectors
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.d_model,
            padding_idx=self.pad_token_id,
        )

        # Positional embeddings: learned position encodings
        self.positional_embedding = nn.Embedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=d_model,
        )

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """
        Initialize embedding weights.
        
        Uses Normal(0, 0.02) initialization following GPT-2 approach.
        """
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0, std=0.02,
        )

        nn.init.normal_(
            self.positional_embedding.weight,
            mean=0.0, std=0.02,
        )

        # Ensure padding embedding is zero
        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].fill_(0.0)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        """
        Forward pass: convert token IDs to embeddings with positional info.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
        
        Returns:
            Embeddings of shape [batch_size, seq_len, d_model]
        """

        batch_size, seq_len = input_ids.shape

        # Validate sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_seq_len}"
            )
        
        # Get token embeddings: [batch_size, seq_len, d_model]
        token_embeds = self.token_embedding(input_ids)

        # Create position indices: [seq_len]
        positions = torch.arange(seq_len, device=input_ids.device)

        # Get positional embeddings: [seq_len, d_model]
        pos_embeds = self.positional_embedding(positions)

        # Add token and positional embeddings
        embeddings = token_embeds + pos_embeds

        # Apply dropout for regularization
        embeddings = self.dropout(embeddings)

        return embeddings
    