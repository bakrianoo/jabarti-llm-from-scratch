"""
jabarti_llm -- a bilingual (Arabic + English) GPT, built from scratch.
"""

from jabarti_llm.config import GenerationConfig, ModelConfig, TrainingConfig
from jabarti_llm.data import (
    ChatDataset,
    PackedDataset,
    chat_collate_fn,
    collate_fn,
    format_chat,
)
from jabarti_llm.generation import generate, generate_ids
from jabarti_llm.model import (
    GPT,
    FeedForward,
    InputEmbedding,
    MultiHeadAttention,
    TransformerBlock,
)
from jabarti_llm.tokenizer import Tokenizer
from jabarti_llm.training import (
    Tracker,
    Trainer,
    freeze_all,
    load_checkpoint,
    model_config_from_checkpoint,
    save_checkpoint,
    unfreeze_tail,
)

__version__ = "0.1.0"

__all__ = [
    "GPT",
    "ChatDataset",
    "FeedForward",
    "GenerationConfig",
    "InputEmbedding",
    "ModelConfig",
    "MultiHeadAttention",
    "PackedDataset",
    "Tokenizer",
    "Tracker",
    "Trainer",
    "TrainingConfig",
    "TransformerBlock",
    "chat_collate_fn",
    "collate_fn",
    "format_chat",
    "freeze_all",
    "generate",
    "generate_ids",
    "load_checkpoint",
    "model_config_from_checkpoint",
    "save_checkpoint",
    "unfreeze_tail",
]

