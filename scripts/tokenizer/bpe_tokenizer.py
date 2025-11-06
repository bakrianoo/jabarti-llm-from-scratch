from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

import re


def normalize_arabic_text(text, verbose=False):
    """
    Normalize Arabic text for tokenization.
    
    Educational Note:
    ----------------
    Text normalization is crucial for reducing vocabulary size and
    improving model generalization. For Arabic, this means handling:
    - Diacritics that create unnecessary token variants
    - Letter forms that represent the same phoneme
    - Inconsistent whitespace and punctuation
    
    Args:
        text (str): Raw Arabic text
        verbose (bool): Print normalization statistics
    
    Returns:
        str: Normalized text
    """
    
    if verbose:
        print("Normalizing text...")
        original_len = len(text)
    
    # Step 1: Remove diacritics (tashkeel)
    # Why? 'أَنْ' and 'أن' are the same word but appear as different tokens
    text = re.sub(r'[\u064B-\u0652\u0640]', '', text)
    
    # Step 2: Normalize alif maqsura
    # Why? ى and ي are often interchangeable in writing
    text = text.replace('ى', 'ي')
    
    # Step 4: Normalize whitespace
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Multiple newlines → double newline (paragraph break)
    
    # Step 6: Clean special Unicode
    special_chars = {'‌': '', '—': '-', '…': '...'}
    for old, new in special_chars.items():
        text = text.replace(old, new)
    
    text = text.strip()
    
    if verbose:
        print(f"  Reduced from {original_len:,} to {len(text):,} characters")
        print(f"  Reduction: {(original_len - len(text)) / original_len * 100:.1f}%")
    
    return text

def create_bpe_tokenizer():
    """
    Create a BPE tokenizer with appropriate configuration for Arabic.

    Returns:
        Tokenizer: Initialized BPE tokenizer
    """

    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    # Pre-tokenizer: Use Metaspace to handle whitespace properly
    tokenizer.pre_tokenizer = Metaspace()

    # Decoder: Use MetaspaceDecoder to convert ▁ back to spaces
    tokenizer.decoder = MetaspaceDecoder()

    return tokenizer


def train_bpe_tokenizer(
        tokenizer,
        text_iterator,
        vocab_size=32_000,
        min_frequency: int=3,
        special_tokens=None,
):

    # Create trainer with parameters
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train the tokenizer
    print(f"\nTraining BPE tokenizer...")
    print(f"  Target vocab size: {vocab_size:,}")
    print(f"  Min frequency: {min_frequency}")
    print(f"  Special tokens: {special_tokens}")

    tokenizer.train_from_iterator(
        text_iterator,
        trainer
    )

    # IMPORTANT: Enable padding after training
    tokenizer.enable_padding(
        pad_token="[PAD]",
        pad_id=tokenizer.token_to_id("[PAD]")
    )

    print(f"✓ Training complete!")
    print(f"  Final vocab size: {tokenizer.get_vocab_size():,}")

    return tokenizer

def save_tokenizer(tokenizer, output_path):
    tokenizer.save(str(output_path))
    print(f"\n✓ Tokenizer saved to: {output_path}")

def load_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"✓ Tokenizer loaded from: {tokenizer_path}")
    return tokenizer
