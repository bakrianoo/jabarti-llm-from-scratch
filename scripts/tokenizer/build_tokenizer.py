from bpe_tokenizer import (
    create_bpe_tokenizer,
    train_bpe_tokenizer,
    save_tokenizer,
    load_tokenizer
)
from datasets import load_dataset

def main(
      vocab_size: int=32_000,
      min_frequency: int=5,
      special_tokens: list=[
          "[PAD]",      # Padding token
          "[UNK]",      # Unknown token
          "[BOS]",      # Beginning of sequence
          "[EOS]",      # End of sequence
      ],
      tokenizer_path: str="./saved/egyptian_wiki_bpe_tokenizer.json"
):

    # ============================================================
    # STEP 1
    # ============================================================
    print("STEP 1: LOAD EGYPTIAN WIKIPEDIA DATASET")

    ds = load_dataset("bakrianoo/jabarti")
    train_data = ds['train_phase_1']

    all_texts = []
    total_chars = 0

    for item in train_data:

        title = item.get('title', '').strip()
        text = item.get('text', '').strip()

        if not text or len(text) < 50:
            continue

        if title and len(title) > 10:
            text = f"{title}\n\n{text}"

        total_chars += len(text)
        all_texts.append(text)

    print(f"✓ Extracted {len(all_texts)} valid articles")
    print(f"✓ Total characters: {total_chars:,}")

    # ============================================================
    # STEP 2
    # ============================================================
    print("STEP 2: CREATE BPE TOKENIZER")

    tokenizer = create_bpe_tokenizer()
    tokenizer = train_bpe_tokenizer(
        tokenizer=tokenizer,
        text_iterator=all_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens
    )

    # ============================================================
    # STEP 3:
    # ============================================================
    print("STEP 3: SAVE TOKENIZER")
    
    save_tokenizer(tokenizer, 
                   tokenizer_path)
    
    print("TOKENIZER BUILD COMPLETE ✓")

if __name__ == '__main__':
    main()