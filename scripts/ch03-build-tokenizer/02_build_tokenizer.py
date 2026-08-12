"""
02 — Build the Bilingual BPE Tokenizer  (Arabic + English)
==========================================================
"""

from pathlib import Path

import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from cleaning import build_normalizer

# ============================================================
# CONFIGURATION
# ============================================================

CURRENT_WD = Path(__file__).parent
CLEAN_PATHS = [
    CURRENT_WD / "output" / "phase1_train_clean.parquet",
    CURRENT_WD / "output" / "phase2_train_clean.parquet",
]

OUTPUT_PATH = CURRENT_WD / "tokenizer.json"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]",
                  "[AR]", "[EN]",
                  "[SYS]", "[USER]", "[ASST]"]

# ============================================================
# 1. Assemble the tokenizer pipeline
# ============================================================

def build_tokenizer():

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = build_normalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always")
    tokenizer.decoder       = decoders.Metaspace(prepend_scheme="always")

    return tokenizer


# ============================================================
# 2. Train BPE on the corpus
# ============================================================
def corpus_iterator(df, batch_size=1000):
    texts = df["text"].tolist()
    for i in range(0, len(texts), batch_size):
        yield texts[i: i + batch_size]

def train(tokenizer, df):
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train_from_iterator(
        corpus_iterator(df),
        trainer=trainer,
        length=len(df),
    )

    return tokenizer

    
# ============================================================
# MAIN
# ============================================================
def main():

    print("STEP 1: LOAD CLEANED CORPUS")

    frames = []
    for clean_path in CLEAN_PATHS:
        frame = pd.read_parquet(clean_path)
        print(f"  {clean_path.name}: {len(frame)}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    print("STEP 2: BUILD PIPELINE + TRAIN BPE")
    tokenizer = build_tokenizer()
    tokenizer = train(tokenizer=tokenizer, df=df)

    print("SANITY CHECK: encode / decode bilingual examples")
    examples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة عاصمتها القاهرة",
        "أحمد إبراهيم آدم وُلد في مصر",
        "The United States and the United Nations signed the agreement",
        "machine learning models require large amounts of training data",
    ]

    for text in examples:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded_text = tokenizer.decode(encoded.ids)

        print(f"\n  input  : {text}")

        print(f"  tokens : {encoded.tokens}")
        print(f"  ids    : {encoded.ids}")

        print(f"  decoded: {decoded_text}")

        print("="*30)

    tokenizer.save(str(OUTPUT_PATH))

if __name__ == "__main__":
    main()        
        


