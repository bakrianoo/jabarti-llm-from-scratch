from bpe_tokenizer import load_tokenizer, normalize_arabic_text
from collections import Counter

def evaluate_tokenizer_quality(
        text: str,
        tokenizer_path: str="./saved/egyptian_wiki_bpe_tokenizer.json",
):

    # Load the Tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"  vocab size: {tokenizer.get_vocab_size():,}")

    encoding = tokenizer.encode(text)
    tokens_str = encoding.tokens
    tokens_ids = encoding.ids

    print(f"  Tokens:")
    for t, _id in zip(tokens_str, tokens_ids):
        print(t, '-->', _id)
    print("=")

    # Metric 1: Compression ratio
    # How many characters per token? Lower is better (more compression)
    chars_per_token = len(text) / len(tokens_str)

    print(f"\n1. COMPRESSION METRICS")
    print(f"  Sample length: {len(text):,} characters")
    print(f"  Token count: {len(tokens_str):,} tokens")
    print(f"  Chars per token: {chars_per_token:.2f}")

    # Metric 2: Token length distribution
    token_lengths = Counter(
        len(token)
        for token in tokens_str
    )

    for token_length in sorted(token_lengths.keys()):
        freq = token_lengths[token_length]
        print(f"Tokens with token length={token_length} -> Freq: {freq}")

    # Metric 3: Encoding Losslessness
    reconstructed_text = tokenizer.decode(tokens_ids)

    is_lossless = (text == reconstructed_text)
    if is_lossless:
        print("✓ Reconstructed: PASS")
    else:
        print("✗ Reconstructed: FAIL")
        print("Original:", text)
        print("Reconstructed:", reconstructed_text)

    
if __name__ == '__main__':
    evaluate_tokenizer_quality(
        text = normalize_arabic_text(
            """كما يذكرون أن والد المؤرخ ( حسن بن إبراهيم بن حسن بن علي بن محمد بن عبد الرحمن الجبرتي ) كان من أعلام علماء الأزهر الشريف في عصره، وكان يقوم بالتدريس فيه وفي مدرسة السنانية ببولاق بجامع سنان باشا، وكان على جانب كبير من الثراء، فكان له 3 بيوت في القاهرة (بالصنادقية وعلى نهر النيل ببولاق وبمصر القديمة)، وكانت مكتبته عامرة بالكتب القيمة والمخطوطات النادرة، كما كانت دوره آهلة في كل وقت بالعلماء والمجاورين (طلاب الأزهر كان يطلق عليهم المجاورين نظراً لتجاورهم في السكنى جنب بعضهم البعض ومنهم سليمان الحلبي)"""
        )
    )