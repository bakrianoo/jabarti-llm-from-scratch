"""
generate.py -- turning a trained model into text

Generation is a loop: predict the next token, append it, predict again. The
only real complication is that recomputing the whole prefix on every step
wastes almost all of the work, which is what the KV cache fixes.
"""

import torch

from jabarti_llm.generation.sampling import sample


@torch.no_grad()
def generate_ids(model, prompt_ids, config, eos_id=None, device=None):
    """Continue prompt_ids, returning prompt and continuation together.

    One sequence at a time. Batched generation needs left-padding and an
    attention mask to keep short prompts from polluting long ones -- real
    complexity that teaches nothing new about sampling.
    """
    model.eval()
    device = device or next(model.parameters()).device

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device=device).manual_seed(config.seed)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    max_seq_len = model.config.max_seq_len

    next_input = ids
    kv_caches = None

    for _ in range(config.max_new_tokens):
        if ids.size(1) >= max_seq_len:
            break

        logits, loss, updated_caches = model(
            next_input, kv_caches=kv_caches, use_cache=config.use_cache
        )

        kv_caches = updated_caches

        next_id = sample(logits, config, generator=generator)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        if eos_id is not None and next_id == eos_id:
            break

        if config.use_cache:
            next_input = ids[:, -1:]    # just the newest token
        else:
            next_input = ids

    return ids[0].tolist()

    # ids.tolist()      →  [[2, 11, 12, 13, 99]]   nested: a list of sequences
    # ids[0].tolist()   →   [2, 11, 12, 13, 99]    flat: one sequence of ints


def generate(model, tokenizer, prompt, config, device=None):
    """Continue a prompt, returning only the newly generated text.

    The prompt is prefixed with [BOS] because training wrapped every document
    that way (ch08); a model given a bare first token is being asked to
    continue from a state it never saw.
    """

    prompt_ids = [tokenizer.BOS] + tokenizer.encode(prompt)

    produced_ids = generate_ids(
        model, prompt_ids, config, eos_id=tokenizer.EOS, device=device
    )

    response_ids = produced_ids[len(prompt_ids):]
    return tokenizer.decode(response_ids)

