"""
run_pretrain.py -- train the model on the packed corpus

    python run_pretrain.py
    python run_pretrain.py --epochs 2
    python run_pretrain.py --resume checkpoints/pretrain_step4000.pt
    python run_pretrain.py --tiny --limit 2000

"""

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from jabarti_llm import (
    GPT,
    ModelConfig,
    PackedDataset,
    Tokenizer,
    Tracker,
    Trainer,
    TrainingConfig,
    collate_fn,
    load_checkpoint,
)
from jabarti_llm.config import steps_for_epochs, unique_run_name

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PACKED_DIR = Path("scripts/ch03-build-tokenizer/output")

def _make_loader(dataset, training_config, model_config,
                  use_shuffle: bool=True, shuffle_seed: int=42):
    if use_shuffle:
        shuffle_seed = shuffle_seed if shuffle_seed is not None else 42
        generator = torch.Generator().manual_seed(shuffle_seed)

    return DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=use_shuffle,
        generator=generator,
        collate_fn=lambda examples: collate_fn(
            examples, pad_id=model_config.pad_id
        )
    )

def _load_packed(path, model_config, limit=None):
    """Open a packed .bin as a dataset of fixed-length windows."""
    print(f"loading {path}")
    dataset = PackedDataset.from_meta(path, model_config)

    print(f"  {dataset.num_tokens:,} tokens → {len(dataset):,} windows "
          f"of {model_config.max_seq_len + 1}")

    if limit is not None and limit < len(dataset):
        dataset = Subset(dataset, range(limit))
        print(f"  --limit {limit:,}: using the first {len(dataset):,} windows")

    return dataset

def log_packing_samples(dataset, tokenizer, n=3, seed=0):
    """Print n random windows, decoded, so packing is visible rather than assumed."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    print(f"packing spot-check ({n} random windows):")

    for rank, idx in enumerate(indices, 1):
        ids = dataset[idx]["input_ids"].tolist()
        seams = ids.count(tokenizer.EOS)
        decoded = tokenizer.decode(ids)
        print(f"  [{rank}] {len(ids)} tokens, {seams} document seam(s) [EOS]")
        print(f"       ids : {ids[:12]}...")
        print(f"       text: {decoded[:120]}...")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--resume", type=Path, help="checkpoint to continue from")
    parser.add_argument("--epochs", type=float, default=1.0,
                        help="passes over the corpus; step count follows its size")
    parser.add_argument("--steps", type=int, default=None,
                        help="optimizer steps; overrides --epochs when given")
    parser.add_argument("--warmup", type=int, default=None,
                        help="warmup steps; lower this for short runs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None,
                        help="micro-batches per optimizer step (gradient accumulation)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)

    parser.add_argument("--limit", type=int, default=None,
                        help="train on only the first N windows (smoke tests)")

    parser.add_argument("--eval-limit", type=int, default=None,
                        help="evaluate on only the first N windows")

    parser.add_argument("--print-every", type=int, default=None,
                        help="print the loss line to the terminal every N steps "
                             "(tracker logging still follows --log-every)")
    parser.add_argument("--sample-every", type=int, default=None,
                        help="generate from fixed prompts every N steps and log them")
    parser.add_argument("--sample-prompt", action="append", dest="sample_prompts",
                        help="prompt to sample during training (repeatable)")
    parser.add_argument("--sample-tokens", type=int, default=None,
                        help="new tokens per generation sample")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                        help="shuffle training data each epoch (default: on)")
    parser.add_argument("--shuffle-seed", type=int, default=None, dest="shuffle_seed",
                        help="seed for the training-data shuffle for reproducibility")

    parser.add_argument("--data-dir", type=Path, default=PACKED_DIR,
                        help="directory holding train.bin and eval.bin")
    parser.add_argument("--no-tracking", action="store_true")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to train on (default: cuda if available)")

    parser.add_argument("--tiny", action="store_true", help="use the CPU-sized model")


    args = parser.parse_args()

    model_config = ModelConfig.tiny() if args.tiny else ModelConfig.jabarti()
    training_config = TrainingConfig(run_name=unique_run_name("pretrain"))

    for name, value in (
        ("warmup_steps", args.warmup),
        ("batch_size", args.batch_size),
        ("accumulation_steps", args.accumulation_steps),
        ("learning_rate", args.lr),
        ("weight_decay", args.weight_decay),
        ("print_every", args.print_every),
        ("sample_every", args.sample_every),
        ("sample_max_new_tokens", args.sample_tokens),
    ):
        if value is not None:
            setattr(training_config, name, value)

    print("loading tokenizer")
    tokenizer = Tokenizer.from_file()

    train_dataset = _load_packed(
        args.data_dir / "train.bin",
        model_config,
        limit=args.limit
    )

    train_loader = _make_loader(
        train_dataset, training_config, model_config,
        use_shuffle=args.shuffle, shuffle_seed=args.shuffle_seed
    )

    eval_dataset = _load_packed(
        args.data_dir / "eval.bin", model_config,
        limit=args.eval_limit
    )

    eval_loader = _make_loader(
        eval_dataset, training_config, model_config,
        use_shuffle=args.shuffle, shuffle_seed=args.shuffle_seed,
    )

    log_packing_samples(train_dataset, tokenizer)

    if args.steps is not None:
        training_config.max_steps = args.steps
    else:
        training_config.max_steps = steps_for_epochs(
            len(train_dataset), training_config.batch_size,
            training_config.accumulation_steps, args.epochs
        )

    model = GPT(model_config)

    print(f"pretrain: {len(train_dataset):,} windows, "
          f"{model.num_parameters():,} parameters")
    print(f"  {args.epochs:g} epoch(s) -> {training_config.max_steps:,} steps, "
          f"effective batch {training_config.batch_size * training_config.accumulation_steps}")

    final = None

    with Tracker(
        training_config=training_config, model_config=model_config
        ) as tracker:

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            config=training_config,
            eval_loader=eval_loader,
            tracker=tracker,
            tokenizer=tokenizer,
            device=args.device
        )
    
        if args.resume:
            trainer.step = load_checkpoint(
                args.resume, model, trainer.optimizer, trainer.scheduler
            )
            print(f"resumed {args.resume} at step {trainer.step:,} "
                  f"(optimizer and schedule restored)")

        trainer.train()

        final = trainer.save("pretrain_final.pt")

    print(f"saved {final}")

if __name__ == "__main__":
    main()
