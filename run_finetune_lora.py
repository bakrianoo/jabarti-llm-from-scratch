"""
run_finetune_lora.py -- turn the base model into an assistant, by training a
detour instead of the model itself

    python run_finetune_lora.py --resume checkpoints/pretrain_final.pt

"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jabarti_llm import (
    GPT,
    ChatDataset,
    ModelConfig,
    Tokenizer,
    Tracker,
    Trainer,
    TrainingConfig,
    apply_lora,
    chat_collate_fn,
    merge_lora,
    save_checkpoint,
)
from jabarti_llm.config import steps_for_epochs, unique_run_name
from jabarti_llm.training import load_checkpoint, model_config_from_checkpoint

FT_TRAIN = "scripts/ch03-build-tokenizer/output/ft_train_filtered.parquet"
FT_EVAL = "scripts/ch03-build-tokenizer/output/ft_eval_filtered.parquet"

FT_SAMPLE_PROMPTS = (
    # --- Arabic ---
    "ما اسم المعبد الجنائزي الذي شيّدته حتشبسوت، وأين يقع؟",                   # ancient history
    "مع أي ممثلة شارك عمر الشريف بطولة فيلم «صراع في الوادي»؟",                 # arts / film
    "إلى أين نُفي أحمد شوقي، ولماذا؟",                                          # literature
    "ما الموقف الشهير الذي اتخذته هدى شعراوي بعد عودتها من مؤتمر روما عام 1923؟",  # women's movement
    "إلى أين نُفي محمد عبده بعد فشل الثورة العرابية؟",                          # religious reform
    "ما الفيلم الذي تناول فيه يوسف شاهين سيرة الفيلسوف ابن رشد؟",               # arts / film
    "ما أول فيلم ظهرت فيه فاتن حمامة وهي طفلة؟",                                # arts / film
    "كيف انتهت حياة بطرس غالي رئيس وزراء مصر؟",                                 # modern history
    "ما اسم الفرقة المسرحية التي أسسها يوسف وهبي؟",                             # theatre
    "ما اسم المعبد الجنائزي الذي بناه رمسيس الثالث في طيبة؟",                   # ancient history
    # --- English ---
    "What is the Arabic title of Ibn al-Haytham's Book of Optics?",              # medieval science
    "What famous scene is painted in the tomb of Khnumhotep II at Beni Hasan?",  # ancient history
    "Which major U.S. science honor did Mostafa El-Sayed receive in 2007?",      # modern science
    "Which famous song did Georges Moustaki write for Édith Piaf?",              # music
    "What was unusual about Cleopatra VII among the Ptolemaic rulers of Egypt when it came to language?",  # ancient history
    "Which children's TV franchise did Haim Saban bring to American audiences in 1993?",  # media / business
    "In which country did Eli Cohen work as an Israeli spy, and under what false name?",  # modern history
    "Which international post did Farouk Hosny run for in 2009?",                # politics / culture
    "What is the title of Leila Ahmed's memoir about growing up in Egypt?",      # literature
    "Which puppet film trilogy is the artist Wael Shawky known for?",            # visual arts
)

FT_SAMPLE_REFERENCES = (
    "حتشبسوت شيّدت معبدها الجنائزي المعروف باسم «جسر جسرو» في الدير البحري بالبر الغربي لمدينة طيبة (الأقصر حالياً).",
    "عمر الشريف شارك فاتن حمامة بطولة فيلم «صراع في الوادي» عام 1954، ثم تزوجا في العام التالي.",
    "أحمد شوقي نفاه الإنجليز إلى إسبانيا عام 1915 بسبب قربه من الخديوي عباس حلمي الثاني بعد عزله، وعاد إلى مصر بعد انتهاء الحرب العالمية الأولى.",
    "هدى شعراوي خلعت نقابها علناً في محطة القطار بالقاهرة عند عودتها من مؤتمر الاتحاد النسائي الدولي في روما عام 1923.",
    "محمد عبده نُفي إلى بيروت عام 1882 بسبب تأييده للثورة العرابية، ثم لحق بجمال الدين الأفغاني في باريس.",
    "يوسف شاهين تناول سيرة ابن رشد في فيلم «المصير» عام 1997.",
    "فاتن حمامة ظهرت لأول مرة وهي طفلة في فيلم «يوم سعيد» عام 1940 أمام محمد عبد الوهاب.",
    "بطرس غالي اغتيل في القاهرة عام 1910 على يد إبراهيم ناصف الورداني.",
    "يوسف وهبي أسس «فرقة رمسيس» المسرحية عام 1923.",
    "رمسيس الثالث بنى معبده الجنائزي في مدينة هابو بالبر الغربي لطيبة.",
    "Ibn al-Haytham's Book of Optics is known in Arabic as Kitāb al-Manāẓir.",
    "Khnumhotep II's tomb at Beni Hasan shows a procession of Asiatic foreigners, the Aamu, led by a chief named Abisha.",
    "Mostafa El-Sayed received the U.S. National Medal of Science in 2007.",
    "Georges Moustaki wrote the lyrics of \"Milord\" for Édith Piaf in 1959.",
    "Cleopatra VII was the first Ptolemaic ruler known to have learned the Egyptian language.",
    "Haim Saban brought the Power Rangers franchise to American television in 1993.",
    "Eli Cohen worked as an Israeli spy in Syria under the false name Kamel Amin Thaabet.",
    "Farouk Hosny ran for Director-General of UNESCO in 2009 but lost to Irina Bokova.",
    "Leila Ahmed's memoir is titled A Border Passage: From Cairo to America—A Woman's Journey.",
    "Wael Shawky is known for Cabaret Crusades, a trilogy of films performed by marionettes.",
)


def build_loader(path, tokenizer, model_config, training_config,
                 limit, use_shuffle: bool=True, shuffle_seed: int=42):

    print(f"loading {path}")
    dataset = ChatDataset.from_parquet(path=path, tokenizer=tokenizer,
                                       config=model_config, limit=limit)

    print(f"  tokenized into {len(dataset):,} pairs")

    generator = None
    if use_shuffle:
        shuffle_seed = shuffle_seed if shuffle_seed is not None else 42
        generator = torch.Generator().manual_seed(shuffle_seed)

    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=use_shuffle,
        generator=generator,
        collate_fn=lambda examples: chat_collate_fn(examples, pad_id=model_config.pad_id)
    )

    return dataset, loader

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--resume", type=Path,
                        help="base checkpoint to finetune; omit to start from scratch")
    parser.add_argument("--epochs", type=float, default=3.0,
                        help="passes over the chat corpus; step count follows dataset size")
    parser.add_argument("--steps", type=int, default=None,
                        help="optimizer steps; overrides --epochs when given")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="micro-batches per optimizer step (gradient accumulation)")

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                            help="device to train on (default: cuda if available)")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=None,
                        help="print the loss line to the terminal every N steps "
                             "(tracker logging still follows --log-every)")
    parser.add_argument("--sample-every", type=int, default=None,
                        help="generate from fixed prompts every N steps and log them")
    parser.add_argument("--sample-prompt", action="append", dest="sample_prompts",
                        help="prompt to sample during training (repeatable)")
    parser.add_argument("--sample-tokens", type=int, default=None,
                        help="new tokens per generation sample")
    parser.add_argument("--save-every", type=int, default=None,
                        help="save a checkpoint every N steps (default: 1000)")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                        help="shuffle training data each epoch (default: on)")
    parser.add_argument("--shuffle-seed", type=int, default=None, dest="shuffle_seed",
                        help="seed for the training-data shuffle for reproducibility")

    parser.add_argument("--lora-r", type=int, default=8, dest="lora_r",
                        help="rank of the LoRA detour on each attention projection -- "
                             "higher r means a richer (and larger) detour")
    parser.add_argument("--lora-alpha", type=int, default=16, dest="lora_alpha",
                        help="LoRA scaling factor; the detour's output is scaled by "
                             "alpha/r before being added to the frozen layer's output")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                            help="where step and final checkpoints are written "
                                 "(default: checkpoints)")

    args = parser.parse_args()

    if args.resume:
        model_config = model_config_from_checkpoint(args.resume)
    else:
        model_config = ModelConfig.jabarti()

    training_config = TrainingConfig(
        run_name=unique_run_name("jabarti-finetune-lora"),
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
    )

    for name, value in (
        ("weight_decay", args.weight_decay),
        ("print_every", args.print_every),
        ("sample_every", args.sample_every),
        ("sample_max_new_tokens", args.sample_tokens),
        ("save_every", args.save_every),
        ("checkpoint_dir", args.checkpoint_dir),
    ):
        if value is not None:
            setattr(training_config, name, value)

    training_config.sample_prompts = FT_SAMPLE_PROMPTS

    print("loading tokenizer")
    tokenizer = Tokenizer.from_file()

    train_dataset, train_loader = build_loader(
        FT_TRAIN, tokenizer, model_config, training_config,
        limit=args.limit, use_shuffle=args.shuffle, 
        shuffle_seed=args.shuffle_seed,
    )

    eval_dataset, eval_loader = build_loader(
        FT_EVAL, tokenizer, model_config, training_config, 
        limit=args.limit, use_shuffle=False
    )

    if args.steps is not None:
        training_config.max_steps = args.steps
    else:
        training_config.max_steps = steps_for_epochs(
           num_examples=len(train_dataset), batch_size=training_config.batch_size,
           accumulation_steps=training_config.accumulation_steps,
           epochs=args.epochs
        )

    model = GPT(model_config)
    if args.resume:
        step = load_checkpoint(args.resume, model)
        print(f"finetuning {args.resume} (pretrained {step} steps)")

    replaced = apply_lora(model=model, r=args.lora_r, alpha=args.lora_alpha)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = model.num_parameters()

    print(f"LoRA: wrapped {replaced} attention projection(s) at rank {args.lora_r} "
          f"(alpha={args.lora_alpha})")
    
    print(f"  {trainable:,} trainable / {total:,} total parameters "
          f"({trainable / total:.2%})")

    print(f"{len(train_dataset):,} question/answer pairs")

    print(f"  {args.epochs:g} epoch(s) -> {training_config.max_steps:,} steps, "
          f"effective batch {training_config.batch_size * training_config.accumulation_steps}")

    with Tracker(
        training_config=training_config,
        model_config=model_config,
    ) as tracker:

        trainer = Trainer(
            model=model, train_loader=train_loader, eval_loader=eval_loader,
            config=training_config, tracker=tracker, tokenizer=tokenizer,
        )

        trainer.train()

    merged = merge_lora(model)
    print(f"merged {merged} LoRA adapter(s) back into their base projections")

    final = save_checkpoint(
        f"{training_config.checkpoint_dir}/finetune_chat_lora_final.pt",
        model, step=trainer.step
    )

    print(f"saved {final}")


if __name__ == "__main__":
    main()