# Running `jabarti_llm`

A bilingual (Arabic + English) GPT built from scratch. This guide takes you from
a clean checkout to a **fully finetuned chat model**, then shows how to test it,
trace training with Trackio, and where every artifact lands.

For *why* each piece is built the way it is, read the chapters in `chapters/` and
the package guide in [`jabarti_llm/README.md`](jabarti_llm/README.md).

---

## 1. Install

The project is a standard Python package (Python 3.10+)

```bash
python -m ensurepip
pip install uv

uv pip install -e "." 
```


---

## Run PreTrain

**Sample Run**

```bash
python run_pretrain.py --steps 60 --warmup 10 --limit 20000 --eval-limit 2000 \
  --batch-size 32 --accumulation-steps 4 --lr 6e-4 \
  --print-every 10 --sample-every 30 --shuffle-seed 42
```

**Full Run**

```bash
python run_pretrain.py --epochs 15 \
  --batch-size 56 --accumulation-steps 4 \
  --lr 6e-4 --warmup 500 --weight-decay 0.1 --eval-every 100 \
  --print-every 100 --sample-every 100 --shuffle-seed 42 \
  --checkpoint-dir ../checkpoints/jabarti-512x8-ep15 \
  --save-every 100
```

The final pretrained model is saved as `pretrain_final.pt` inside `--checkpoint-dir`.

---

## Run LoRA Finetune

Point `--resume` at the pretrained checkpoint. The LoRA adapters are merged back
into the model at the end and saved as `finetune_chat_lora_final.pt` inside
`--checkpoint-dir`.

**Sample Run**

```bash
python run_finetune_lora.py --resume ../checkpoints/jabarti-512x8-ep15/pretrain_final.pt \
  --steps 60 --warmup 10 --limit 2000 \
  --batch-size 8 --lr 2e-4 --lora-r 8 --lora-alpha 16 \
  --print-every 10 --sample-every 30 --shuffle-seed 42
```

**Full Run**

```bash
python run_finetune_lora.py \
  --resume ../checkpoints/jabarti-512x8-ep15/jabarti-512x8/pretrain_final.pt \
  --epochs 30 --batch-size 56 --accumulation-steps 4 \
  --lr 2e-4 --warmup 100 --lora-r 16 --lora-alpha 32 \
  --print-every 50 --sample-every 100 --shuffle-seed 42 \
  --checkpoint-dir /workspace/checkpoints/jabarti-512x8-lora \
  --save-every 500
```

- `--lora-r` — rank of the LoRA detour (higher = more capacity, more trainable params).
- `--lora-alpha` — scaling factor; the detour output is scaled by `alpha / r`.

---

## TrackIO Panel 

**Run**

```bash
trackio show --project jabarti-llm --host 0.0.0.0
```

### Using CloudFlare to Access TrackIO Panel Remotely

1. Install Cloudflare Tunnel

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

2. Run

```bash
cloudflared tunnel --protocol http2 --url http://localhost:7860
```

