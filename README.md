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
python run_pretrain.py --epochs 1 \
  --batch-size 32 --accumulation-steps 4 \
  --lr 6e-4 --warmup 300 --weight-decay 0.1 \
  --print-every 100 --sample-every 500 --shuffle-seed 42
```

## TrackIO Panel 

**Run**

```bash
trackio show --project jabarti-llm --host 0.0.0.0
```

### Using CloudFlare Tunnel to access 

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

