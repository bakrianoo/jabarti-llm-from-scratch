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