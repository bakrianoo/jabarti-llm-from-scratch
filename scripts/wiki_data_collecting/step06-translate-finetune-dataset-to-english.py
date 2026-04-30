"""
Step 06 — Translate the finetune_qa phase to English
=====================================================

The ``finetune_qa`` split of the jabarti dataset contains Arabic Q&A pairs
for instruction fine-tuning.  This step translates both the question (``text``)
and the answer (``answer``) from Arabic to English using a locally-running vLLM
service that hosts ``google/translategemma-4b-it``.

The resulting English Q&A pairs are appended to the corpus so the model can
be fine-tuned bilingually, in the same spirit as step05 for the base-model
phases.

Output schema per JSONL line (mirrors the HF dataset layout + provenance):

    {
        "text":              <translated English question>,
        "answer":            <translated English answer>,
        "title":             "",
        "url":               <English Wikipedia URL>,
        "category":          <category string>,
        "rec_id":            -1,
        "train_phase_2_id":  <int>,
        "qa_id":             <int>,
        "language":          "en",
        "phase":             "finetune_qa",
        "ar_text":           <original Arabic question>,
        "ar_answer":         <original Arabic answer>
    }

Usage
-----
    # translate full finetune_qa split (default output path)
    python step06-translate-finetune-dataset-to-english.py

    # custom output path and worker count
    python step06-translate-finetune-dataset-to-english.py \\
        --output ./data/finetune-en.jsonl \\
        --workers 4

    # point at a non-default vLLM server
    python step06-translate-finetune-dataset-to-english.py \\
        --vllm-url http://my-server:8000

The script is fully resumable: it reads qa_id values already present in the
output file and skips them, so interrupted runs can be restarted safely.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import requests
from datasets import load_dataset
from jinja2 import Environment, StrictUndefined
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "finetune-en.jsonl"
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_WORKERS = 4
DEFAULT_MAX_TOKENS = 512
TRANSLATION_MODEL = "google/translategemma-4b-it"
REQUEST_TIMEOUT = 120          # seconds per individual HTTP request
MAX_RETRIES = 5                # per-call translation retries
RETRY_BACKOFF_BASE = 1.0       # seconds; doubled on each failure

TEMPLATE_PATH = (
    Path(__file__).parent.parent / "assets" / "translategemma_chat_template.jinja"
)

# ---------------------------------------------------------------------------
# Jinja2 chat template helpers (matches test_translategemma.py)
# ---------------------------------------------------------------------------


def load_template(template_path: Path = TEMPLATE_PATH):
    env = Environment(undefined=StrictUndefined, keep_trailing_newline=True)

    def _raise(msg):
        raise RuntimeError(msg)

    env.globals["raise_exception"] = _raise
    return env.from_string(template_path.read_text(encoding="utf-8"))


def render_prompt(template, source_lang: str, target_lang: str, text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang,
                    "target_lang_code": target_lang,
                    "text": text,
                }
            ],
        }
    ]
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        bos_token="<bos>",
    )


# ---------------------------------------------------------------------------
# Translation client
# ---------------------------------------------------------------------------


@dataclass
class TranslationClient:
    base_url: str
    model: str = TRANSLATION_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = REQUEST_TIMEOUT
    max_retries: int = MAX_RETRIES

    def translate(
        self,
        template,
        text: str,
        source_lang: str = "ar",
        target_lang: str = "en",
    ) -> Optional[str]:
        """Translate *text* with retries.  Returns None on permanent failure."""
        if not text or not text.strip():
            return ""

        prompt = render_prompt(template, source_lang, target_lang, text)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            # Template already emits <bos>; don't let the server add another.
            "add_special_tokens": False,
        }

        backoff = RETRY_BACKOFF_BASE
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["text"].strip()
            except (requests.RequestException, KeyError, ValueError) as exc:
                if attempt == self.max_retries:
                    print(
                        f"\n  [translate] permanent failure after {attempt} attempts: {exc}",
                        file=sys.stderr,
                    )
                    return None
                wait = backoff + (attempt * 0.1)
                time.sleep(wait)
                backoff = min(backoff * 2, 30.0)

        return None


# ---------------------------------------------------------------------------
# Resumability helpers
# ---------------------------------------------------------------------------


def load_translated_ids(output_path: Path) -> set[int]:
    """Return the set of qa_id values already present in the output file."""
    if not output_path.exists():
        return set()
    translated: set[int] = set()
    try:
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qa_id = rec.get("qa_id")
                if qa_id is not None and qa_id != -1:
                    translated.add(int(qa_id))
        print(f"Resuming: found {len(translated):,} already-translated records in {output_path}")
    except Exception as exc:
        print(f"Warning: could not read existing output ({exc}); starting fresh.")
        return set()
    return translated


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def translate_record(
    row: dict,
    client: TranslationClient,
    template,
) -> Optional[dict]:
    """Translate a single finetune_qa record.  Returns None on failure."""
    ar_text = (row.get("text") or "").strip()
    ar_answer = (row.get("answer") or "").strip()

    en_text = client.translate(template, ar_text)
    if en_text is None:
        return None

    en_answer = client.translate(template, ar_answer)
    if en_answer is None:
        return None

    return {
        "text": en_text,
        "answer": en_answer,
        "title": row.get("title") or "",
        "url": row.get("url") or "",
        "category": row.get("category") or "",
        "rec_id": int(row.get("rec_id", -1)),
        "train_phase_2_id": int(row.get("train_phase_2_id", -1)),
        "qa_id": int(row.get("qa_id", -1)),
        "language": "en",
        "phase": "finetune_qa",
        "ar_text": ar_text,
        "ar_answer": ar_answer,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Translate finetune_qa Q&A pairs from Arabic to English "
        "using a local vLLM translategemma server."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--vllm-url",
        default=DEFAULT_VLLM_URL,
        help=f"Base URL of the vLLM server (default: {DEFAULT_VLLM_URL})",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel translation workers (default: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens per translation call (default: {DEFAULT_MAX_TOKENS})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only the first N records (useful for testing)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ---- validate vLLM is reachable ----------------------------------------
    try:
        r = requests.get(f"{args.vllm_url}/v1/models", timeout=10)
        r.raise_for_status()
        available_models = [m["id"] for m in r.json().get("data", [])]
        if TRANSLATION_MODEL not in available_models:
            print(
                f"Warning: '{TRANSLATION_MODEL}' not found in vLLM server models: "
                f"{available_models}",
                file=sys.stderr,
            )
    except requests.RequestException as exc:
        print(f"Error: cannot reach vLLM server at {args.vllm_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- load HuggingFace dataset ------------------------------------------
    print("Loading bakrianoo/jabarti finetune_qa split …")
    ds = load_dataset("bakrianoo/jabarti", split="finetune_qa")
    records = list(ds)
    if args.limit:
        records = records[: args.limit]
    print(f"Total records to consider: {len(records):,}")

    # ---- resume: skip already translated qa_ids ----------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    translated_ids = load_translated_ids(args.output)
    pending = [r for r in records if int(r.get("qa_id", -1)) not in translated_ids]
    print(f"Pending (not yet translated): {len(pending):,}")

    if not pending:
        print("Nothing to do — all records already translated.")
        return

    # ---- set up client and template ----------------------------------------
    client = TranslationClient(
        base_url=args.vllm_url,
        max_tokens=args.max_tokens,
    )
    template = load_template()

    # ---- translate with thread pool ----------------------------------------
    write_lock = Lock()
    failed_ids: list[int] = []

    with args.output.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(translate_record, row, client, template): row
                for row in pending
            }

            with tqdm(total=len(pending), desc="Translating finetune_qa", unit="rec") as pbar:
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = None
                        print(
                            f"\n  [worker] unhandled error for qa_id={row.get('qa_id')}: {exc}",
                            file=sys.stderr,
                        )

                    if result is None:
                        failed_ids.append(int(row.get("qa_id", -1)))
                    else:
                        with write_lock:
                            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_fh.flush()

                    pbar.update(1)

    # ---- summary -----------------------------------------------------------
    success = len(pending) - len(failed_ids)
    print(f"\nDone.  Translated: {success:,} | Failed: {len(failed_ids):,}")
    print(f"Output: {args.output}")
    if failed_ids:
        print(f"Failed qa_ids (first 20): {failed_ids[:20]}")


if __name__ == "__main__":
    main()
