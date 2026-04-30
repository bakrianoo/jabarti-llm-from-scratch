"""
Step 06 — Translate the finetune_qa phase to English
=====================================================

The ``finetune_qa`` split of the jabarti dataset contains Arabic Q&A pairs
for instruction fine-tuning.  This step translates each Q&A pair from Arabic
to English using a locally-running vLLM service that hosts
``google/translategemma-4b-it``.

Translation strategy: **joint title + question + answer**
---------------------------------------------------------
Translating the question and the answer in two independent calls causes
entity-name drift — the same Arabic figure name ends up with two different
English transliterations between Q and A (e.g. "Antoine Shuha" vs
"Antwan Chouché").

To fix this, the AR title (looked up from ``train_phase_2`` via
``train_phase_2_id``), the question and the answer are concatenated with a
unique sentinel and translated in a **single** call.  The model sees all
three segments in the same context and keeps named-entity transliterations
consistent.  We then split the response back into the three pieces and
store the question and answer.

If the sentinel is lost in the response (rare with translategemma), we fall
back to per-segment translation — the legacy behaviour.

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
        "ar_answer":         <original Arabic answer>,
        "ar_title_anchor":   <AR title used as the anchor>,
        "en_title_anchor":   <translated EN title>,
        "joint_translation": <bool: True if the joint call succeeded>
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

    # disable joint translation and use the legacy per-segment strategy
    python step06-translate-finetune-dataset-to-english.py --no-joint

The script is fully resumable: it reads qa_id values already present in the
output file and skips them, so interrupted runs can be restarted safely.
"""

from __future__ import annotations

import argparse
import json
import re
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
# Joint mode bundles 3 segments in one call, so the budget is ~3x larger
# than what one segment needs.
DEFAULT_MAX_TOKENS = 1024
TRANSLATION_MODEL = "google/translategemma-4b-it"
REQUEST_TIMEOUT = 120          # seconds per individual HTTP request
MAX_RETRIES = 5                # per-call translation retries
RETRY_BACKOFF_BASE = 1.0       # seconds; doubled on each failure

TEMPLATE_PATH = (
    Path(__file__).parent.parent / "assets" / "translategemma_chat_template.jinja"
)

HF_DATASET = "bakrianoo/jabarti"

# Sentinel used to separate the three segments inside one joint translation
# call.  Picked to be unlikely to occur in real text and to be copied
# verbatim by translategemma.  ``SENTINEL_RE`` is the tolerant matcher used
# when splitting the model output (allows minor whitespace shifts).
SENTINEL = "<<<|SEGMENT|>>>"
SENTINEL_RE = re.compile(r"\s*<<<\s*\|?\s*SEGMENT\s*\|?\s*>>>\s*")

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
# Title anchor lookup (built from train_phase_2)
# ---------------------------------------------------------------------------


def build_title_index() -> dict[int, str]:
    """Map ``train_phase_2`` row index -> Arabic title.

    Each ``finetune_qa`` row has a ``train_phase_2_id`` that points to the
    AR LLM-generated article it was derived from.  That article's title is
    the Arabic biographical headline of the figure (e.g.
    "حسن باشا الاسكندراني: أمير البحار المصري في القرن التاسع عشر").
    Including it as the first segment of the joint translation gives the
    model a strong anchor for the figure's name.
    """
    print("Loading train_phase_2 to build the title lookup …")
    ds = load_dataset(HF_DATASET, split="train_phase_2")
    titles: dict[int, str] = {}
    for idx, row in enumerate(ds):
        title = (row.get("title") or "").strip()
        if title:
            titles[idx] = title
    print(f"  built {len(titles):,} title entries")
    return titles


# ---------------------------------------------------------------------------
# Joint translation helpers
# ---------------------------------------------------------------------------


def _build_bundle(title: str, question: str, answer: str) -> str:
    sep = f"\n\n{SENTINEL}\n\n"
    return f"{title.strip()}{sep}{question.strip()}{sep}{answer.strip()}"


def _split_translation(out: str) -> Optional[list[str]]:
    """Split *out* on the sentinel; return exactly 3 trimmed pieces or None."""
    parts = [p.strip() for p in SENTINEL_RE.split(out)]
    parts = [p for p in parts if p != ""]
    if len(parts) == 3:
        return parts
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def translate_record(
    row: dict,
    client: TranslationClient,
    template,
    titles: dict[int, str],
    *,
    use_joint: bool = True,
) -> Optional[dict]:
    """Translate a single ``finetune_qa`` record.  Returns None on failure.

    By default this uses the **joint** strategy: title + question + answer
    are translated in one call separated by ``SENTINEL`` so the model keeps
    entity-name transliterations consistent across the three segments.
    Pass ``use_joint=False`` to fall back to the legacy per-segment calls.
    """
    ar_text = (row.get("text") or "").strip()
    ar_answer = (row.get("answer") or "").strip()
    tp2_id = int(row.get("train_phase_2_id", -1))
    ar_title = titles.get(tp2_id, "").strip() if titles else ""
    # Placeholder so the bundle always has 3 non-empty segments.
    ar_title_for_bundle = ar_title or "—"

    en_title = ""
    joint_ok = False

    if use_joint:
        bundle = _build_bundle(ar_title_for_bundle, ar_text, ar_answer)
        out = client.translate(template, bundle)
        if out is None:
            return None
        parts = _split_translation(out)
        if parts is not None:
            en_title, en_text, en_answer = parts
            joint_ok = True

    if not joint_ok:
        # Legacy / fallback path: translate each segment independently.
        en_text = client.translate(template, ar_text)
        if en_text is None:
            return None
        en_answer = client.translate(template, ar_answer)
        if en_answer is None:
            return None
        if use_joint and ar_title:
            # Best effort: still try to translate the title for provenance.
            en_title = client.translate(template, ar_title) or ""

    return {
        "text": en_text,
        "answer": en_answer,
        "title": row.get("title") or "",
        "url": row.get("url") or "",
        "category": row.get("category") or "",
        "rec_id": int(row.get("rec_id", -1)),
        "train_phase_2_id": tp2_id,
        "qa_id": int(row.get("qa_id", -1)),
        "language": "en",
        "phase": "finetune_qa",
        "ar_text": ar_text,
        "ar_answer": ar_answer,
        "ar_title_anchor": ar_title,
        "en_title_anchor": en_title,
        "joint_translation": joint_ok,
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
    p.add_argument(
        "--no-joint",
        action="store_true",
        help="Disable joint title+Q+A translation; use legacy per-segment "
             "calls (causes entity-name drift between Q and A).",
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

    # ---- title anchor lookup (skipped when joint mode is off) --------------
    use_joint = not args.no_joint
    titles: dict[int, str] = build_title_index() if use_joint else {}

    # ---- load HuggingFace dataset ------------------------------------------
    print(f"Loading {HF_DATASET} finetune_qa split …")
    ds = load_dataset(HF_DATASET, split="finetune_qa")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    records = list(ds)
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
    joint_ok_count = 0
    fallback_count = 0

    with args.output.open("a", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    translate_record, row, client, template, titles,
                    use_joint=use_joint,
                ): row
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
                        if result.get("joint_translation"):
                            joint_ok_count += 1
                        elif use_joint:
                            fallback_count += 1
                        with write_lock:
                            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_fh.flush()

                    pbar.update(1)

    # ---- summary -----------------------------------------------------------
    success = len(pending) - len(failed_ids)
    print(f"\nDone.  Translated: {success:,} | Failed: {len(failed_ids):,}")
    if use_joint:
        print(f"  Joint OK : {joint_ok_count:,}")
        print(f"  Fallback : {fallback_count:,}  (sentinel split failed)")
    print(f"Output: {args.output}")
    if failed_ids:
        print(f"Failed qa_ids (first 20): {failed_ids[:20]}")


if __name__ == "__main__":
    main()
