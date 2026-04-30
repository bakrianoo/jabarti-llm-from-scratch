#!/usr/bin/env python3
"""
clean_and_build_dataset.py  —  jabarti bilingual dataset builder
=================================================================

Cleans the original jabarti HuggingFace dataset and merges it with the
English extensions produced by steps 05, 06 and 07, then writes a new
DatasetDict ready for upload to the HuggingFace Hub.

Sources
-------
  HF dataset  : bakrianoo/jabarti
    train_phase_1   — 32,015 Arabic Wikipedia articles
    train_phase_2   — 18,088 Arabic LLM-generated articles (EN-sourced figures)
    finetune_qa     — 36,176 Arabic Q&A pairs

  Local JSONL (produced by steps 05, 06 & 07):
    data/phase1-en.jsonl    — 35,105 English Wikipedia counterparts of phase_1
    data/phase2-en.jsonl    —  9,214 English Wikipedia pages for phase_2 figures
    data/finetune-en.jsonl  — 36,176 English translations of finetune_qa

Output splits
-------------
  train_phase_1  : cleaned Arabic phase_1  +  English phase_1 equivalents
  train_phase_2  : cleaned Arabic phase_2  +  English phase_2 equivalents
  finetune_qa    : Arabic Q&A pairs        +  English translated Q&A pairs

Unified schema (all splits)
---------------------------
  text             str   Article body (base splits) or question (finetune)
  answer           str   Answer text; "" for base splits
  title            str   Article / page title; "" for finetune rows
  url              str   Source Wikipedia URL
  category         str   Category string
  language         str   "ar" | "en"
  phase            str   "phase_1" | "phase_2" | "finetune_qa"
  rec_id           str   Stable record ID (normalised to str; "" when absent)
  qa_id            int   Q&A pair ID; -1 for base-model records
  train_phase_2_id int   Link to train_phase_2 record; -1 for base records
  ar_url           str   Linked Arabic Wikipedia URL (EN records); "" if none
  ar_title         str   Linked Arabic article title (EN records); "" if none

Cleaning applied per source
----------------------------
  AR train_phase_1 (HF):
    • drop rows with empty text
    • drop Wikipedia meta-pages: template (قالب:), portal (بوابة:) titles
    • drop texts shorter than MIN_CHARS after stripping
    • truncate texts longer than MAX_CHARS at the last paragraph break
    • optionally drop stub articles (category contains "بذرة")

  AR train_phase_2 (HF):
    • normalise schema only (already clean: min 295 chars, 0 empty)

  EN phase1-en.jsonl (step05):
    • normalise schema (already filtered by step05: 279–19,998 chars)

  EN phase2-en.jsonl (step05):
    • fix ar_url / ar_title bug: step05 copies the English URL into ar_url
      for phase_2 records (source had no Arabic URLs); reset both to ""
    • normalise schema

  AR finetune_qa (HF) + EN finetune-en.jsonl (step06):
    • normalise schema only (both sources are already clean)

Usage
-----
    # dry-run: clean & save locally, skip Hub upload
    python clean_and_build_dataset.py --no-upload

    # full run: clean, save locally, push to Hub
    python clean_and_build_dataset.py --hub-repo bakrianoo/jabarti-bilingual

    # custom data paths
    python clean_and_build_dataset.py \\
        --data-dir  ./scripts/wiki_data_collecting/data \\
        --output-dir ./data/clean_build \\
        --hub-repo  bakrianoo/jabarti-bilingual

    # adjust quality thresholds
    python clean_and_build_dataset.py \\
        --min-chars 300 \\
        --max-chars 15000 \\
        --keep-stubs          # keep stub articles (default: drop them)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Filtering thresholds  (override via CLI flags)
# ---------------------------------------------------------------------------

DEFAULT_MIN_CHARS: int = 250        # minimum text length after stripping
DEFAULT_MAX_CHARS: int = 20_000     # hard cap; truncated at last paragraph break
DEFAULT_FILTER_STUBS: bool = True   # drop articles whose category contains "بذرة"

# Arabic Wikipedia meta-namespace prefixes that should be excluded
AR_NOISE_PREFIXES = (
    "قالب:",    # Template:
    "بوابة:",   # Portal:  (note: some titles start with "بوابة" without colon)
    "بوابة",    # Portal (no colon variant)
    "تصنيف:",   # Category:
    "ملف:",     # File:
    "مستخدم:",  # User:
    "نقاش:",    # Talk:
    "ويكيبيديا:",  # Wikipedia:
)

# ---------------------------------------------------------------------------
# Unified schema helpers
# ---------------------------------------------------------------------------

_SCHEMA_FIELDS = (
    "text",
    "answer",
    "title",
    "url",
    "category",
    "language",
    "phase",
    "rec_id",
    "qa_id",
    "train_phase_2_id",
    "ar_url",
    "ar_title",
)

_SCHEMA_DEFAULTS: dict = {
    "text": "",
    "answer": "",
    "title": "",
    "url": "",
    "category": "",
    "language": "",
    "phase": "",
    "rec_id": "",
    "qa_id": -1,
    "train_phase_2_id": -1,
    "ar_url": "",
    "ar_title": "",
}


def _to_str(v) -> str:
    """Coerce a value to str, replacing None / -1 / missing with ""."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s == "-1" else s


def _to_int(v, default: int = -1) -> int:
    """Coerce a value to int, returning *default* on failure."""
    if v is None:
        return default
    try:
        return int(v)
    except (ValueError, TypeError):
        return default


def normalise(record: dict) -> dict:
    """Return a new dict containing exactly the unified schema fields."""
    out = dict(_SCHEMA_DEFAULTS)
    for field in _SCHEMA_FIELDS:
        if field in record:
            raw = record[field]
            if field in ("qa_id", "train_phase_2_id"):
                out[field] = _to_int(raw)
            elif field == "rec_id":
                out[field] = _to_str(raw)
            else:
                out[field] = "" if raw is None else str(raw)
    return out


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Collapse excessive blank lines."""
    return _MULTI_BLANK_RE.sub("\n\n", text).strip()


def truncate_at(text: str, max_chars: int) -> str:
    """
    Truncate *text* to at most *max_chars* characters, preferring a clean
    paragraph boundary (double newline).  Falls back to the last single
    newline, then to a hard cut.
    """
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    # prefer last double-newline
    idx = candidate.rfind("\n\n")
    if idx > max_chars // 2:
        return candidate[:idx].rstrip()
    # fall back to last single newline
    idx = candidate.rfind("\n")
    if idx > max_chars // 2:
        return candidate[:idx].rstrip()
    # hard cut
    return candidate.rstrip()


def is_noise_page(title: str) -> bool:
    """Return True for Wikipedia meta-namespace pages (templates, portals …)."""
    if not title:
        return False
    for prefix in AR_NOISE_PREFIXES:
        if title.startswith(prefix):
            return True
    return False


def is_stub(category: str) -> bool:
    """Return True when the category marks this as a stub article (بذرة)."""
    return bool(category and "بذرة" in category)


# ---------------------------------------------------------------------------
# Per-source processors
# ---------------------------------------------------------------------------

def process_ar_phase1(
    hf_split,
    min_chars: int,
    max_chars: int,
    filter_stubs: bool,
) -> tuple[list[dict], dict]:
    """
    Clean the HF ``train_phase_1`` Arabic split.

    Returns (records, stats) where stats holds per-reason drop counts.
    """
    stats: dict[str, int] = {
        "input": len(hf_split),
        "drop_empty": 0,
        "drop_noise_page": 0,
        "drop_stub": 0,
        "drop_too_short": 0,
        "truncated": 0,
        "kept": 0,
    }
    out: list[dict] = []

    for row in hf_split:
        text: str = (row.get("text") or "").strip()
        title: str = (row.get("title") or "").strip()
        category: str = (row.get("category") or "").strip()

        if not text:
            stats["drop_empty"] += 1
            continue
        if is_noise_page(title):
            stats["drop_noise_page"] += 1
            continue
        if filter_stubs and is_stub(category):
            stats["drop_stub"] += 1
            continue
        if len(text) < min_chars:
            stats["drop_too_short"] += 1
            continue

        if len(text) > max_chars:
            text = truncate_at(text, max_chars)
            stats["truncated"] += 1

        out.append(normalise({
            "text": clean_text(text),
            "answer": "",
            "title": title,
            "url": row.get("url") or "",
            "category": category,
            "language": "ar",
            "phase": "phase_1",
            "rec_id": _to_str(row.get("rec_id")),
            "qa_id": -1,
            "train_phase_2_id": -1,
            "ar_url": "",
            "ar_title": "",
        }))

    stats["kept"] = len(out)
    return out, stats


def process_ar_phase2(hf_split) -> tuple[list[dict], dict]:
    """
    Normalise the HF ``train_phase_2`` Arabic split (already clean).
    """
    out: list[dict] = []
    for row in hf_split:
        out.append(normalise({
            "text": (row.get("text") or "").strip(),
            "answer": "",
            "title": (row.get("title") or "").strip(),
            "url": row.get("url") or "",
            "category": (row.get("category") or "").strip(),
            "language": "ar",
            "phase": "phase_2",
            "rec_id": _to_str(row.get("rec_id")),
            "qa_id": -1,
            "train_phase_2_id": -1,
            "ar_url": "",
            "ar_title": "",
        }))
    return out, {"input": len(hf_split), "kept": len(out)}


def process_ar_finetune(hf_split) -> tuple[list[dict], dict]:
    """
    Normalise the HF ``finetune_qa`` Arabic split (already clean).
    """
    out: list[dict] = []
    for row in hf_split:
        out.append(normalise({
            "text": (row.get("text") or "").strip(),
            "answer": (row.get("answer") or "").strip(),
            "title": (row.get("title") or "").strip(),
            "url": row.get("url") or "",
            "category": (row.get("category") or "").strip(),
            "language": "ar",
            "phase": "finetune_qa",
            "rec_id": _to_str(row.get("rec_id")),
            "qa_id": _to_int(row.get("qa_id")),
            "train_phase_2_id": _to_int(row.get("train_phase_2_id")),
            "ar_url": "",
            "ar_title": "",
        }))
    return out, {"input": len(hf_split), "kept": len(out)}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] {path.name}:{lineno}: JSON parse error — {exc}",
                      file=sys.stderr)
    return records


def process_en_phase1(path: Path) -> tuple[list[dict], dict]:
    """
    Normalise the English phase_1 JSONL (already filtered by step05).
    """
    raw = load_jsonl(path)
    out: list[dict] = []
    for r in raw:
        out.append(normalise({
            "text": (r.get("text") or "").strip(),
            "answer": "",
            "title": (r.get("title") or "").strip(),
            "url": r.get("url") or "",
            "category": (r.get("category") or "").strip(),
            "language": "en",
            "phase": "phase_1",
            "rec_id": _to_str(r.get("rec_id")),
            "qa_id": -1,
            "train_phase_2_id": -1,
            "ar_url": r.get("ar_url") or "",
            "ar_title": (r.get("ar_title") or "").strip(),
        }))
    return out, {"input": len(raw), "kept": len(out)}


def process_en_phase2(path: Path) -> tuple[list[dict], dict]:
    """
    Normalise the English phase_2 JSONL and fix the ar_url / ar_title bug.

    Bug: step05 populates ``ar_url`` with the English URL (the source
    ``egyptian_figures_data.jsonl`` only has English URLs, so there is no
    real Arabic URL to link).  We reset both ``ar_url`` and ``ar_title``
    to "" to avoid misleading downstream consumers.
    """
    raw = load_jsonl(path)
    bug_fixed = 0
    out: list[dict] = []
    for r in raw:
        url = r.get("url") or ""
        ar_url_raw = r.get("ar_url") or ""

        # Detect the bug: ar_url was set to the same English URL
        if ar_url_raw and ar_url_raw == url:
            ar_url_fixed = ""
            bug_fixed += 1
        else:
            ar_url_fixed = ar_url_raw

        out.append(normalise({
            "text": (r.get("text") or "").strip(),
            "answer": "",
            "title": (r.get("title") or "").strip(),
            "url": url,
            "category": (r.get("category") or "").strip(),
            "language": "en",
            "phase": "phase_2",
            # rec_id in phase2-en holds the article title (not a numeric id)
            "rec_id": _to_str(r.get("rec_id")),
            "qa_id": -1,
            "train_phase_2_id": -1,
            "ar_url": ar_url_fixed,
            "ar_title": "",
        }))
    return out, {"input": len(raw), "kept": len(out), "ar_url_bug_fixed": bug_fixed}


def process_en_finetune(path: Path) -> tuple[list[dict], dict]:
    """
    Normalise the English finetune JSONL (already clean from step06).
    """
    raw = load_jsonl(path)
    out: list[dict] = []
    for r in raw:
        out.append(normalise({
            "text": (r.get("text") or "").strip(),
            "answer": (r.get("answer") or "").strip(),
            "title": (r.get("title") or "").strip(),
            "url": r.get("url") or "",
            "category": (r.get("category") or "").strip(),
            "language": "en",
            "phase": "finetune_qa",
            "rec_id": _to_str(r.get("rec_id")),
            "qa_id": _to_int(r.get("qa_id")),
            "train_phase_2_id": _to_int(r.get("train_phase_2_id")),
            "ar_url": "",
            "ar_title": "",
        }))
    return out, {"input": len(raw), "kept": len(out)}


def process_aug_jsonl(path: Path, phase: str) -> tuple[list[dict], dict]:
    """
    Process an augmentation JSONL file produced by step07 (any language).

    The ``language`` field is read from each record so both EN and AR aug
    files are handled identically by this single function.  The ``phase``
    argument pins the phase label ("phase_1" or "phase_2") for the whole
    file, consistent with the output file name convention used by step07.
    """
    raw = load_jsonl(path)
    out: list[dict] = []
    for r in raw:
        lang = (r.get("language") or "en").strip()
        out.append(normalise({
            "text":             (r.get("text") or "").strip(),
            "answer":           "",
            "title":            (r.get("title") or "").strip(),
            "url":              r.get("url") or "",
            "category":         (r.get("category") or "").strip(),
            "language":         lang,
            "phase":            phase,
            "rec_id":           _to_str(r.get("rec_id")),
            "qa_id":            -1,
            "train_phase_2_id": -1,
            "ar_url":           r.get("ar_url") or "",
            "ar_title":         (r.get("ar_title") or "").strip(),
        }))
    return out, {"input": len(raw), "kept": len(out)}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_stats(label: str, stats: dict) -> None:
    kept = stats.get("kept", "?")
    total = stats.get("input", "?")
    print(f"  {label}: {kept:,} / {total:,} kept", end="")
    extras = {k: v for k, v in stats.items() if k not in ("input", "kept") and v}
    if extras:
        details = ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                            for k, v in extras.items())
        print(f"  [{details}]", end="")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "wiki_data_collecting" / "data",
        help="Directory containing the JSONL files from steps 05 & 06 "
             "(default: ./wiki_data_collecting/data)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "wiki_data_collecting" / "data" / "clean_build",
        help="Directory to write the cleaned JSONL files "
             "(default: ./wiki_data_collecting/data/clean_build)",
    )
    p.add_argument(
        "--hf-source",
        default="bakrianoo/jabarti",
        help="HuggingFace dataset id to load as the Arabic base "
             "(default: bakrianoo/jabarti)",
    )
    p.add_argument(
        "--hub-repo",
        default=None,
        help="HuggingFace Hub repository to push the final dataset to "
             "(e.g. 'bakrianoo/jabarti-bilingual').  Required unless --no-upload.",
    )
    p.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Skip the Hub upload step; only save locally.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create the Hub repository as private.",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum text length to keep (default: {DEFAULT_MIN_CHARS})",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Hard text length cap; longer texts are truncated at the last "
             f"paragraph break (default: {DEFAULT_MAX_CHARS})",
    )
    p.add_argument(
        "--keep-stubs",
        action="store_true",
        default=False,
        help="Keep stub articles (category contains 'بذرة'); dropped by default.",
    )
    p.add_argument(
        "--phase1-en-aug",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional augmentation JSONL for phase_1 produced by step07 "
             "(e.g. data/phase1-en-aug.jsonl).  Records are merged into the "
             "train_phase_1 split after the main phase1-en.jsonl is processed.",
    )
    p.add_argument(
        "--phase2-en-aug",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional augmentation JSONL for phase_2 produced by step07 "
             "(e.g. data/phase2-en-aug.jsonl).  Records are merged into the "
             "train_phase_2 split after the main phase2-en.jsonl is processed.",
    )
    p.add_argument(
        "--phase1-ar-aug",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional Arabic augmentation JSONL for phase_1 produced by step07 "
             "(e.g. data/phase1-ar-aug.jsonl).  Records are merged into the "
             "train_phase_1 split.",
    )
    p.add_argument(
        "--phase2-ar-aug",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional Arabic augmentation JSONL for phase_2 produced by step07 "
             "(e.g. data/phase2-ar-aug.jsonl).  Records are merged into the "
             "train_phase_2 split.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.no_upload and not args.hub_repo:
        print(
            "error: provide --hub-repo <owner/repo> or pass --no-upload.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- 1. Locate input files -----------------------------------------------
    data_dir = args.data_dir
    p1_en_path = data_dir / "phase1-en.jsonl"
    p2_en_path = data_dir / "phase2-en.jsonl"
    ft_en_path = data_dir / "finetune-en.jsonl"

    for path in (p1_en_path, p2_en_path, ft_en_path):
        if not path.exists():
            print(f"error: required input file not found: {path}", file=sys.stderr)
            sys.exit(1)

    # -- 2. Load HF dataset ---------------------------------------------------
    print(f"\nLoading HF dataset '{args.hf_source}' …")
    try:
        from datasets import load_dataset, Dataset, DatasetDict, Features, Value
    except ImportError:
        print("error: 'datasets' package is required.  pip install datasets",
              file=sys.stderr)
        sys.exit(1)

    ds = load_dataset(args.hf_source)
    print(f"  Loaded splits: {list(ds.keys())}")

    # -- 3. Process each source -----------------------------------------------
    print("\nCleaning & normalising …")

    ar_p1, stats_ar_p1 = process_ar_phase1(
        ds["train_phase_1"],
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        filter_stubs=not args.keep_stubs,
    )
    print_stats("AR train_phase_1", stats_ar_p1)

    ar_p2, stats_ar_p2 = process_ar_phase2(ds["train_phase_2"])
    print_stats("AR train_phase_2", stats_ar_p2)

    ar_ft, stats_ar_ft = process_ar_finetune(ds["finetune_qa"])
    print_stats("AR finetune_qa  ", stats_ar_ft)

    en_p1, stats_en_p1 = process_en_phase1(p1_en_path)
    print_stats("EN phase1-en    ", stats_en_p1)

    en_p2, stats_en_p2 = process_en_phase2(p2_en_path)
    print_stats("EN phase2-en    ", stats_en_p2)

    en_ft, stats_en_ft = process_en_finetune(ft_en_path)
    print_stats("EN finetune-en  ", stats_en_ft)

    # -- 3b. Optional augmentation files from step07 -------------------------
    if args.phase1_en_aug:
        if args.phase1_en_aug.exists():
            en_p1_aug, stats_en_p1_aug = process_en_phase1(args.phase1_en_aug)
            print_stats("EN phase1-en-aug", stats_en_p1_aug)
            en_p1 = en_p1 + en_p1_aug
        else:
            print(
                f"  [warn] --phase1-en-aug not found: {args.phase1_en_aug}",
                file=sys.stderr,
            )

    if args.phase2_en_aug:
        if args.phase2_en_aug.exists():
            en_p2_aug, stats_en_p2_aug = process_en_phase2(args.phase2_en_aug)
            print_stats("EN phase2-en-aug", stats_en_p2_aug)
            en_p2 = en_p2 + en_p2_aug
        else:
            print(
                f"  [warn] --phase2-en-aug not found: {args.phase2_en_aug}",
                file=sys.stderr,
            )

    # -- 3c. Optional Arabic augmentation files from step07 -----------------
    if args.phase1_ar_aug:
        if args.phase1_ar_aug.exists():
            ar_p1_aug, stats_ar_p1_aug = process_aug_jsonl(args.phase1_ar_aug, "phase_1")
            print_stats("AR phase1-ar-aug", stats_ar_p1_aug)
            ar_p1 = ar_p1 + ar_p1_aug
        else:
            print(
                f"  [warn] --phase1-ar-aug not found: {args.phase1_ar_aug}",
                file=sys.stderr,
            )

    if args.phase2_ar_aug:
        if args.phase2_ar_aug.exists():
            ar_p2_aug, stats_ar_p2_aug = process_aug_jsonl(args.phase2_ar_aug, "phase_2")
            print_stats("AR phase2-ar-aug", stats_ar_p2_aug)
            ar_p2 = ar_p2 + ar_p2_aug
        else:
            print(
                f"  [warn] --phase2-ar-aug not found: {args.phase2_ar_aug}",
                file=sys.stderr,
            )

    # -- 4. Merge into final splits -------------------------------------------
    print("\nMerging splits …")

    split_phase1 = ar_p1 + en_p1
    split_phase2 = ar_p2 + en_p2
    split_finetune = ar_ft + en_ft

    print(f"  train_phase_1 : {len(ar_p1):>7,} AR  +  {len(en_p1):>6,} EN  =  {len(split_phase1):>7,} total")
    print(f"  train_phase_2 : {len(ar_p2):>7,} AR  +  {len(en_p2):>6,} EN  =  {len(split_phase2):>7,} total")
    print(f"  finetune_qa   : {len(ar_ft):>7,} AR  +  {len(en_ft):>6,} EN  =  {len(split_finetune):>7,} total")

    # -- 5. Sanity checks -----------------------------------------------------
    print("\nRunning sanity checks …")
    errors: list[str] = []

    for split_name, records in [
        ("train_phase_1", split_phase1),
        ("train_phase_2", split_phase2),
        ("finetune_qa", split_finetune),
    ]:
        empty_text = sum(1 for r in records if not r["text"].strip())
        wrong_fields = [r for r in records if set(r.keys()) != set(_SCHEMA_FIELDS)]
        schema_ok = len(wrong_fields) == 0
        langs = set(r["language"] for r in records)

        print(f"  {split_name}: {len(records):,} rows, langs={sorted(langs)}, "
              f"empty_text={empty_text}, schema_ok={schema_ok}")

        if empty_text:
            errors.append(f"{split_name}: {empty_text} rows have empty text")
        if not schema_ok:
            errors.append(f"{split_name}: {len(wrong_fields)} rows have wrong schema fields")
        if langs - {"ar", "en"}:
            errors.append(f"{split_name}: unexpected language values: {langs - {'ar','en'}}")

    if errors:
        print("\n  WARNINGS:", file=sys.stderr)
        for e in errors:
            print(f"    - {e}", file=sys.stderr)
    else:
        print("  All checks passed.")

    # -- 6. Write local JSONL files -------------------------------------------
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting cleaned JSONL files to {out_dir} …")

    write_jsonl(split_phase1, out_dir / "train_phase_1.jsonl")
    write_jsonl(split_phase2, out_dir / "train_phase_2.jsonl")
    write_jsonl(split_finetune, out_dir / "finetune_qa.jsonl")

    # Write per-source stats
    all_stats = {
        "ar_phase_1": stats_ar_p1,
        "ar_phase_2": stats_ar_p2,
        "ar_finetune": stats_ar_ft,
        "en_phase_1": stats_en_p1,
        "en_phase_2": stats_en_p2,
        "en_finetune": stats_en_ft,
        "merged": {
            "train_phase_1": len(split_phase1),
            "train_phase_2": len(split_phase2),
            "finetune_qa": len(split_finetune),
        },
    }
    stats_path = out_dir / "_build_stats.json"
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, ensure_ascii=False, indent=2)
    print(f"  Stats written to {stats_path}")

    # -- 7. Build HuggingFace DatasetDict -------------------------------------
    print("\nBuilding HuggingFace DatasetDict …")

    features = Features({
        "text":             Value("string"),
        "answer":           Value("string"),
        "title":            Value("string"),
        "url":              Value("string"),
        "category":         Value("string"),
        "language":         Value("string"),
        "phase":            Value("string"),
        "rec_id":           Value("string"),
        "qa_id":            Value("int32"),
        "train_phase_2_id": Value("int32"),
        "ar_url":           Value("string"),
        "ar_title":         Value("string"),
    })

    def records_to_dataset(records: list[dict]) -> Dataset:
        columns: dict[str, list] = {f: [] for f in _SCHEMA_FIELDS}
        for rec in records:
            for f in _SCHEMA_FIELDS:
                columns[f].append(rec[f])
        return Dataset.from_dict(columns, features=features)

    dataset_dict = DatasetDict({
        "train_phase_1": records_to_dataset(split_phase1),
        "train_phase_2": records_to_dataset(split_phase2),
        "finetune_qa":   records_to_dataset(split_finetune),
    })
    print(f"  {dataset_dict}")

    # -- 8. Upload to Hub (optional) ------------------------------------------
    if args.no_upload:
        print("\nSkipping Hub upload (--no-upload).")
        print(f"\nDone.  Local files are in {out_dir}")
        return

    print(f"\nPushing to Hub: {args.hub_repo} …")
    dataset_dict.push_to_hub(
        repo_id=args.hub_repo,
        private=args.private,
        commit_message=(
            "Clean & bilingual build: filtered AR phase_1, "
            "merged EN extensions (step05/06), unified schema"
        ),
    )
    print(f"  Uploaded successfully → https://huggingface.co/datasets/{args.hub_repo}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
