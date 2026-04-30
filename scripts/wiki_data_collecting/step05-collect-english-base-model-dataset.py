# pip install requests mwparserfromhell tqdm

"""
Step 05 — Collect the English-language counterpart of the base-model corpus
==========================================================================

This step extends the (currently Arabic-only) jabarti base-model corpus with
domain-matched English Wikipedia content so the LLM can be trained as a
genuine bilingual model that exhibits cross-lingual transfer on the
Egyptian / MENA history domain.

Two phases are produced (the same two phases the base model already uses):

  - phase_1 (en):
        Source = ./data/articles-urls.jsonl  (Arabic Wikipedia URLs from step01)
        Action = resolve each Arabic article's English equivalent via the
                 MediaWiki "langlinks" API; fetch the English wikitext;
                 clean and filter.
        Output = ./data/phase1-en.jsonl

  - phase_2 (en):
        Source = ./data/egyptian_figures_data.jsonl  (English Wikipedia URLs
                 from step03 — these are exactly the entities GPT used to
                 generate the Arabic phase_2 articles in step04, so the
                 English side is conceptually parallel at the entity level).
        Action = fetch the full English wikitext; clean and filter.
        Output = ./data/phase2-en.jsonl

Output schema per JSONL line (matches the HF dataset layout used by jabarti):

    {
        "text":              <cleaned English article body>,
        "title":             <English page title>,
        "url":               <English Wikipedia URL>,
        "category":          <category from the source jsonl>,
        "language":          "en",
        "phase":             "phase_1" | "phase_2",
        "ar_url":            <linked Arabic URL, when known>,
        "ar_title":          <linked Arabic title, when known>,
        "rec_id":            <stable id from source jsonl, if any>,
        "char_count":        <int>,
        "word_count":        <int>
    }

The script is fully resumable (skips URLs already in the output file),
multi-threaded with a polite rate limit, and respects the Wikimedia
User-Agent policy.

Usage
-----
    # phase_1: Arabic URLs → English versions via langlinks
    python step05-collect-english-base-model-dataset.py \
        --phase phase_1 \
        --input  ./data/articles-urls.jsonl \
        --output ./data/phase1-en.jsonl

    # phase_2: re-fetch full English wikitext for the figures from step03
    python step05-collect-english-base-model-dataset.py \
        --phase phase_2 \
        --input  ./data/egyptian_figures_data.jsonl \
        --output ./data/phase2-en.jsonl

    # both, sequentially, with defaults
    python step05-collect-english-base-model-dataset.py --phase all
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import mwparserfromhell
import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AR_API_URL = "https://ar.wikipedia.org/w/api.php"
EN_API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia requires a descriptive UA with contact info.
USER_AGENT = (
    "JabartiBilingualCollector/1.0 "
    "(https://github.com/bakrianoo/jabarti-llm-from-scratch; "
    "educational NLP course project) "
    "python-requests"
)

# Tail sections we drop from English Wikipedia articles (boilerplate at
# the end that hurts language modelling more than it helps).
EN_TAIL_SECTIONS = {
    "see also",
    "references",
    "notes",
    "footnotes",
    "citations",
    "further reading",
    "external links",
    "bibliography",
    "sources",
    "works cited",
    "general references",
    "general and cited references",
}

# Quality thresholds for the cleaned English text. Tuned to roughly mirror
# the phase_2 length distribution (median ~1.2k chars) while still keeping
# longer phase_1-style articles, but capping pathological cases.
MIN_CHARS = 250
MIN_WORDS = 50
MAX_CHARS = 20_000          # hard cap; longer articles are truncated cleanly
ASCII_LETTER_RATIO_MIN = 0.60   # cheap English language ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    return s


class RateLimiter:
    """Thread-safe minimum-interval limiter shared across workers.

    Wikimedia asks bots to keep the global request rate modest. We enforce a
    *single* monotonic schedule so that N workers in parallel never exceed
    1/min_interval requests per second in aggregate. We also support a
    cooldown period that all threads honor after a 429.
    """

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self.min_interval

    def cool_down(self, seconds: float):
        """Push the next-allowed timestamp forward for everyone."""
        if seconds <= 0:
            return
        with self._lock:
            target = time.monotonic() + seconds
            if target > self._next_allowed:
                self._next_allowed = target


# Global limiter; configured in main() based on --rps.
LIMITER: "RateLimiter" = RateLimiter(0.25)


def api_get(session: requests.Session,
            url: str,
            params: Dict,
            *,
            max_retries: int = 6,
            timeout: int = 30) -> Optional[Dict]:
    """GET a MediaWiki API endpoint with shared rate limiting and retries.

    Handles HTTP 429 (honors Retry-After), 5xx, and transient network
    errors with exponential backoff + jitter. Returns the parsed JSON dict
    on success, or None on permanent failure.
    """
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        LIMITER.wait()
        try:
            r = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            sleep_for = backoff + random.uniform(0, 0.5)
            print(f"  [api] network error (attempt {attempt}/{max_retries}): "
                  f"{e}; sleeping {sleep_for:.1f}s")
            LIMITER.cool_down(sleep_for)
            backoff = min(backoff * 2, 60.0)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as e:
                print(f"  [api] bad JSON: {e}")
                return None

        if r.status_code == 429 or 500 <= r.status_code < 600:
            retry_after = r.headers.get("Retry-After")
            try:
                wait_s = float(retry_after) if retry_after else backoff
            except ValueError:
                wait_s = backoff
            wait_s = max(wait_s, backoff) + random.uniform(0, 1.0)
            print(f"  [api] HTTP {r.status_code} (attempt "
                  f"{attempt}/{max_retries}); cooling down {wait_s:.1f}s")
            LIMITER.cool_down(wait_s)
            backoff = min(backoff * 2, 60.0)
            continue

        # Other 4xx: don't retry.
        print(f"  [api] HTTP {r.status_code}; giving up: {r.text[:200]}")
        return None

    print(f"  [api] exhausted retries")
    return None


def title_from_url(url: str) -> Optional[str]:
    """Extract the (URL-decoded) page title from a /wiki/<title> URL."""
    if not url:
        return None
    m = re.search(r"/wiki/(.+)$", url)
    if not m:
        return None
    return unquote(m.group(1)).replace("_", " ").strip()


def normalize_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{unquote(p.path).rstrip('/')}"


def looks_like_english(text: str) -> bool:
    """Cheap language check: are most letters ASCII?"""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return (ascii_letters / len(letters)) >= ASCII_LETTER_RATIO_MIN


# ---------------------------------------------------------------------------
# MediaWiki API wrappers
# ---------------------------------------------------------------------------

def resolve_english_titles(session: requests.Session,
                           ar_titles: List[str]) -> Dict[str, Optional[str]]:
    """
    Given a batch of Arabic page titles, return {ar_title: en_title or None}
    using the langlinks property (lllang=en). Up to 50 titles per call.
    """
    out: Dict[str, Optional[str]] = {t: None for t in ar_titles}
    if not ar_titles:
        return out

    for i in range(0, len(ar_titles), 50):
        chunk = ar_titles[i:i + 50]
        params = {
            "action": "query",
            "format": "json",
            "prop": "langlinks",
            "lllang": "en",
            "lllimit": "max",
            "redirects": 1,
            "titles": "|".join(chunk),
        }
        data = api_get(session, AR_API_URL, params)
        if data is None:
            print(f"  [langlinks] batch failed permanently")
            continue

        # MediaWiki may rewrite titles via "normalized" / "redirects".
        rewrites: Dict[str, str] = {}
        for entry in data.get("query", {}).get("normalized", []) or []:
            rewrites[entry["to"]] = entry["from"]
        for entry in data.get("query", {}).get("redirects", []) or []:
            rewrites[entry["to"]] = rewrites.get(entry["from"], entry["from"])

        for page in data.get("query", {}).get("pages", {}).values():
            final_title = page.get("title")
            requested = rewrites.get(final_title, final_title)
            ll = page.get("langlinks") or []
            if ll and isinstance(ll, list):
                en_title = ll[0].get("*") or ll[0].get("title")
                if requested in out:
                    out[requested] = en_title
                # also write under the resolved title in case the request
                # used a redirect form
                out[final_title] = en_title

    return out


def fetch_en_wikitext(session: requests.Session,
                      title: str) -> Optional[str]:
    """Fetch the raw wikitext of an English Wikipedia page."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "redirects": 1,
    }
    data = api_get(session, EN_API_URL, params)
    if data is None:
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            return None
        revs = page.get("revisions") or []
        if revs:
            return revs[0]["slots"]["main"]["*"]
    return None


# ---------------------------------------------------------------------------
# English-specific cleaning
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
_REF_OPEN_RE = re.compile(r"<ref[^>]*?>.*?</ref>", flags=re.DOTALL | re.IGNORECASE)
_REF_SELF_RE = re.compile(r"<ref[^/]*?/>", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FILE_LINK_RE = re.compile(
    r"\[\[(File|Image|Category):[^\[\]]*?(?:\[\[[^\]]*\]\][^\[\]]*?)*\]\]",
    flags=re.IGNORECASE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _split_into_sections(wikitext: str) -> List[Tuple[str, str]]:
    """
    Split wikitext into (heading, body) sections. The lead (before the first
    heading) is returned with heading == "".
    """
    sections: List[Tuple[str, str]] = []
    current_heading = ""
    current_body: List[str] = []
    for line in wikitext.split("\n"):
        m = _HEADING_RE.match(line.strip())
        if m:
            sections.append((current_heading, "\n".join(current_body)))
            current_heading = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    sections.append((current_heading, "\n".join(current_body)))
    return sections


def clean_english_wikitext(wikitext: str) -> str:
    """
    Convert raw English wikitext to clean plain text suitable for LM
    pretraining:
      * drop tail sections (See also, References, External links, ...)
      * strip <ref>...</ref>, HTML tags, file/category links
      * use mwparserfromhell to strip the remaining wiki markup
      * keep section headings as their bare text (one per line)
      * collapse excessive blank lines
    """
    if not wikitext:
        return ""

    cleaned_sections: List[str] = []
    for heading, body in _split_into_sections(wikitext):
        if heading.lower().strip() in EN_TAIL_SECTIONS:
            continue

        body = _REF_OPEN_RE.sub("", body)
        body = _REF_SELF_RE.sub("", body)
        body = _FILE_LINK_RE.sub("", body)

        try:
            stripped = mwparserfromhell.parse(body).strip_code(
                normalize=True, collapse=True
            )
        except Exception:
            stripped = body

        stripped = _HTML_TAG_RE.sub("", stripped)
        stripped = stripped.strip()

        if heading and stripped:
            cleaned_sections.append(f"{heading}\n{stripped}")
        elif stripped:
            cleaned_sections.append(stripped)

    text = "\n\n".join(cleaned_sections)
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()
    return text


def quality_ok(text: str) -> Tuple[bool, str]:
    if len(text) < MIN_CHARS:
        return False, "too_short_chars"
    if len(text.split()) < MIN_WORDS:
        return False, "too_short_words"
    if not looks_like_english(text):
        return False, "not_english"
    # Drop disambiguation pages / pure list pages: very low sentence count.
    if text.count(".") < 5:
        return False, "too_few_sentences"
    return True, "ok"


def truncate_clean(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate at the nearest paragraph boundary <= max_chars."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut == -1 or cut < max_chars // 2:
        cut = text.rfind(". ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].rstrip()


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_done_urls(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("url"):
                done.add(normalize_url(rec["url"]))
    return done


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_phase1_inputs(input_path: Path) -> List[Dict]:
    """
    Load step01 output and de-dupe by URL. Returns rows with keys:
      ar_url, ar_title, category, parent_categories
    """
    seen = set()
    rows: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            url = rec.get("url")
            if not url:
                continue
            n = normalize_url(url)
            if n in seen:
                continue
            seen.add(n)

            # Skip non-article namespaces (templates, portals, ...) — these
            # were the empty-text rows we already saw in the HF dataset.
            ar_title = rec.get("title") or title_from_url(url) or ""
            if any(ar_title.startswith(p) for p in
                   ("قالب:", "بوابة:", "تصنيف:", "ملف:", "مستخدم:")):
                continue

            rows.append({
                "ar_url": url,
                "ar_title": ar_title,
                "category": rec.get("category", ""),
                "parent_categories": rec.get("parent_categories", []) or [],
            })
    return rows


def load_phase2_inputs(input_path: Path) -> List[Dict]:
    """
    Load step03 figures and de-dupe by URL. Returns rows with keys:
      en_url, en_title, category, name
    """
    seen = set()
    rows: List[Dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            url = rec.get("url")
            if not url or "en.wikipedia.org" not in url:
                continue
            n = normalize_url(url)
            if n in seen:
                continue
            seen.add(n)
            rows.append({
                "en_url": url,
                "en_title": rec.get("title") or title_from_url(url) or "",
                "category": rec.get("category", ""),
                "name": rec.get("name", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

def write_record(out_f, lock: Lock, record: Dict):
    with lock:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()


def process_phase1(input_path: Path,
                   output_path: Path,
                   workers: int,
                   max_records: Optional[int]):
    rows = load_phase1_inputs(input_path)
    print(f"Loaded {len(rows):,} unique Arabic articles from {input_path}")

    done_urls = load_done_urls(output_path)
    print(f"Already collected: {len(done_urls):,} English articles "
          f"(resume mode)")

    # We don't know the English URL until langlinks resolves it, so we need
    # a separate "tried" log to avoid re-resolving titles that previously
    # had no English equivalent. We piggyback on a sibling .missing file.
    missing_path = output_path.with_suffix(output_path.suffix + ".missing")
    missing_titles: set = set()
    if missing_path.exists():
        with open(missing_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    missing_titles.add(line)
        print(f"Previously-missing langlinks: {len(missing_titles):,}")

    # Filter rows whose English URL is already saved or known-missing.
    # English URL isn't known yet, so we only filter known-missing here;
    # the saved-URL check happens after fetch.
    rows = [r for r in rows if r["ar_title"] not in missing_titles]
    if max_records:
        rows = rows[:max_records]

    print(f"To process: {len(rows):,}")

    session = make_session()
    write_lock = Lock()
    missing_lock = Lock()
    counters = {"saved": 0, "no_langlink": 0, "fetch_failed": 0,
                "filtered": 0, "dup": 0}

    # ----- Step A: resolve langlinks in batches of 50 (sequential, cheap) -
    print("Resolving English equivalents via langlinks ...")
    title_to_row: Dict[str, Dict] = {r["ar_title"]: r for r in rows
                                     if r["ar_title"]}
    ar_titles = list(title_to_row.keys())

    resolved: Dict[str, Optional[str]] = {}
    with tqdm(total=len(ar_titles), desc="langlinks", unit="t") as pbar:
        for i in range(0, len(ar_titles), 50):
            chunk = ar_titles[i:i + 50]
            mapping = resolve_english_titles(session, chunk)
            resolved.update(mapping)
            pbar.update(len(chunk))

    # Persist newly-missing titles for next resume.
    new_missing = [t for t, en in resolved.items() if not en]
    if new_missing:
        with open(missing_path, "a", encoding="utf-8") as mf:
            for t in new_missing:
                mf.write(t + "\n")
    counters["no_langlink"] = len(new_missing)
    print(f"  resolved: {sum(1 for v in resolved.values() if v):,}, "
          f"missing: {len(new_missing):,}")

    # ----- Step B: fetch + clean in parallel ------------------------------
    todo = []
    for ar_title, en_title in resolved.items():
        if not en_title:
            continue
        # Some entries in `resolved` correspond to MediaWiki-resolved
        # (post-redirect) titles that are not in our request map; skip them.
        row = title_to_row.get(ar_title)
        if row is None:
            continue
        en_url = "https://en.wikipedia.org/wiki/" + en_title.replace(" ", "_")
        if normalize_url(en_url) in done_urls:
            counters["dup"] += 1
            continue
        todo.append((row, en_title, en_url))

    print(f"Fetching {len(todo):,} English articles ...")

    with open(output_path, "a", encoding="utf-8") as out_f:

        def _job(row_en_title_en_url):
            row, en_title, en_url = row_en_title_en_url
            wt = fetch_en_wikitext(session, en_title)
            if not wt:
                return ("fetch_failed", None)
            cleaned = clean_english_wikitext(wt)
            cleaned = truncate_clean(cleaned)
            ok, reason = quality_ok(cleaned)
            if not ok:
                return ("filtered", reason)
            record = {
                "text": cleaned,
                "title": en_title,
                "url": en_url,
                "category": row["category"],
                "language": "en",
                "phase": "phase_1",
                "ar_url": row["ar_url"],
                "ar_title": row["ar_title"],
                "rec_id": "",
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
            }
            write_record(out_f, write_lock, record)
            return ("saved", None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_job, item): item for item in todo}
            with tqdm(total=len(todo), desc="phase_1 en", unit="art") as pbar:
                for fut in as_completed(futures):
                    status, _reason = fut.result()
                    counters[status] = counters.get(status, 0) + 1
                    pbar.update(1)
                    pbar.set_postfix({k: v for k, v in counters.items()
                                      if k in ("saved", "filtered",
                                              "fetch_failed", "dup")})

    print(f"\n[phase_1 en] {counters}")
    print(f"Saved to: {output_path}")


def process_phase2(input_path: Path,
                   output_path: Path,
                   workers: int,
                   max_records: Optional[int]):
    rows = load_phase2_inputs(input_path)
    print(f"Loaded {len(rows):,} English-Wikipedia figures from {input_path}")

    done_urls = load_done_urls(output_path)
    print(f"Already collected: {len(done_urls):,} (resume mode)")

    rows = [r for r in rows
            if normalize_url(r["en_url"]) not in done_urls]
    if max_records:
        rows = rows[:max_records]
    print(f"To process: {len(rows):,}")

    session = make_session()
    write_lock = Lock()
    counters = {"saved": 0, "fetch_failed": 0, "filtered": 0}

    with open(output_path, "a", encoding="utf-8") as out_f:

        def _job(row):
            title = row["en_title"] or title_from_url(row["en_url"])
            if not title:
                return ("fetch_failed", None)
            wt = fetch_en_wikitext(session, title)
            if not wt:
                return ("fetch_failed", None)
            cleaned = clean_english_wikitext(wt)
            cleaned = truncate_clean(cleaned)
            ok, reason = quality_ok(cleaned)
            if not ok:
                return ("filtered", reason)
            record = {
                "text": cleaned,
                "title": title,
                "url": row["en_url"],
                "category": row["category"],
                "language": "en",
                "phase": "phase_2",
                # phase_2 source is already English; the Arabic counterpart
                # is the LLM-generated article in llm_output_dataset.jsonl,
                # which is keyed by the same URL.
                "ar_url": row["en_url"],
                "ar_title": "",
                "rec_id": row.get("name", ""),
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
            }
            write_record(out_f, write_lock, record)
            return ("saved", None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_job, r): r for r in rows}
            with tqdm(total=len(rows), desc="phase_2 en", unit="art") as pbar:
                for fut in as_completed(futures):
                    status, _reason = fut.result()
                    counters[status] = counters.get(status, 0) + 1
                    pbar.update(1)
                    pbar.set_postfix(counters)

    print(f"\n[phase_2 en] {counters}")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Collect English Wikipedia counterpart for jabarti "
                    "phase_1 and phase_2 (base-model corpus).")
    p.add_argument("--phase", choices=["phase_1", "phase_2", "all"],
                   default="all")
    p.add_argument("--input", type=str, default=None,
                   help="Override input path. Defaults: "
                        "phase_1=./data/articles-urls.jsonl, "
                        "phase_2=./data/egyptian_figures_data.jsonl")
    p.add_argument("--output", type=str, default=None,
                   help="Override output path. Defaults: "
                        "phase_1=./data/phase1-en.jsonl, "
                        "phase_2=./data/phase2-en.jsonl")
    p.add_argument("--workers", type=int, default=2,
                   help="Concurrent fetch workers. Combined with --rps, the "
                        "global request rate is capped, so this mainly hides "
                        "network latency.")
    p.add_argument("--rps", type=float, default=4.0,
                   help="Maximum global requests per second across ALL "
                        "workers. Wikimedia tolerates a few req/s for a "
                        "well-identified bot; default 4 is safe. Lower this "
                        "if you see HTTP 429.")
    p.add_argument("--max-records", type=int, default=None,
                   help="Cap input rows for a smoke test.")
    args = p.parse_args()

    # Configure the global rate limiter.
    global LIMITER
    interval = 1.0 / max(args.rps, 0.1)
    LIMITER = RateLimiter(interval)
    print(f"Global rate limit: {args.rps:.2f} req/s "
          f"(min interval {interval*1000:.0f} ms)")

    base_dir = Path(__file__).resolve().parent / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("phase_1", "all"):
        in_p = Path(args.input) if (args.input and args.phase == "phase_1") \
            else base_dir / "articles-urls.jsonl"
        out_p = Path(args.output) if (args.output and args.phase == "phase_1") \
            else base_dir / "phase1-en.jsonl"
        if not in_p.exists():
            print(f"[phase_1] input not found: {in_p} — skipping")
        else:
            print(f"\n=== phase_1 (en) ===")
            process_phase1(in_p, out_p, args.workers, args.max_records)

    if args.phase in ("phase_2", "all"):
        in_p = Path(args.input) if (args.input and args.phase == "phase_2") \
            else base_dir / "egyptian_figures_data.jsonl"
        out_p = Path(args.output) if (args.output and args.phase == "phase_2") \
            else base_dir / "phase2-en.jsonl"
        if not in_p.exists():
            print(f"[phase_2] input not found: {in_p} — skipping")
        else:
            print(f"\n=== phase_2 (en) ===")
            process_phase2(in_p, out_p, args.workers, args.max_records)


if __name__ == "__main__":
    main()
