#!/usr/bin/env python3
"""
Step 07 — Augment the English and Arabic dataset (phases 1 & 2)
================================================================

Four complementary, domain-bounded sources of new training records:

  Source A/B — Figure summary recovery  (→ phase_2)
  ---------------------------------------------------
  For every entry in ``egyptian_figures_data.jsonl`` that step05 did *not*
  successfully fetch (stub articles, disambiguation pages, etc.):

    A) Use the ``summary`` field already stored in the source file when it
       is long enough  →  zero extra API calls.

    B) Call the Wikimedia REST ``/api/rest_v1/page/summary/<title>``
       endpoint for entries whose stored summary is too short.  The REST
       call is automatically routed to ``en.`` or ``ar.`` based on the
       URL host of the figure record.

  Source C — EN Wikipedia category-tree walk  (→ phase_1 by default)
  --------------------------------------------------------------------
  Recursively enumerate article pages under one or more seed EN Wikipedia
  categories that define the target domain, then bulk-fetch their intro
  sections via ``prop=extracts&exintro=1&explaintext=1``  (plain text, no
  wikitext parsing).  Only URLs not already present in the existing phase
  files are kept.

  Default seed categories (Egypt / MENA domain):
    History of Egypt, Ancient Egypt, Pharaohs, Islamic Egypt,
    Egyptian culture, Egypt, Nile, Cairo, Egyptian people,
    Egyptian diaspora

  For phase_2-style biographical content use:
    --categories "Egyptian people" "Egyptian diaspora" \\
    --category-phase phase_2

  Source D — AR Wikipedia category-tree walk  (→ phase_1 by default)
  -------------------------------------------------------------------
  Same BFS + extracts approach as Source C but on Arabic Wikipedia.
  Deduplicates against ``articles-urls.jsonl`` (the full phase_1 AR source
  list) so only genuinely new Arabic articles are written.  Arabic namespace
  meta-pages (قالب، بوابة، ملف …) are filtered out automatically.

  Default Arabic seed categories:
    تاريخ مصر, مصر القديمة, الفراعنة, ثقافة مصر, القاهرة,
    نهر النيل, الحضارة المصرية القديمة, شخصيات مصرية, محافظات مصر

  For phase_2-style biographical AR content use:
    --ar-categories "شخصيات مصرية" "مصريون في الخارج" \\
    --ar-category-phase phase_2

Output files (append-mode, fully resumable)
-------------------------------------------
  data/phase1-en-aug.jsonl   — new phase_1 English records
  data/phase2-en-aug.jsonl   — new phase_2 English records
  data/phase1-ar-aug.jsonl   — new phase_1 Arabic records
  data/phase2-ar-aug.jsonl   — new phase_2 Arabic records
  data/_aug_stats.json       — per-source counters from the latest run

Output schema per record  (identical to step05 so that
``clean_and_build_dataset.py`` can consume all four files via
``--phase1-en-aug`` / ``--phase2-en-aug`` / ``--phase1-ar-aug`` /
``--phase2-ar-aug`` without any conversion):

    {
        "text":       <cleaned plain-text intro / article body>,
        "title":      <page title>,
        "url":        <https://<lang>.wikipedia.org/wiki/…>,
        "category":   <seed / walk category that led to this page>,
        "language":   "en" | "ar",
        "phase":      "phase_1" | "phase_2",
        "ar_url":     "",
        "ar_title":   "",
        "rec_id":     "",
        "char_count": <int>,
        "word_count": <int>
    }

Usage
-----
    # Quick smoke test — discover up to 30 candidate records per source
    # (final count after quality filtering may be lower)
    python step07-augment-english-dataset.py \\
        --limit 30 \\
        --output-dir /tmp/aug_test

    # Source A/B only (figure summary recovery → phase_2)
    python step07-augment-english-dataset.py --source figures

    # Source C only — custom domain categories → phase_1
    python step07-augment-english-dataset.py \\
        --source categories \\
        --categories "History of Egypt" "Ancient Egypt" "Pharaohs" \\
        --max-depth 2

    # Full run with 4 parallel workers at 2 req/s
    python step07-augment-english-dataset.py \\
        --source all \\
        --workers 4 \\
        --rps 2.0

    # Source D only — Arabic Wikipedia category walk → phase_1
    python step07-augment-english-dataset.py \\
        --source ar-categories \\
        --ar-categories "تاريخ مصر" "مصر القديمة" "الفراعنة" \\
        --max-depth 2

    # After running, rebuild the full dataset with all augmentation
    python ../clean_and_build_dataset.py \\
        --no-upload \\
        --phase1-en-aug ./data/phase1-en-aug.jsonl \\
        --phase2-en-aug ./data/phase2-en-aug.jsonl \\
        --phase1-ar-aug ./data/phase1-ar-aug.jsonl \\
        --phase2-ar-aug ./data/phase2-ar-aug.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import random
import urllib.parse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths and configuration defaults
# ---------------------------------------------------------------------------

_SCRIPT_DIR  = Path(__file__).parent
_DATA_DIR    = _SCRIPT_DIR / "data"

FIGURES_SOURCE  = _DATA_DIR / "egyptian_figures_data.jsonl"
PHASE1_EXISTING = _DATA_DIR / "phase1-en.jsonl"
PHASE2_EXISTING = _DATA_DIR / "phase2-en.jsonl"

DEFAULT_PHASE1_AUG    = "phase1-en-aug.jsonl"
DEFAULT_PHASE2_AUG    = "phase2-en-aug.jsonl"
DEFAULT_PHASE1_AR_AUG = "phase1-ar-aug.jsonl"
DEFAULT_PHASE2_AR_AUG = "phase2-ar-aug.jsonl"
STATS_FILE            = "_aug_stats.json"

# Wikipedia meta-namespaces to skip when walking categories (EN)
_NS_PREFIXES = (
    "Talk:", "User:", "User talk:", "Wikipedia:", "Wikipedia talk:",
    "File:", "File talk:", "MediaWiki:", "Template:", "Template talk:",
    "Help:", "Category talk:", "Portal:", "Draft:", "Module:",
)

# Arabic Wikipedia meta-namespace prefixes to skip
_AR_NS_PREFIXES = (
    "نقاش:", "مستخدم:", "مستخدم نقاش:", "ويكيبيديا:", "ويكيبيديا نقاش:",
    "ملف:", "ملف نقاش:", "ميدياويكي:", "ميدياويكي نقاش:", "قالب:",
    "قالب نقاش:", "مساعدة:", "مساعدة نقاش:", "تصنيف نقاش:", "بوابة:",
    "بوابة نقاش:", "مسودة:", "وحدة:", "وحدة نقاش:",
)

# Low-value title patterns (lists, indexes, outlines, disambiguation pages).
# These rarely yield useful prose for LLM training.
_EN_LOW_VALUE_RE = re.compile(
    r"^(List of |Index of |Outline of |Lists of |Timeline of )"
    r"|\(disambiguation\)\s*$",
    re.IGNORECASE,
)
_AR_LOW_VALUE_RE = re.compile(
    r"^(قائمة |قوائم |مسرد |مخطط زمني |جدول زمني )"
    r"|\(توضيح\)\s*$"
)

# Wikimedia requires a descriptive UA with contact info.
USER_AGENT = (
    "JabartiBilingualCollector/1.0 "
    "(https://github.com/bakrianoo/jabarti-llm-from-scratch; "
    "educational NLP course project) "
    "python-requests"
)

# API endpoints
EN_API         = "https://en.wikipedia.org/w/api.php"
AR_API         = "https://ar.wikipedia.org/w/api.php"
EN_REST_BASE   = "https://en.wikipedia.org/api/rest_v1/page/summary"
AR_REST_BASE   = "https://ar.wikipedia.org/api/rest_v1/page/summary"

# MediaWiki maxlag value (server backs off when DB replicas are lagging).
MAXLAG = 5

# Tail sections to strip (safety net; extracts API usually omits them anyway).
_EN_TAIL_RE = re.compile(
    r"\n(?:See also|References|Notes|Footnotes|Further reading|"
    r"External links|Bibliography|Sources)\s*\n.*",
    re.IGNORECASE | re.DOTALL,
)
_AR_TAIL_RE = re.compile(
    r"\n(?:انظر(?:\s+(?:أيضا|أيضًا|أيضاً))?|"
    r"مراجع|المراجع|مصادر|المصادر|"
    r"وصلات خارجية|روابط خارجية|"
    r"اقرأ أيضا|قراءات إضافية|هوامش|ملاحظات)\s*\n.*",
    re.DOTALL,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Quality thresholds
DEFAULT_MIN_CHARS  = 150    # lower than step05 — REST summaries are naturally shorter
DEFAULT_MAX_CHARS  = 20_000
DEFAULT_RPS        = 1.0    # requests per second (shared, polite default)
DEFAULT_WORKERS    = 4
DEFAULT_MAX_DEPTH  = 2
EXTRACT_BATCH_SIZE = 20     # titles per extracts API call (MediaWiki limit: 20 for anon)

DEFAULT_CATEGORIES = [
    "History of Egypt",
    "Ancient Egypt",
    "Pharaohs",
    "Islamic Egypt",
    "Egyptian culture",
    "Egypt",
    "Nile",
    "Cairo",
    "Egyptian people",
    "Egyptian diaspora",
]

# Arabic Wikipedia seed categories (Egypt / MENA domain).
# The walker prepends "تصنيف:" automatically; do not include it here.
DEFAULT_AR_CATEGORIES = [
    "تاريخ مصر",
    "مصر القديمة",
    "الفراعنة",
    "ثقافة مصر",
    "مدن مصر",
    "القاهرة",
    "نهر النيل",
    "الحضارة المصرية القديمة",
    "تاريخ الإسلام في مصر",
    "شخصيات مصرية",
    "محافظات مصر",
    "الأسرات الحاكمة المصرية",
]


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Thread-safe token-bucket rate limiter shared across all workers."""

    def __init__(self, rps: float) -> None:
        self._interval = 1.0 / max(float(rps), 0.01)
        self._lock = Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval

    def cooldown(self, seconds: float) -> None:
        """Push the next-allowed time forward (used after a 429)."""
        with self._lock:
            target = time.monotonic() + seconds
            if target > self._next_allowed:
                self._next_allowed = target


# Single global limiter; reconfigured by main() from --rps.
LIMITER: RateLimiter = RateLimiter(DEFAULT_RPS)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_THREAD_LOCAL = threading.local()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    return s


def thread_session() -> requests.Session:
    """Return a per-thread :class:`requests.Session` to reuse TCP connections."""
    s = getattr(_THREAD_LOCAL, "session", None)
    if s is None:
        s = make_session()
        _THREAD_LOCAL.session = s
    return s


def api_get(
    session: requests.Session,
    url: str,
    params: dict,
    *,
    max_retries: int = 5,
    timeout: int = 30,
) -> Optional[dict]:
    """GET a MediaWiki API endpoint with shared rate limiting and retries.

    Automatically injects ``maxlag`` for ``action=query`` requests so we
    back off when WMF replicas are lagging.
    """
    if params.get("action") == "query" and "maxlag" not in params:
        params = {**params, "maxlag": MAXLAG}

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        LIMITER.wait()
        try:
            r = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            wait = backoff + random.uniform(0, 0.5)
            print(
                f"\n  [api] network error attempt {attempt}/{max_retries}: "
                f"{exc}; sleep {wait:.1f}s",
                file=sys.stderr,
            )
            LIMITER.cooldown(wait)
            backoff = min(backoff * 2, 60.0)
            continue

        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError as exc:
                print(
                    f"\n  [api] JSON decode error: {exc}; "
                    f"body[:200]={r.text[:200]!r}",
                    file=sys.stderr,
                )
                return None
            # maxlag triggers HTTP 200 with an "error" envelope; retry.
            err = data.get("error") if isinstance(data, dict) else None
            if err and err.get("code") == "maxlag":
                retry_after = r.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else backoff
                except ValueError:
                    wait = backoff
                wait = max(wait, backoff) + random.uniform(0, 1.0)
                print(
                    f"\n  [api] maxlag attempt {attempt}/{max_retries}; "
                    f"sleep {wait:.1f}s",
                    file=sys.stderr,
                )
                LIMITER.cooldown(wait)
                backoff = min(backoff * 2, 60.0)
                continue
            return data

        if r.status_code == 429 or 500 <= r.status_code < 600:
            retry_after = r.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else backoff
            except ValueError:
                wait = backoff
            wait = max(wait, backoff) + random.uniform(0, 1.0)
            print(
                f"\n  [api] HTTP {r.status_code} attempt {attempt}/{max_retries}; "
                f"sleep {wait:.1f}s",
                file=sys.stderr,
            )
            LIMITER.cooldown(wait)
            backoff = min(backoff * 2, 60.0)
            continue

        # 4xx non-429: don't retry
        return None

    return None


def rest_summary(
    session: requests.Session,
    title: str,
    *,
    lang: str = "en",
    max_retries: int = 4,
    timeout: int = 20,
) -> Optional[str]:
    """
    Fetch the lead paragraph for *title* via REST /api/rest_v1/page/summary.
    Returns clean plain text, ``""`` for empty extract, or ``None`` on
    permanent failure / 404.  Retries transient errors with backoff.
    """
    base = AR_REST_BASE if lang == "ar" else EN_REST_BASE
    encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        LIMITER.wait()
        try:
            r = session.get(f"{base}/{encoded}", timeout=timeout)
        except requests.RequestException as exc:
            wait = backoff + random.uniform(0, 0.5)
            print(
                f"\n  [rest] network error attempt {attempt}/{max_retries} "
                f"on {title!r}: {exc}; sleep {wait:.1f}s",
                file=sys.stderr,
            )
            LIMITER.cooldown(wait)
            backoff = min(backoff * 2, 30.0)
            continue

        if r.status_code == 200:
            try:
                return r.json().get("extract") or ""
            except ValueError:
                return None
        if r.status_code == 404:
            return None
        if r.status_code == 429 or 500 <= r.status_code < 600:
            retry_after = r.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else backoff
            except ValueError:
                wait = backoff
            wait = max(wait, backoff) + random.uniform(0, 1.0)
            LIMITER.cooldown(wait)
            backoff = min(backoff * 2, 30.0)
            continue
        return None
    return None


def fetch_extracts_batch(
    titles: list[str],
    wiki_api: str = EN_API,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Fetch the plain-text intro section for up to ``EXTRACT_BATCH_SIZE`` titles
    in a single MediaWiki API call.

    Returns a 3-tuple:
      * ``extracts``    : ``{resolved_title: plain_text}``
      * ``redirects``   : ``{resolved_title: original_requested_title}``
      * ``canonical_urls``: ``{resolved_title: full_url}`` (from prop=info)

    Uses a per-thread session to reuse TCP connections.  Pass *wiki_api* to
    target a non-English Wikipedia (e.g. ``AR_API``).
    """
    if not titles:
        return {}, {}, {}
    session = thread_session()
    params = {
        "action":          "query",
        "format":          "json",
        "titles":          "|".join(titles),
        "prop":            "extracts|info",
        "exintro":         1,
        "explaintext":     1,
        "exsectionformat": "plain",
        "inprop":          "url",
        "redirects":       1,
    }
    data = api_get(session, wiki_api, params)
    if not data:
        return {}, {}, {}

    query = data.get("query", {}) or {}
    # Map resolved title back to the originally requested title.
    redirects: dict[str, str] = {}
    for hop in query.get("redirects", []) or []:
        if hop.get("to") and hop.get("from"):
            redirects[hop["to"]] = hop["from"]
    # Likewise normalization (e.g. underscores → spaces, casing).
    for hop in query.get("normalized", []) or []:
        if hop.get("to") and hop.get("from"):
            # If the normalized title was further redirected, chain it.
            redirects.setdefault(hop["to"], hop["from"])

    extracts: dict[str, str] = {}
    canonical_urls: dict[str, str] = {}
    for pid, page in (query.get("pages") or {}).items():
        if pid == "-1":
            continue
        title = page.get("title", "")
        if not title:
            continue
        extract = page.get("extract") or ""
        if extract:
            extracts[title] = extract
        full_url = page.get("fullurl") or page.get("canonicalurl")
        if full_url:
            canonical_urls[title] = full_url
    return extracts, redirects, canonical_urls


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = _EN_TAIL_RE.sub("", text)
    text = _AR_TAIL_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def truncate_at(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    candidate = text[:max_chars]
    idx = candidate.rfind("\n\n")
    if idx > max_chars // 2:
        return candidate[:idx].rstrip()
    idx = candidate.rfind("\n")
    if idx > max_chars // 2:
        return candidate[:idx].rstrip()
    return candidate.rstrip()


def title_to_url(title: str, lang: str = "en") -> str:
    return (
        f"https://{lang}.wikipedia.org/wiki/"
        + urllib.parse.quote(title.replace(" ", "_"), safe=":/")
    )


def url_to_title(url: str) -> str:
    m = re.search(r"/wiki/(.+)$", url)
    if not m:
        return ""
    return urllib.parse.unquote(m.group(1)).replace("_", " ")


def url_lang(url: str) -> str:
    """Return ``"ar"`` for Arabic Wikipedia URLs, otherwise ``"en"``."""
    m = re.match(r"https?://([a-z\-]+)\.wikipedia\.org/", url or "")
    if m and m.group(1).startswith("ar"):
        return "ar"
    return "en"


def normalize_url(url: str) -> str:
    """
    Normalize a Wikipedia URL for deduplication:
      * lowercase the scheme + host
      * percent-decode the title path
      * convert underscores to spaces, then back to underscores
        (idempotent space/underscore handling)
      * strip trailing slash and fragment
    """
    if not url:
        return ""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return url
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path
    m = re.match(r"^(/wiki/)(.+)$", path)
    if m:
        title = urllib.parse.unquote(m.group(2)).replace("_", " ").strip()
        # Re-encode with consistent rules (underscores, no fragment)
        path = m.group(1) + urllib.parse.quote(title.replace(" ", "_"), safe=":/")
    return urllib.parse.urlunsplit((scheme, netloc, path.rstrip("/"), "", ""))


def is_main_namespace(
    title: str,
    noise_prefixes: tuple = _NS_PREFIXES,
    *,
    drop_low_value: bool = True,
    low_value_re: Optional[re.Pattern] = None,
) -> bool:
    if any(title.startswith(p) for p in noise_prefixes):
        return False
    if drop_low_value:
        pattern = low_value_re or _EN_LOW_VALUE_RE
        if pattern.search(title):
            return False
    return True


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def load_existing_urls(path: Path) -> set[str]:
    """Return the set of normalized URLs already written to *path*."""
    if not path.exists():
        return set()
    urls: set[str] = set()
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    u = json.loads(line).get("url", "")
                    if u:
                        urls.add(normalize_url(u))
                except json.JSONDecodeError:
                    pass
    except OSError as exc:
        print(f"  [warn] could not read {path}: {exc}", file=sys.stderr)
    return urls


def make_record(
    *,
    text: str,
    title: str,
    url: str,
    category: str,
    phase: str,
    language: str = "en",
) -> dict:
    """Build a record with the step05-compatible output schema."""
    text = truncate_at(clean_text(text), DEFAULT_MAX_CHARS)
    return {
        "text":       text,
        "title":      title,
        "url":        url,
        "category":   category,
        "language":   language,
        "phase":      phase,
        "ar_url":     "",
        "ar_title":   "",
        "rec_id":     "",
        "char_count": len(text),
        "word_count": len(text.split()),
    }


class JsonlWriter:
    """Thread-safe append writer with normalized-URL deduplication.

    Opens the output file once and keeps the handle for the lifetime of the
    writer, flushing on every record so that interrupted runs lose at most
    the in-flight record.
    """

    def __init__(self, path: Path, existing_urls: set[str]) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = {normalize_url(u) for u in existing_urls if u}
        self._lock = Lock()
        self._count = 0
        self._fh = open(self._path, "a", encoding="utf-8")

    def write(self, record: dict) -> bool:
        """Append *record*; return True if written, False if a duplicate."""
        url = normalize_url(record.get("url", ""))
        with self._lock:
            if url and url in self._seen:
                return False
            if url:
                self._seen.add(url)
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._count += 1
            return True

    @property
    def seen_urls(self) -> set[str]:
        """Return a *copy* of the seen-URL set (thread-safe)."""
        with self._lock:
            return set(self._seen)

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def close(self) -> None:
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Source A/B — figure summary recovery
# ---------------------------------------------------------------------------

def run_figures_source(
    figures_path: Path,
    phase2_existing_urls: set[str],
    writer: JsonlWriter,
    min_chars: int,
    max_chars: int,
    workers: int,
    limit: Optional[int],
) -> dict:
    """
    Recover phase_2 records for figures that step05 failed to fetch.

    A) Use the ``summary`` field already in the source file (zero API calls).
    B) Call REST /page/summary (per-language) for figures whose stored
       summary is too short.

    Returns a stats dict.
    """
    all_figs = [
        json.loads(line)
        for line in open(figures_path, encoding="utf-8")
        if line.strip()
    ]

    seen_norm = {normalize_url(u) for u in phase2_existing_urls if u}

    # Only figures that are missing AND are real article pages
    missing = [
        f for f in all_figs
        if normalize_url(f.get("url", "")) not in seen_norm
        and "/wiki/Template:" not in f.get("url", "")
        and "/wiki/قالب:" not in f.get("url", "")
    ]
    print(f"  Figures missing from phase2: {len(missing):,}")
    if limit:
        missing = missing[:limit]
        print(f"  (limited to {limit} for this run)")

    # ---- A: stored summary field (free) ------------------------------------
    needs_api: list[dict] = []
    written_a = 0
    for fig in tqdm(missing, desc="  A stored summaries", dynamic_ncols=True):
        url  = fig.get("url", "")
        lang = url_lang(url)
        summary = clean_text((fig.get("summary") or "").strip())
        if len(summary) >= min_chars:
            rec = make_record(
                text=summary,
                title=(fig.get("title") or fig.get("name") or url_to_title(url)).strip(),
                url=url,
                category=(fig.get("category") or "").strip(),
                phase="phase_2",
                language=lang,
            )
            if writer.write(rec):
                written_a += 1
        else:
            needs_api.append(fig)

    print(f"  A: {written_a:,} records from stored summaries")
    print(f"  B: {len(needs_api):,} figures need REST /page/summary call")

    # ---- B: REST /page/summary (parallel, per-language) --------------------
    written_b = 0
    failed_b  = 0

    def _fetch_one(fig: dict) -> Optional[dict]:
        url  = fig.get("url", "")
        lang = url_lang(url)
        title = url_to_title(url)
        if not title:
            return None
        session = thread_session()
        extract = rest_summary(session, title, lang=lang)
        if not extract:
            return None
        extract = clean_text(extract.strip())
        if len(extract) < min_chars:
            return None
        return make_record(
            text=extract,
            title=(fig.get("title") or title).strip(),
            url=url,
            category=(fig.get("category") or "").strip(),
            phase="phase_2",
            language=lang,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, fig): fig for fig in needs_api}
        with tqdm(total=len(needs_api), desc="  B REST summary", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                rec = future.result()
                if rec:
                    if writer.write(rec):
                        written_b += 1
                else:
                    failed_b += 1
                pbar.update(1)

    print(
        f"  B: {written_b:,} records from REST API  "
        f"({failed_b:,} skipped — too short or not found)"
    )
    return {
        "candidates":            len(missing),
        "written_from_stored":   written_a,
        "written_from_rest":     written_b,
        "rest_failed_or_short":  failed_b,
    }


# ---------------------------------------------------------------------------
# Category-tree walk (parallel, level-by-level BFS)
# ---------------------------------------------------------------------------

def _fetch_category_members(
    cat_title: str,
    *,
    wiki_api: str,
) -> list[dict]:
    """Fetch all members of one category (handles cmcontinue pagination)."""
    session = thread_session()
    members: list[dict] = []
    cmcontinue: Optional[str] = None
    while True:
        params: dict = {
            "action":  "query",
            "format":  "json",
            "list":    "categorymembers",
            "cmtitle": cat_title,
            "cmtype":  "page|subcat",
            "cmlimit": 500,
            "cmprop":  "title|type",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(session, wiki_api, params)
        if not data:
            break
        for member in data.get("query", {}).get("categorymembers", []):
            members.append(member)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return members


def _iter_walk_categories(
    seed_categories: list[str],
    max_depth: int,
    seen_urls: set[str],
    *,
    wiki_api: str = EN_API,
    lang: str = "en",
    cat_ns_prefix: str = "Category:",
    noise_prefixes: tuple = _NS_PREFIXES,
    low_value_re: Optional[re.Pattern] = None,
    exclude_cat_re: Optional[re.Pattern] = None,
    workers: int = DEFAULT_WORKERS,
):
    """
    Generator: parallel level-by-level BFS over a Wikipedia category tree.
    Yields ``{title, url, category}`` dicts for article pages not already in
    *seen_urls*.  *seen_urls* is updated in-place so callers can stop the
    generator at any time without losing dedup state.

    Works for any language Wikipedia via *wiki_api* / *lang* /
    *cat_ns_prefix* / *noise_prefixes* / *low_value_re*.
    """
    visited_cats: set[str] = set()

    def _norm_cat(c: str) -> str:
        return c if c.startswith(cat_ns_prefix) else f"{cat_ns_prefix}{c}"

    def _excluded(cat_title: str) -> bool:
        if exclude_cat_re is None:
            return False
        label = cat_title.removeprefix(cat_ns_prefix)
        return bool(exclude_cat_re.search(label))

    current_level: list[str] = [_norm_cat(c) for c in seed_categories]

    for depth in range(max_depth + 1):
        # Deduplicate within the level and against already-visited.
        level = []
        for c in current_level:
            if c in visited_cats:
                continue
            visited_cats.add(c)
            if _excluded(c):
                continue
            level.append(c)
        if not level:
            break

        next_level: list[str] = []

        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_fetch_category_members, c, wiki_api=wiki_api): c
                for c in level
            }
            for future in as_completed(futures):
                cat_title = futures[future]
                cat_label = cat_title.removeprefix(cat_ns_prefix)
                try:
                    members = future.result()
                except Exception as exc:
                    print(
                        f"\n  [walk] error on {cat_title}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                for member in members:
                    title = member["title"]
                    mtype = member.get("type", "page")
                    if mtype == "subcat":
                        if (
                            depth < max_depth
                            and title not in visited_cats
                            and not _excluded(title)
                        ):
                            next_level.append(title)
                        continue
                    if not is_main_namespace(
                        title, noise_prefixes, low_value_re=low_value_re
                    ):
                        continue
                    url = title_to_url(title, lang)
                    norm = normalize_url(url)
                    if norm in seen_urls:
                        continue
                    seen_urls.add(norm)
                    yield {"title": title, "url": url, "category": cat_label}

        current_level = next_level


def run_category_source(
    seed_categories: list[str],
    max_depth: int,
    all_existing_urls: set[str],
    writer: JsonlWriter,
    min_chars: int,
    max_chars: int,
    workers: int,
    phase: str,
    limit: Optional[int],
    target_records: Optional[int] = None,
    *,
    wiki_api: str = EN_API,
    lang: str = "en",
    cat_ns_prefix: str = "Category:",
    noise_prefixes: tuple = _NS_PREFIXES,
    low_value_re: Optional[re.Pattern] = None,
    exclude_cat_re: Optional[re.Pattern] = None,
) -> dict:
    """
    Walk a Wikipedia category tree and write article intros to *writer*.

    Uses ``prop=extracts&exintro=1&explaintext=1`` for clean plain text —
    no wikitext parsing required.

    Stopping conditions (whichever fires first):
      * the BFS exhausts all categories at *max_depth*;
      * *limit* candidate pages have been **discovered** (pre-filter cap);
      * *target_records* records have been **written** to *writer*.

    Returns a stats dict.
    """
    seen = {normalize_url(u) for u in all_existing_urls if u}

    print(
        f"  Seed categories ({len(seed_categories)}): "
        f"{', '.join(seed_categories[:5])}"
        + (" …" if len(seed_categories) > 5 else "")
    )
    print(f"  Max depth: {max_depth}")
    if target_records:
        print(f"  Target records (post-filter): {target_records:,}")

    walker = _iter_walk_categories(
        seed_categories, max_depth, seen,
        wiki_api=wiki_api, lang=lang,
        cat_ns_prefix=cat_ns_prefix, noise_prefixes=noise_prefixes,
        low_value_re=low_value_re, exclude_cat_re=exclude_cat_re,
        workers=workers,
    )

    def _fetch_batch(
        batch: list[dict],
    ) -> tuple[list[dict], int, int]:
        """Fetch extracts for one batch.

        Returns ``(kept_records, too_short_count, fetched_count)`` where
        ``fetched_count`` is the number of titles for which the API returned
        an extract (anything else counts as a fetch failure).
        """
        titles = [d["title"] for d in batch]
        meta_by_requested = {d["title"]: d for d in batch}
        extracts, redirects, canonical_urls = fetch_extracts_batch(titles, wiki_api)
        kept: list[dict] = []
        too_short = 0
        for resolved_title, text in extracts.items():
            text = clean_text(text)
            if len(text) < min_chars:
                too_short += 1
                continue
            text = truncate_at(text, max_chars)
            # Walk the redirect chain back to the originally requested title
            # so we can reattach the seed-category label.
            origin = resolved_title
            for _ in range(5):  # bounded chain traversal
                prev = redirects.get(origin)
                if prev is None or prev == origin:
                    break
                origin = prev
            meta = (
                meta_by_requested.get(origin)
                or meta_by_requested.get(resolved_title)
                or {}
            )
            url = (
                canonical_urls.get(resolved_title)
                or meta.get("url")
                or title_to_url(resolved_title, lang)
            )
            kept.append(make_record(
                text=text,
                title=resolved_title,
                url=url,
                category=meta.get("category") or "",
                phase=phase,
                language=lang,
            ))
        return kept, too_short, len(extracts)

    discovered_total = 0
    written          = 0
    duplicates       = 0
    too_short        = 0
    fetch_failed     = 0

    walker_exhausted = False
    walk_pbar  = tqdm(desc="  Walking",  unit="page", dynamic_ncols=True)
    fetch_pbar = tqdm(desc="  Written",  unit="rec",  dynamic_ncols=True,
                      total=target_records)

    try:
        while True:
            # ---- discover up to BATCH_FETCH_GROUP * EXTRACT_BATCH_SIZE pages
            BATCH_FETCH_GROUP = max(1, workers)  # parallelize this many batches
            wave_target = BATCH_FETCH_GROUP * EXTRACT_BATCH_SIZE
            wave: list[dict] = []
            for cand in walker:
                wave.append(cand)
                discovered_total += 1
                walk_pbar.update(1)
                if limit and discovered_total >= limit:
                    break
                if len(wave) >= wave_target:
                    break
            else:
                walker_exhausted = True

            if not wave:
                break

            # ---- fetch extracts for this wave in parallel batches
            batches = [
                wave[i : i + EXTRACT_BATCH_SIZE]
                for i in range(0, len(wave), EXTRACT_BATCH_SIZE)
            ]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_fetch_batch, b): b for b in batches}
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        kept, batch_too_short, fetched_count = future.result()
                    except Exception as exc:
                        print(
                            f"\n  [extracts] batch error: {exc}",
                            file=sys.stderr,
                        )
                        fetch_failed += len(batch)
                        continue
                    too_short    += batch_too_short
                    fetch_failed += max(0, len(batch) - fetched_count)
                    for rec in kept:
                        if writer.write(rec):
                            written += 1
                            fetch_pbar.update(1)
                        else:
                            duplicates += 1

            # ---- check stopping conditions
            if target_records and written >= target_records:
                break
            if limit and discovered_total >= limit:
                break
            if walker_exhausted:
                break
    finally:
        walk_pbar.close()
        fetch_pbar.close()

    print(
        f"  Category walk: {written:,} new records written  "
        f"(discovered {discovered_total:,}, "
        f"{duplicates:,} duplicates, "
        f"{too_short:,} below min_chars, "
        f"{fetch_failed:,} fetch-failed)"
    )
    return {
        "discovered":   discovered_total,
        "written":      written,
        "duplicates":   duplicates,
        "too_short":    too_short,
        "fetch_failed": fetch_failed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ---- source selection --------------------------------------------------
    p.add_argument(
        "--source",
        choices=["figures", "categories", "ar-categories", "all"],
        default="all",
        help=(
            "Which augmentation source to run.  "
            "'figures' = figure summary recovery (phase_2); "
            "'categories' = EN category-tree walk (phase_1 by default); "
            "'ar-categories' = AR category-tree walk (phase_1 by default); "
            "'all' = all sources  (default: all)"
        ),
    )

    # ---- paths -------------------------------------------------------------
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DATA_DIR,
        help="Directory containing the existing phase JSONL files "
             f"(default: {_DATA_DIR})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DATA_DIR,
        help="Directory to write the augmentation JSONL files "
             f"(default: same as --data-dir)",
    )
    p.add_argument(
        "--figures-source",
        type=Path,
        default=FIGURES_SOURCE,
        help=f"Path to egyptian_figures_data.jsonl (default: {FIGURES_SOURCE})",
    )

    # ---- category walk options ---------------------------------------------
    p.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        metavar="CAT",
        help=(
            "EN Wikipedia category names to walk (space-separated, "
            "no 'Category:' prefix needed).  "
            f"Default: {DEFAULT_CATEGORIES[:3]} … ({len(DEFAULT_CATEGORIES)} total)"
        ),
    )
    p.add_argument(
        "--category-phase",
        choices=["phase_1", "phase_2"],
        default="phase_1",
        help=(
            "Assign EN category-walk records to this phase label "
            "(default: phase_1).  Use phase_2 with biographical categories "
            "such as 'Egyptian people'."
        ),
    )

    # ---- Arabic category walk options --------------------------------------
    p.add_argument(
        "--ar-categories",
        nargs="+",
        default=DEFAULT_AR_CATEGORIES,
        metavar="CAT",
        help=(
            "AR Wikipedia category names to walk (in Arabic, space-separated, "
            "no 'تصنيف:' prefix needed).  "
            f"Default: {DEFAULT_AR_CATEGORIES[:3]} … ({len(DEFAULT_AR_CATEGORIES)} total)"
        ),
    )
    p.add_argument(
        "--ar-category-phase",
        choices=["phase_1", "phase_2"],
        default="phase_1",
        help=(
            "Assign AR category-walk records to this phase label "
            "(default: phase_1).  Use phase_2 with biographical categories "
            "such as 'شخصيات مصرية'."
        ),
    )

    p.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum category recursion depth (default: {DEFAULT_MAX_DEPTH})",
    )

    p.add_argument(
        "--keep-low-value",
        action="store_true",
        help=(
            "Keep 'List of …', 'Index of …', 'Outline of …', disambiguation, "
            "and the Arabic equivalents.  By default these low-prose pages "
            "are filtered out of the category walk."
        ),
    )

    p.add_argument(
        "--exclude-cat-regex",
        type=str,
        default=None,
        metavar="PATTERN",
        help=(
            "Regex (case-insensitive) matched against category labels (the "
            "part after 'Category:' / 'تصنيف:').  Any category whose label "
            "matches is skipped — both as a seed and as a sub-category found "
            "during the walk.  Useful for dropping Wikipedia stub-tag "
            "categories which dominate output, e.g. "
            r"--exclude-cat-regex '(?i)stubs?$|^بذرة |بذرة$'."
        ),
    )

    # ---- quality thresholds ------------------------------------------------
    p.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Minimum text length to keep a record (default: {DEFAULT_MIN_CHARS})",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Hard text length cap; longer texts are truncated at the last "
             f"paragraph break (default: {DEFAULT_MAX_CHARS})",
    )

    # ---- performance -------------------------------------------------------
    p.add_argument(
        "--rps",
        type=float,
        default=DEFAULT_RPS,
        help=(
            "Maximum requests per second shared across all workers.  "
            "Wikimedia allows bots well above 1 req/s, but be polite: "
            f"2–4 is reasonable for sustained runs  (default: {DEFAULT_RPS})"
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"Parallel worker threads for API calls  (default: {DEFAULT_WORKERS}).  "
            "Each worker reuses one HTTP session via thread-local storage."
        ),
    )

    # ---- testing -----------------------------------------------------------
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Hard cap on the number of candidate pages **discovered** per "
            "category source (BEFORE quality filtering).  The number of "
            "records actually written may be lower than N because some "
            "candidates fall below --min-chars.  Combine with --target-records "
            "for a smoke test that yields a predictable number of records."
        ),
    )
    p.add_argument(
        "--target-records",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop each category source once at least N records have been "
            "**written** (i.e. passed quality filtering).  Useful for smoke "
            "tests: '--target-records 30' guarantees at least 30 saved "
            "samples even when many candidates are short stub articles.  "
            "The actual count may slightly exceed N because the in-flight "
            "fetch wave (workers × 20 titles) finishes before stopping.  "
            "When combined with --limit the source stops as soon as either "
            "cap is hit."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Reconfigure global rate limiter from CLI argument
    global LIMITER
    LIMITER = RateLimiter(args.rps)

    data_dir   = args.data_dir
    out_dir    = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    p1_aug_path    = out_dir / DEFAULT_PHASE1_AUG
    p2_aug_path    = out_dir / DEFAULT_PHASE2_AUG
    p1_ar_aug_path = out_dir / DEFAULT_PHASE1_AR_AUG
    p2_ar_aug_path = out_dir / DEFAULT_PHASE2_AR_AUG
    stats_path     = out_dir / STATS_FILE

    print("\n=== step07: English & Arabic dataset augmentation ===\n")
    print(f"  source         : {args.source}")
    print(f"  rps            : {args.rps}")
    print(f"  workers        : {args.workers}")
    print(f"  min_chars      : {args.min_chars}")
    print(f"  max_chars      : {args.max_chars}")
    print(f"  drop_low_value : {not args.keep_low_value}")
    if args.limit:
        print(f"  limit          : {args.limit}  ← TEST MODE (pre-filter cap)")
    if args.target_records:
        print(f"  target_records : {args.target_records}  ← stop after N written")
    print()

    en_low_value = None if args.keep_low_value else _EN_LOW_VALUE_RE
    ar_low_value = None if args.keep_low_value else _AR_LOW_VALUE_RE
    exclude_cat_re = (
        re.compile(args.exclude_cat_regex, re.IGNORECASE)
        if args.exclude_cat_regex else None
    )
    if exclude_cat_re:
        print(f"  exclude_cat    : {args.exclude_cat_regex!r}")

    # ---- Deduplication: load URLs from existing files + any previous aug run
    print("Loading existing URLs for deduplication …")
    p1_existing = (
        load_existing_urls(data_dir / "phase1-en.jsonl")
        | load_existing_urls(p1_aug_path)
    )
    p2_existing = (
        load_existing_urls(data_dir / "phase2-en.jsonl")
        | load_existing_urls(p2_aug_path)
    )
    # Arabic dedup: use articles-urls.jsonl (the full phase_1 AR source) as
    # baseline so we never re-collect articles already in the HF dataset.
    p1_ar_existing = (
        load_existing_urls(data_dir / "articles-urls.jsonl")
        | load_existing_urls(p1_ar_aug_path)
    )
    p2_ar_existing = load_existing_urls(p2_ar_aug_path)

    print(
        f"  EN phase1 seen: {len(p1_existing):,}  "
        f"EN phase2 seen: {len(p2_existing):,}\n"
        f"  AR phase1 seen: {len(p1_ar_existing):,}  "
        f"AR phase2 seen: {len(p2_ar_existing):,}"
    )

    p1_writer    = JsonlWriter(p1_aug_path,    p1_existing)
    p2_writer    = JsonlWriter(p2_aug_path,    p2_existing)
    p1_ar_writer = JsonlWriter(p1_ar_aug_path, p1_ar_existing)
    p2_ar_writer = JsonlWriter(p2_ar_aug_path, p2_ar_existing)

    stats: dict = {
        "args": {
            "source":            args.source,
            "rps":               args.rps,
            "workers":           args.workers,
            "min_chars":         args.min_chars,
            "max_chars":         args.max_chars,
            "max_depth":         args.max_depth,
            "limit":             args.limit,
            "target_records":    args.target_records,
            "drop_low_value":    not args.keep_low_value,
            "category_phase":    args.category_phase,
            "ar_category_phase": args.ar_category_phase,
        },
        "sources": {},
    }

    try:
        # ---- Source A/B: figure summary recovery → phase_2 ----------------
        if args.source in ("figures", "all"):
            if not args.figures_source.exists():
                print(
                    f"[warn] figures source not found: {args.figures_source}",
                    file=sys.stderr,
                )
            else:
                print(f"\n-- Source A/B: Figure summary recovery → "
                      f"{p2_aug_path.name}/{p2_ar_aug_path.name} --")
                # The figures file may contain both EN and AR URLs.  We pass
                # both writers' seen-URL views so we don't double-write either
                # language.  The writer itself also dedups.
                fig_seen = p2_writer.seen_urls | p2_ar_writer.seen_urls
                # The figures source historically targets EN phase_2; route
                # writes through p2_writer for EN and p2_ar_writer for AR.
                # We implement that with a small dispatching wrapper.
                class _Dispatch:
                    @staticmethod
                    def write(rec: dict) -> bool:
                        if rec.get("language") == "ar":
                            return p2_ar_writer.write(rec)
                        return p2_writer.write(rec)
                # Mimic JsonlWriter just enough for run_figures_source:
                fig_writer = _Dispatch()  # type: ignore[assignment]
                stats["sources"]["figures"] = run_figures_source(
                    figures_path=args.figures_source,
                    phase2_existing_urls=fig_seen,
                    writer=fig_writer,  # type: ignore[arg-type]
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                    workers=args.workers,
                    limit=args.limit,
                )
                print(f"  phase2-en-aug running total: {p2_writer.count:,} records")
                print(f"  phase2-ar-aug running total: {p2_ar_writer.count:,} records")

        # ---- Source C: EN category-tree walk → phase_1 (or phase_2) -------
        if args.source in ("categories", "all"):
            target_writer   = p1_writer if args.category_phase == "phase_1" else p2_writer
            # Dedup against on-disk EN URLs **and** anything the figures
            # source just wrote into either EN writer this run.
            target_existing = (
                p1_existing
                | p2_existing
                | p1_writer.seen_urls
                | p2_writer.seen_urls
            )

            aug_file = p1_aug_path if args.category_phase == "phase_1" else p2_aug_path
            print(f"\n-- Source C: EN Category walk → {aug_file.name} "
                  f"({args.category_phase}) --")

            stats["sources"]["categories_en"] = run_category_source(
                seed_categories=args.categories,
                max_depth=args.max_depth,
                all_existing_urls=target_existing,
                writer=target_writer,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                workers=args.workers,
                phase=args.category_phase,
                limit=args.limit,
                target_records=args.target_records,
                low_value_re=en_low_value,
                exclude_cat_re=exclude_cat_re,
            )
            print(f"  phase1-en-aug running total: {p1_writer.count:,} records")

        # ---- Source D: AR category-tree walk → phase_1 (or phase_2) -------
        if args.source in ("ar-categories", "all"):
            ar_target_writer = (
                p1_ar_writer if args.ar_category_phase == "phase_1" else p2_ar_writer
            )
            # Dedup against on-disk AR URLs **and** anything any AR writer
            # has accumulated this run (figures source may have written AR
            # records into p2_ar_writer).
            ar_target_existing = (
                p1_ar_existing
                | p2_ar_existing
                | p1_ar_writer.seen_urls
                | p2_ar_writer.seen_urls
            )

            ar_aug_file = (
                p1_ar_aug_path if args.ar_category_phase == "phase_1" else p2_ar_aug_path
            )
            print(
                f"\n-- Source D: AR Category walk → {ar_aug_file.name} "
                f"({args.ar_category_phase}) --"
            )

            stats["sources"]["categories_ar"] = run_category_source(
                seed_categories=args.ar_categories,
                max_depth=args.max_depth,
                all_existing_urls=ar_target_existing,
                writer=ar_target_writer,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                workers=args.workers,
                phase=args.ar_category_phase,
                limit=args.limit,
                target_records=args.target_records,
                wiki_api=AR_API,
                lang="ar",
                cat_ns_prefix="تصنيف:",
                noise_prefixes=_AR_NS_PREFIXES,
                low_value_re=ar_low_value,
                exclude_cat_re=exclude_cat_re,
            )
            print(f"  phase1-ar-aug running total: {p1_ar_writer.count:,} records")
    finally:
        stats["totals"] = {
            DEFAULT_PHASE1_AUG:    p1_writer.count,
            DEFAULT_PHASE2_AUG:    p2_writer.count,
            DEFAULT_PHASE1_AR_AUG: p1_ar_writer.count,
            DEFAULT_PHASE2_AR_AUG: p2_ar_writer.count,
        }
        try:
            with open(stats_path, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"  [warn] could not write stats: {exc}", file=sys.stderr)

        p1_writer.close()
        p2_writer.close()
        p1_ar_writer.close()
        p2_ar_writer.close()

    # ---- Final summary -----------------------------------------------------
    print("\n=== Done ===")
    print(f"  {p1_aug_path.name}    : {p1_writer.count:,} new EN records")
    print(f"  {p2_aug_path.name}    : {p2_writer.count:,} new EN records")
    print(f"  {p1_ar_aug_path.name} : {p1_ar_writer.count:,} new AR records")
    print(f"  {p2_ar_aug_path.name} : {p2_ar_writer.count:,} new AR records")
    print(f"  stats              : {stats_path}")
    print(
        "\nNext step — rebuild the full dataset with all augmentation:\n"
        "  python ../clean_and_build_dataset.py \\\n"
        f"      --no-upload \\\n"
        f"      --phase1-en-aug {p1_aug_path} \\\n"
        f"      --phase2-en-aug {p2_aug_path} \\\n"
        f"      --phase1-ar-aug {p1_ar_aug_path} \\\n"
        f"      --phase2-ar-aug {p2_ar_aug_path}"
    )


if __name__ == "__main__":
    main()
