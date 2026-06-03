"""
scrape_berezkin.py — fetch the ~2,557 motif pages from ruthenia.ru/folklore/berezkin
that match our v2 motif IDs, parse out abstracts + tradition names + citations,
write to dataset/mapping_v2/motif_abstracts.parquet.

This is the original Berezkin & Duvakin Russian catalogue. Pages are in
windows-1251 encoding. Each page has:
  - title (heading)
  - one-line description
  - many <p class="NormalMai"> blocks, each is one regional bucket; inside it
    are abstract chunks with tradition name <u>X</u> and trailing citation
    "Author Year, № N: pp.".

Politeness:
  robots.txt asks Crawl-Delay: 10. We default to --sleep 3 (still
  conservative for an academic server). Per-host rate-limited; sequential
  requests.

Outputs:
  dataset/raw_downloads/berezkin_html/{motif_id}.html   raw HTML cache
  dataset/mapping_v2/motif_abstracts.parquet            parsed records

Resumable: if the cached HTML exists and is non-empty, we skip the fetch.

Usage:
    python scrape_berezkin.py                       # all our 2,557 motifs
    python scrape_berezkin.py --sleep 5             # slower
    python scrape_berezkin.py --resume              # skip already-parsed
    python scrape_berezkin.py --limit 20            # smoke test
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))  # project root for shared utils

import argparse, re, time, sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]  # project root
MAP = ROOT / "dataset/mapping_v2"
HTML_CACHE = ROOT / "raw_downloads/berezkin_html"
HTML_CACHE.mkdir(parents=True, exist_ok=True)
OUT = MAP / "motif_abstracts.parquet"

BASE = "http://www.ruthenia.ru/folklore/berezkin"
UA = ("Mozilla/5.0 (research-bot; pareidolia-myth-project; "
      "contact: research@local)")


def fetch_page(motif_id: str, sleep_s: float, max_retries: int = 3) -> str | None:
    cache = HTML_CACHE / f"{motif_id}.html"
    if cache.exists() and cache.stat().st_size > 300:
        return cache.read_text(encoding="cp1251", errors="replace")
    url = f"{BASE}/{motif_id}.html"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=60)
            if r.status_code == 404:
                # don't retry, motif page genuinely missing
                cache.write_bytes(b"")
                return None
            r.raise_for_status()
            cache.write_bytes(r.content)
            time.sleep(sleep_s)
            return r.content.decode("cp1251", errors="replace")
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {motif_id} attempt {attempt+1}/{max_retries}] {type(e).__name__}: {e}; sleep {wait}s")
            time.sleep(wait)
    return None


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

# Pattern for the motif's own header in <p class="NormalLin"> at top:
# "A32. Фигура на лунном диске, A751."
HEADER_RE = re.compile(r'<p class="NormalLin">\s*([A-Za-z]\d+\w*)\.\s*(.*?)\.\s*</p>',
                       re.S)
# Description in <p class="NormalLis">
DESC_RE = re.compile(r'<p class="NormalLis">\s*(.*?)\s*</p>', re.S)
# Regional block in <p class="NormalMai">
REGION_RE = re.compile(
    r'<p class="NormalMai">\s*<b>(.*?)</b>\s*(.*?)\s*</p>',
    re.S
)
# Within a regional block, traditions are <u>Name</u>...content...citation
# We'll split on <u>...</u> tags.
TRADITION_RE = re.compile(r"<u>(.*?)</u>", re.S)
# Citations look like "Author Year, № N: pp." at end of bracketed content
# or just "Author Year: pp." — we'll capture the trailing "...: pp.]"
CITATION_RE = re.compile(
    r"\]:\s*([^\[\]]+?\d{4}[^\[\]]*?)(?:\s*\.\s*)?$",
    re.S
)


def strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_motif_page(motif_id: str, html: str) -> list[dict]:
    """Returns a list of records, one per (motif, region, tradition, abstract)."""
    if not html:
        return []
    # Strip leading framing junk so regexes don't pull index header garbage
    # by anchoring on the <div id="main"> block
    m = re.search(r'<div id="main">(.*?)</body>', html, re.S)
    body = m.group(1) if m else html

    out = []
    # Header
    h = HEADER_RE.search(body)
    title = strip_html(h.group(2)) if h else ""
    # Description
    d = DESC_RE.search(body)
    description = strip_html(d.group(1)) if d else ""

    for rm in REGION_RE.finditer(body):
        region = strip_html(rm.group(1))
        block_html = rm.group(2)
        # Split into per-tradition chunks: text between <u> tags
        # Approach: find each <u>...</u> tradition name, then take text up to
        # the next <u>...</u> or end of block.
        positions = []
        for um in TRADITION_RE.finditer(block_html):
            positions.append((um.start(), um.end(), strip_html(um.group(1))))
        if not positions:
            # no <u>...</u> — store the whole block as one record
            text = strip_html(block_html)
            cit = ""
            cm = CITATION_RE.search(text)
            if cm:
                cit = cm.group(1).strip().rstrip(".").strip()
                text = text[:cm.start()].strip()
            out.append({
                "motif_id": motif_id, "title": title, "description": description,
                "region": region, "tradition_name_ru": None,
                "abstract_ru": text, "citation": cit,
            })
            continue
        # for each tradition: text from end of <u> to start of next <u> (or end of block)
        for i, (us, ue, name) in enumerate(positions):
            end = positions[i+1][0] if i+1 < len(positions) else len(block_html)
            chunk_html = block_html[ue:end]
            text = strip_html(chunk_html)
            # citation: ":<stuff>" trailing
            cit = ""
            cm = CITATION_RE.search(text)
            if cm:
                cit = cm.group(1).strip().rstrip(".").strip()
                text = text[:cm.start()].strip()
            out.append({
                "motif_id": motif_id, "title": title, "description": description,
                "region": region, "tradition_name_ru": name,
                "abstract_ru": text, "citation": cit,
            })
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="seconds between HTTP requests (default 3s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to first N motifs (for smoke test)")
    ap.add_argument("--resume", action="store_true",
                    help="skip motifs already parsed in the output parquet")
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()

    motifs = pd.read_parquet(MAP / "motifs.parquet")[["motif_id"]]
    if args.limit:
        motifs = motifs.head(args.limit)
    print(f"[scrape] {len(motifs)} motifs to fetch")

    done = set()
    all_records = []
    if args.resume and OUT.exists():
        prev = pd.read_parquet(OUT)
        done = set(prev["motif_id"].unique())
        all_records = prev.to_dict("records")
        print(f"[resume] {len(done)} motifs already in {OUT}")

    todo = motifs[~motifs["motif_id"].isin(done)]
    t0 = time.time()
    n_since_save = 0
    n_no_page = 0
    try:
        for mid in tqdm(todo["motif_id"], desc="motifs"):
            html = fetch_page(mid, args.sleep)
            if html is None:
                n_no_page += 1
                continue
            recs = parse_motif_page(mid, html)
            all_records.extend(recs)
            n_since_save += 1
            if n_since_save >= args.save_every:
                pd.DataFrame(all_records).to_parquet(OUT, index=False)
                n_since_save = 0
    except KeyboardInterrupt:
        print("\n[interrupt] saving partial...")

    pd.DataFrame(all_records).to_parquet(OUT, index=False)
    elapsed = time.time() - t0
    df = pd.DataFrame(all_records)
    print(f"\n[done] {len(df):,} abstract records from "
          f"{df['motif_id'].nunique() if len(df) else 0} motifs in {elapsed/60:.1f} min")
    print(f"  page-not-found (404): {n_no_page}")
    if len(df):
        print(f"  regions: {df['region'].nunique()}")
        print(f"  traditions (RU): {df['tradition_name_ru'].nunique()}")
        print(f"  abstracts/motif: median {df.groupby('motif_id').size().median():.0f}, "
              f"max {df.groupby('motif_id').size().max()}")


if __name__ == "__main__":
    main()
