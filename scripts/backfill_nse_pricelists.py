#!/usr/bin/env python3
"""Backfill real NSE daily history from the exchange's daily price lists.

A one-off. The NSE strategies need 50 daily bars and the live ticker adds
one per session; this reads the published price lists for the sessions
before that, so the strategies can start in days rather than weeks. How a
price is read and checked is in src/connectors/nse_pricelist.py.

The lists are scanned images, so this needs an OCR engine that is not part
of the app image. Install it into a scratch folder first (it disappears at
the next redeploy, which is fine: the reading is cached):

    python3 -m pip install -q --no-cache-dir --target /tmp/nseocr --no-deps \
        rapidocr-onnxruntime==1.4.4
    python3 -m pip install -q --no-cache-dir --target /tmp/nseocr \
        opencv-python-headless onnxruntime==1.30.0 pymupdf==1.28.2 \
        pyclipper shapely six tqdm pyyaml pillow

Then, from /app:

    # Read and check. Writes nothing but the reading cache. ~40 s a list.
    nice -n 10 python3 scripts/backfill_nse_pricelists.py --days 70

    # Store the accepted bars. Uses the cache, so reads no PDF again.
    python3 scripts/backfill_nse_pricelists.py --days 70 --write

What each list said is cached in data/nse_pricelists/, one file per session,
so an interrupted run resumes where it stopped.

Every NSE stock, not only the watched ones: the lists identify stocks by
ISIN, and the ISINs of all but nine are learned by matching the lists'
prices against what the NSE ticker feed recorded on the same sessions
(src/connectors/nse_pricelist.learn_isins). The agent records every listed
stock from the feed, so after a few sessions of that, run this again
(cached lists are not read twice) and it learns the rest and fills their
history. Learned ISINs are kept in data/nse_isin_map.json.
"""
import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.connectors import nse_isin  # noqa: E402
from src.connectors import nse_pricelist as pl  # noqa: E402
from src.connectors.nse_scraper import DATA_DIR as NSE_CSV_DIR  # noqa: E402
from src.connectors.nse_scraper import EAT  # noqa: E402
from src.utils.paths import DATA_DIR  # noqa: E402

CACHE_DIR = DATA_DIR / "nse_pricelists"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
NEEDED = 50  # TechnicalStrategy lookback_period


def say(msg: str = "") -> None:
    print(msg, flush=True)  # flushed, so a log followed with tail stays current


def sessions(end: date, days: int) -> List[date]:
    """The last `days` weekdays up to and including end, oldest first."""
    out, d = [], end
    while len(out) < days:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


class Reader:
    """Downloads and OCRs lists, loading the engine only if one is needed."""

    def __init__(self, ocr_path: str, threads: int, pause: float):
        self.threads, self.pause = threads, pause
        self._engine = None
        # First on the path, before anything imports numpy: the OCR stack
        # (pymupdf, onnxruntime, opencv and the numpy they were built with)
        # lives only here. Adding it when the engine first loaded was too
        # late, because ocr() imports pymupdf before it asks for the engine,
        # so every run on the server stopped with "No module named 'fitz'".
        if ocr_path not in sys.path:
            sys.path.insert(0, ocr_path)

    def engine(self):
        if self._engine is None:
            # Not in the app image by design; installed to --ocr-path for this run.
            from rapidocr_onnxruntime import RapidOCR  # pylint: disable=import-error
            self._engine = RapidOCR(intra_op_num_threads=self.threads)
        return self._engine

    def ocr(self, pdf: bytes) -> List[dict]:
        import numpy as np
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        doc = fitz.open(stream=pdf, filetype="pdf")
        pages = []
        # The watched stocks are on the first two pages; the third holds notes.
        for pno in range(min(2, doc.page_count)):
            pix = doc[pno].get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result, _ = self.engine()(img[:, :, :3])
            pages.append({"width": pix.width,
                          "boxes": [list(b) for b in pl.boxes_from_ocr(result)]})
        return pages

    def read(self, d: date, today: date) -> Optional[dict]:
        cache = CACHE_DIR / f"{d.isoformat()}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        import requests
        url = pl.PRICELIST_URL.format(name=pl.pricelist_name(d))
        t0 = time.time()
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except Exception as e:
            say(f"  {d}: download failed ({type(e).__name__}); will retry next run")
            return None
        finally:
            time.sleep(self.pause)  # one request every few seconds, never a burst
        if r.status_code == 404:
            entry = {"url": url, "status": 404}
            if (today - d).days <= 3:  # may simply not be published yet
                say(f"  {d}: not published (yet)")
                return entry
        elif r.status_code != 200:
            say(f"  {d}: HTTP {r.status_code}; will retry next run")
            return None
        else:
            entry = {"url": url, "status": 200, "pages": self.ocr(r.content)}
            say(f"  {d}: read {len(entry['pages'])} pages in {time.time() - t0:.0f}s")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(entry))
        return entry


def readings_of(entry: dict, isin_map: Optional[Dict[str, str]] = None,
                by_isin: bool = False) -> Dict[str, pl.Reading]:
    """One list's rows, keyed by ticker (mapped ISINs only) or by ISIN."""
    out: Dict[str, pl.Reading] = {}
    anchors = None
    for page in entry.get("pages", []):
        boxes = [pl.Box(*b) for b in page["boxes"]]
        rows, anchors = (pl.read_isin_rows(boxes, page["width"], anchors) if by_isin
                         else pl.read_page(boxes, page["width"], anchors, isin_map))
        for key, row in rows.items():
            # A stock on two pages is two rows claiming it: trust neither.
            out[key] = {c: None for c in pl.COLUMNS} if key in out else row
    return out


def learn(entries: Dict[date, dict]) -> Dict[str, str]:
    """Learn unknown ISINs from the lists read and the ticker feed's record."""
    known = nse_isin.load()
    by_isin = {d: readings_of(e, by_isin=True) for d, e in entries.items()}
    tickers = sorted(p.stem.upper() for p in NSE_CSV_DIR.glob("*.csv"))
    found = pl.learn_isins(by_isin, pl.recorded_closes(tickers), known)
    return nse_isin.remember(found, "price lists matched to the NSE ticker feed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--days", type=int, default=70, help="sessions to read, ending at --end")
    p.add_argument("--end", help="last session, YYYY-MM-DD (default: yesterday)")
    p.add_argument("--write", action="store_true", help="store the accepted bars")
    p.add_argument("--ocr-path", default="/tmp/nseocr")
    p.add_argument("--threads", type=int, default=2, help="CPU threads for OCR")
    p.add_argument("--pause", type=float, default=3.0, help="seconds between downloads")
    p.add_argument("--no-learn", action="store_true",
                   help="do not learn new ISINs; only the stocks already mapped")
    args = p.parse_args()

    today = datetime.now(EAT).date()
    end = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    dates = sessions(end, args.days)
    say(f"NSE price lists: {len(dates)} sessions, {dates[0]} to {dates[-1]}")

    reader = Reader(args.ocr_path, args.threads, args.pause)
    entries: Dict[date, dict] = {}
    readings: Dict[date, Dict[str, pl.Reading]] = {}
    unpublished, failed = [], []
    for d in dates:
        try:
            entry = reader.read(d, today)
        except ImportError as e:
            say(f"\nThe OCR engine is not installed ({e}). Install it with the two pip "
                f"commands at the top of this script, then run it again.")
            return 2
        if entry is None:
            failed.append(d)
        elif entry["status"] == 404:
            unpublished.append(d)
        else:
            entries[d] = entry

    if not args.no_learn:
        added = learn(entries)
        say(f"\nISINs learned this run: {len(added)}"
            + (f" ({', '.join(f'{s} {i}' for i, s in sorted(added.items(), key=lambda kv: kv[1]))})"
               if added else " (a stock needs 3 sessions the ticker feed also recorded)"))
    isin_map = nse_isin.load()
    symbols = sorted(set(isin_map.values()))
    readings = {d: readings_of(e, isin_map) for d, e in entries.items()}

    verdicts = pl.validate(readings, symbols)
    say(f"\nRead {len(readings)} lists. Not published (holidays): "
        f"{', '.join(map(str, unpublished)) or 'none'}. Failed: {', '.join(map(str, failed)) or 'none'}.")
    say(f"\n{'SYMBOL':7} {'ACCEPTED':>8} {'CONFIRMED':>10} {'RANGE-CHK':>10} {'REJECTED':>9}")
    for sym in symbols:
        mine = [v for v in verdicts if v.symbol == sym]
        ok = [v for v in mine if v.bar]
        say(f"{sym:7} {len(ok):>8} {sum(v.how == 'confirmed' for v in ok):>10} "
            f"{sum(v.how == 'range-checked' for v in ok):>10} {len(mine) - len(ok):>9}"
            f"{'' if len(ok) >= NEEDED else f'   (short of the {NEEDED} the strategies need)'}")

    say("\nRejected, and why:")
    for sym in symbols:
        rejected = [v for v in verdicts if v.symbol == sym and not v.bar]
        for v in rejected[:8]:
            say(f"  {sym:5} {v.date}  {v.how}")
        if len(rejected) > 8:
            say(f"  {sym:5} ... and {len(rejected) - 8} more")

    say("\nLatest accepted prices, to compare with the NSE website:")
    for sym in symbols:
        ok = [v for v in verdicts if v.symbol == sym and v.bar][-3:]
        say(f"  {sym:5} " + "  ".join(f"{v.date} {v.bar.close:.2f}" for v in ok))

    if not args.write:
        total = sum(1 for v in verdicts if v.bar)
        say(f"\nNothing written. Run again with --write to store the {total} accepted bars.")
        return 0
    say("\nWriting (never over a bar a live source already recorded):")
    for sym in symbols:
        bars = [v.bar for v in verdicts if v.symbol == sym and v.bar]
        written, skipped = pl.merge_into_csv(sym, bars)
        say(f"  {sym:5} {written} written, {skipped} kept as already recorded")
    say("\nDone. The agent retries its history warm-start hourly and will pick these "
        "up within the hour; restarting the backend picks them up at once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
