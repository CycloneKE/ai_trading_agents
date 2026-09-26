"""Which ISIN is which NSE ticker.

The daily price lists identify each stock by its ISIN (the international
securities code), because OCR reads that reliably and garbles company
names. Nine ISINs were checked by hand when the price-list reader was
written. The rest are learned, never guessed:

- from the NSE ticker feed, when its rows carry an ISIN beside the ticker;
- from the price lists themselves (nse_pricelist.learn_isins): an unknown
  ISIN's printed prices are matched against the prices the ticker feed
  recorded for each stock on the same sessions, and a mapping is kept only
  when exactly one stock matches, session after session.

Learned mappings are saved in data/nse_isin_map.json with how they were
learned, so they survive redeploys and can be checked. The feed states the
mapping outright, so a mapping read from it replaces one inferred from
prices; nothing replaces the hand-checked nine.
"""
import json
import logging
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import fcntl
except ImportError:  # not on Windows; the in-process lock still applies
    fcntl = None

logger = logging.getLogger(__name__)

# Read from the price lists and checked by hand against the NSE website.
VERIFIED: Dict[str, str] = {
    "KE1000001402": "SCOM", "KE0000000554": "EQTY", "KE0000000315": "KCB",
    "KE1000001568": "COOP", "KE0000000448": "SCBK", "KE0000000067": "ABSA",
    "KE0000000075": "BAT", "KE0000000216": "EABL", "KE0000000406": "NCBA",
}

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}\d$")
_lock = threading.Lock()


def _default_path() -> Path:
    from src.utils.paths import DATA_DIR
    return DATA_DIR / "nse_isin_map.json"


class CorruptMap(Exception):
    """The saved map exists but cannot be read."""


def _read(path: Path, strict: bool = False) -> Dict[str, Any]:
    """The saved map; {} when there is none. A file that exists but does not
    parse reads as {} for lookups, and raises CorruptMap when `strict`, so a
    write never replaces a damaged map with only the new entries."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        if strict:
            raise CorruptMap(str(e)) from e
        logger.error(f"NSE ISIN map {path} cannot be read ({e}); using the verified nine only")
        return {}
    if not isinstance(data, dict):
        if strict:
            raise CorruptMap("not a JSON object")
        return {}
    return data


@contextmanager
def _locked(path: Path):
    """One writer at a time, across threads and across processes (the
    backfill script and the running agent both write this file)."""
    with _lock:
        if fcntl is None:
            yield
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{path}.lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def _write(path: Path, data: Dict[str, Any]) -> None:
    """Write whole or not at all: a temporary file, then an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load(path: Optional[Path] = None) -> Dict[str, str]:
    """ISIN -> ticker: the verified nine plus every learned mapping."""
    learned = _read(path or _default_path())
    out = {isin: v.get("symbol") for isin, v in learned.items()
           if isinstance(v, dict) and v.get("symbol")}
    out.update(VERIFIED)  # the checked nine always win
    return out


def remember(mappings: Dict[str, str], how: str, path: Optional[Path] = None,
             from_feed: bool = False) -> Dict[str, str]:
    """Save new ISIN -> ticker mappings; returns the ones actually added.

    Never maps a ticker that already has an ISIN, or an ISIN to a second
    ticker: a conflict is logged and skipped, because a wrong mapping would
    put one company's prices under another. The one exception is
    `from_feed`: the NSE ticker feed states the mapping itself, so it
    replaces a conflicting mapping that was only inferred from prices. The
    hand-checked nine are never replaced.
    """
    path = Path(path or _default_path())
    added: Dict[str, str] = {}
    with _locked(path):
        try:
            learned = _read(path, strict=True)
        except CorruptMap as e:
            logger.error(f"NSE ISIN map {path} is damaged ({e}); not writing to it. "
                         f"Repair or remove the file to resume learning.")
            return {}
        for isin, sym in mappings.items():
            isin, sym = str(isin).strip().upper(), str(sym).strip().upper()
            if not ISIN_RE.match(isin) or not sym:
                continue
            current = {i: v.get("symbol") for i, v in learned.items()
                       if isinstance(v, dict) and v.get("symbol")}
            current.update(VERIFIED)
            by_symbol = {s: i for i, s in current.items()}
            if current.get(isin) == sym:
                continue
            clashes = {i for i in (isin, by_symbol.get(sym)) if i and i in current}
            if clashes:
                replaceable = from_feed and all(
                    i not in VERIFIED and not learned.get(i, {}).get("from_feed") for i in clashes)
                if not replaceable:
                    logger.warning(f"NSE ISIN map: {isin} -> {sym} conflicts with "
                                   f"{current.get(isin) or by_symbol.get(sym)}; not stored")
                    continue
                for i in clashes:
                    logger.warning(f"NSE ISIN map: the ticker feed maps {isin} to {sym}; "
                                   f"replacing {i} -> {current[i]} ({learned[i].get('how')})")
                    learned.pop(i, None)
            learned[isin] = {"symbol": sym, "how": how, **({"from_feed": True} if from_feed else {})}
            added[isin] = sym
        if added:
            try:
                _write(path, learned)
                logger.info(f"NSE ISIN map: learned {len(added)} ({how}): {added}")
            except OSError as e:
                logger.error(f"NSE ISIN map: could not save: {e}")
                return {}
    return added
