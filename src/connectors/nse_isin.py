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
learned, so they survive redeploys and can be checked.
"""
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, Optional

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


def _read(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load(path: Optional[Path] = None) -> Dict[str, str]:
    """ISIN -> ticker: the verified nine plus every learned mapping."""
    learned = _read(path or _default_path())
    out = {isin: v.get("symbol") for isin, v in learned.items()
           if isinstance(v, dict) and v.get("symbol")}
    out.update(VERIFIED)  # the checked nine always win
    return out


def remember(mappings: Dict[str, str], how: str, path: Optional[Path] = None) -> Dict[str, str]:
    """Save new ISIN -> ticker mappings; returns the ones actually added.

    Never overwrites a mapping, and never maps a ticker that already has an
    ISIN or an ISIN to a second ticker: a conflict is logged and skipped,
    because a wrong mapping would put one company's prices under another.
    """
    path = path or _default_path()
    added: Dict[str, str] = {}
    with _lock:
        learned = _read(path)
        current = load(path)
        by_symbol = {sym: isin for isin, sym in current.items()}
        for isin, sym in mappings.items():
            isin, sym = str(isin).strip().upper(), str(sym).strip().upper()
            if not ISIN_RE.match(isin) or not sym:
                continue
            if current.get(isin) == sym:
                continue
            if isin in current or sym in by_symbol:
                logger.warning(f"NSE ISIN map: {isin} -> {sym} conflicts with "
                               f"{current.get(isin) or by_symbol.get(sym)}; not stored")
                continue
            learned[isin] = {"symbol": sym, "how": how}
            current[isin], by_symbol[sym] = sym, isin
            added[isin] = sym
        if added:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(learned, indent=2, sort_keys=True))
                logger.info(f"NSE ISIN map: learned {len(added)} ({how}): {added}")
            except OSError as e:
                logger.error(f"NSE ISIN map: could not save: {e}")
    return added
