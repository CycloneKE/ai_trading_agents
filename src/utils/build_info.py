"""A fingerprint of the dashboard's source, so a stale dashboard can tell.

The server and the dashboard are built from the same commit into two
images. When only one of them is rebuilt, or a browser keeps an old copy
of the page, the dashboard shows features the server no longer has, or
misses ones it has, with nothing on screen to say so.

Both sides fingerprint the same files the same way: the server here at
start-up, from the dashboard source copied into its image, and the
dashboard at build time (frontend/next.config.js). The dashboard compares
the two and says so when they differ.

The fingerprint is SHA-256 over every .js file under frontend/components,
frontend/pages and frontend/utils, in sorted order of path relative to
frontend/, each as path, a zero byte, its bytes, a zero byte. The first 12
hex digits are used.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

FRONTEND = Path(__file__).resolve().parent.parent.parent / 'frontend'
FOLDERS = ('components', 'pages', 'utils')


def source_hash(root: Path = FRONTEND) -> Optional[str]:
    files = sorted(p.relative_to(root).as_posix()
                   for folder in FOLDERS if (root / folder).is_dir()
                   for p in (root / folder).rglob('*.js') if p.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    for rel in files:
        digest.update(rel.encode() + b'\0' + (root / rel).read_bytes() + b'\0')
    return digest.hexdigest()[:12]


@lru_cache(maxsize=1)
def dashboard_source_hash() -> Optional[str]:
    try:
        return source_hash()
    except OSError:
        return None
