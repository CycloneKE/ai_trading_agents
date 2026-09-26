"""The dashboard's source fingerprint (src/utils/build_info.py), which lets
the dashboard tell when it was built from other code than the server."""
import shutil
import subprocess
from pathlib import Path

import pytest

from src.utils import build_info

ROOT = Path(__file__).resolve().parent.parent


def _tree(tmp_path):
    for rel, text in {'components/A.js': 'a', 'components/views/B.js': 'b', 'pages/index.js': 'i',
                      'utils/apiBase.js': 'u', 'node_modules/x/index.js': 'ignored',
                      'components/notes.md': 'ignored'}.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    return tmp_path


def test_the_fingerprint_covers_the_dashboard_source_only(tmp_path):
    root = _tree(tmp_path)
    first = build_info.source_hash(root)
    assert len(first) == 12 and build_info.source_hash(root) == first
    (root / 'node_modules/x/index.js').write_text('changed')
    (root / 'components/notes.md').write_text('changed')
    assert build_info.source_hash(root) == first               # not dashboard code
    (root / 'components/views/B.js').write_text('b2')
    assert build_info.source_hash(root) != first               # dashboard code changed
    assert build_info.source_hash(tmp_path / 'missing') is None


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_the_dashboard_build_computes_the_same_fingerprint(tmp_path):
    root = _tree(tmp_path)
    shutil.copy(ROOT / 'frontend' / 'next.config.js', root / 'next.config.js')
    out = subprocess.run(['node', '-e', "console.log(require('./next.config.js').env.NEXT_PUBLIC_SOURCE_HASH)"],
                         cwd=root, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == build_info.source_hash(root)


def test_the_server_reports_the_fingerprint():
    assert build_info.dashboard_source_hash() == build_info.source_hash()
