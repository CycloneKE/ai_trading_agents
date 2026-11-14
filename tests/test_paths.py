# tests/test_paths.py
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.real_data_dir


def test_data_dir_is_absolute_and_inside_repo():
    from src.utils.paths import DATA_DIR, PROJECT_ROOT
    assert DATA_DIR.is_absolute()
    assert DATA_DIR == PROJECT_ROOT / 'data'
    assert (PROJECT_ROOT / 'src').is_dir()  # sanity: root really is the repo


def test_data_dir_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import importlib
    import src.utils.paths as p
    importlib.reload(p)
    assert str(tmp_path) not in str(p.DATA_DIR)


def test_agent_data_dir_moves_the_data_folder(tmp_path, monkeypatch):
    """The smoke test runs a whole agent against a temporary folder this way."""
    import importlib
    import src.utils.paths as p
    monkeypatch.setenv('AGENT_DATA_DIR', str(tmp_path / 'elsewhere'))
    try:
        importlib.reload(p)
        assert p.DATA_DIR == (tmp_path / 'elsewhere').resolve() and p.DATA_DIR.is_dir()
    finally:
        monkeypatch.delenv('AGENT_DATA_DIR')
        importlib.reload(p)
    assert p.DATA_DIR == p.PROJECT_ROOT / 'data'
