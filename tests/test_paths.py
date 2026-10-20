# tests/test_paths.py
import os
from pathlib import Path


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
