import os
import importlib
from feature_flags import is_enabled, set_flag, clear_flags


def test_default_flag_behavior(monkeypatch):
    clear_flags()
    # ensure env var not set
    monkeypatch.delenv('FF_TESTFLAG', raising=False)
    assert is_enabled('testflag', default=False) is False
    assert is_enabled('testflag', default=True) is True


def test_set_flag_runtime():
    clear_flags()
    set_flag('hello', True)
    assert is_enabled('hello') is True
    set_flag('hello', False)
    assert is_enabled('hello') is False


def test_env_var_override(monkeypatch):
    clear_flags()
    monkeypatch.setenv('FF_WHATEVER', 'true')
    assert is_enabled('whatever') is True
    monkeypatch.setenv('FF_WHATEVER', '0')
    assert is_enabled('whatever') is False
