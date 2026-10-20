"""`datetime.utcnow()` must not reappear in the live source tree.

It is deprecated in Python 3.12+ and scheduled for removal, so on a future
interpreter every timestamp in the system becomes an AttributeError — in a
process that places orders. The suite emitted 111 DeprecationWarnings for it.

The replacement is src.utils.timeutil.utc_now(), which is byte-format
identical to the old output (see tests/test_timeutil.py for why that
matters: the journals sort ISO strings as TEXT).

Detection is AST-based rather than textual so prose mentioning the old API
in a docstring — like this one — does not trip the check.
"""

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'


def _utcnow_calls(path):
    """Line numbers referencing `<something>.utcnow`.

    Matches bare attribute references as well as calls. A reference is just
    as deprecated and strictly harder to spot: `task_queue.Task` had
    `created_at: datetime = field(default_factory=datetime.utcnow)`, where the
    call happens inside dataclass-generated code and surfaces only as an
    anonymous `<string>:9` warning.
    """
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except SyntaxError:
        return []
    return sorted({node.lineno for node in ast.walk(tree)
                   if isinstance(node, ast.Attribute) and node.attr == 'utcnow'})


def all_source_files():
    return sorted(p for p in SRC.rglob('*.py') if '__pycache__' not in p.parts)


def test_source_tree_is_free_of_deprecated_utcnow():
    offenders = {}
    for path in all_source_files():
        hits = _utcnow_calls(path)
        if hits:
            offenders[str(path.relative_to(SRC.parent))] = hits
    assert not offenders, (
        'datetime.utcnow() is deprecated; use src.utils.timeutil.utc_now():\n'
        + '\n'.join(f'  {f}: lines {ls}' for f, ls in sorted(offenders.items())))


def test_timeutil_is_the_single_source_of_utc_time():
    """The helper itself is allowed no deprecated call either."""
    assert _utcnow_calls(SRC / 'utils' / 'timeutil.py') == []


@pytest.mark.parametrize('module', [
    'src.agent.order_journal',
    'src.agent.decision_journal',
    'src.agent.exposure',
    'src.agent.self_assessment',
    'src.connectors.base_broker',
])
def test_money_path_modules_import_cleanly_without_deprecation(module):
    """Importing these must not warn — they are on the order path."""
    import importlib
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        importlib.import_module(module)
