"""The container must install what the code imports without a guard.

This exists because of a real outage-shaped bug. `src/api/auth.py` imports
`jwt` and `bcrypt` at module scope with no try/except. Both were listed in
requirements.txt and in requirements-ci.txt (added there under the comment
"Auth stack"), but never in requirements-runtime.txt, which is the only file
the Dockerfile installs.

The result: CI passed, the image built, the agent started, and every
protected API route answered 503 because AUTH_AVAILABLE was False. Nobody
could log in to the dashboard, and nothing in the build said why.

Nothing compared what the container installs against what the code needs, so
these tests do. A guarded import (try/except ImportError) is exempt by
design: those degrade on purpose and are deliberately left out of the lean
runtime image.
"""
import ast
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME_REQ = os.path.join(ROOT, 'scripts', 'requirements-runtime.txt')

# Import name -> distribution name, where they differ.
DISTRIBUTION = {
    'jwt': 'pyjwt',
    'dotenv': 'python-dotenv',
    'flask_cors': 'flask-cors',
    'prometheus_client': 'prometheus-client',
    'psycopg2': 'psycopg2-binary',
    'yaml': 'pyyaml',
    'dateutil': 'python-dateutil',
    'websocket': 'websocket-client',
    'OpenSSL': 'pyopenssl',
}

# Packages whose absence stops the deployed system doing its job, with the
# consequence spelled out so a future reader does not "tidy" one away.
CRITICAL = {
    'pyjwt': 'src/api/auth.py imports jwt unguarded; without it every '
             'protected API route returns 503 and nobody can log in',
    'bcrypt': 'src/api/auth.py imports bcrypt unguarded; same lockdown',
    'flask': 'the REST API does not exist without it',
    'flask-cors': 'the dashboard cannot call the API across origins',
    'waitress': 'api_server falls back to the Flask development server, '
                'which warns against its own use in production',
    'pandas': 'every strategy operates on DataFrames',
    'numpy': 'same',
}


def _declared():
    """Distribution names declared in requirements-runtime.txt, lower-cased."""
    names = set()
    with open(RUNTIME_REQ, encoding='utf-8') as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line or line.startswith('-'):
                continue
            name = re.split(r'[<>=!~\[;]', line, maxsplit=1)[0].strip()
            if name:
                names.add(name.lower())
    return names


def _unguarded_imports(path):
    """Top-level import names not wrapped in try/except.

    Only module-body statements count. Anything inside a Try is an optional
    dependency the code is prepared to lose, and those are deliberately
    absent from the lean runtime image.
    """
    with open(path, encoding='utf-8') as f:
        tree = ast.parse(f.read())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split('.')[0])
    return found


def _third_party(names):
    stdlib = set(sys.stdlib_module_names)
    local = {'src', 'scripts', 'tests', 'config'}
    return {n for n in names if n not in stdlib and n not in local}


# --------------------------------------------------------------- the checks

def test_requirements_file_exists_and_parses():
    assert os.path.exists(RUNTIME_REQ), (
        'The Dockerfile installs this file. It must exist.')
    assert _declared(), 'requirements-runtime.txt declared no packages at all'


@pytest.mark.parametrize('package', sorted(CRITICAL))
def test_critical_package_is_in_the_runtime_requirements(package):
    assert package in _declared(), (
        f'{package} is missing from scripts/requirements-runtime.txt, which '
        f'is what the Docker image installs. Consequence: {CRITICAL[package]}. '
        f'Note that requirements.txt and requirements-ci.txt are NOT installed '
        f'into the image, so listing it there does not help.')


def test_auth_module_dependencies_are_installed_in_the_container():
    """The specific regression: auth imports that the image did not ship."""
    auth = os.path.join(ROOT, 'src', 'api', 'auth.py')
    declared = _declared()
    missing = []
    for name in sorted(_third_party(_unguarded_imports(auth))):
        dist = DISTRIBUTION.get(name, name).lower()
        if dist not in declared:
            missing.append(f'{name} (distribution: {dist})')
    assert not missing, (
        f'src/api/auth.py imports {missing} at module scope with no '
        f'try/except, but the container does not install them. The API will '
        f'set AUTH_AVAILABLE=False and answer every protected route with 503.')


def test_api_server_dependencies_are_installed_in_the_container():
    api = os.path.join(ROOT, 'src', 'api', 'api_server.py')
    declared = _declared()
    missing = []
    for name in sorted(_third_party(_unguarded_imports(api))):
        dist = DISTRIBUTION.get(name, name).lower()
        if dist not in declared:
            missing.append(f'{name} (distribution: {dist})')
    assert not missing, (
        f'src/api/api_server.py imports {missing} at module scope with no '
        f'try/except, but the container does not install them. The API will '
        f'not start at all.')


def test_optional_heavy_packages_stay_out_of_the_runtime_image():
    """The lean image is a feature, not an oversight.

    Each of these is guarded at its import site and the system degrades
    rather than failing. Pulling them in would add gigabytes to an image that
    has to rebuild on every deploy.
    """
    declared = _declared()
    for heavy in ('torch', 'tensorflow', 'transformers', 'spacy'):
        assert heavy not in declared, (
            f'{heavy} was added to the runtime image. It is guarded at its '
            f'import site and the system runs without it; adding it makes '
            f'every deploy multi-gigabyte.')
