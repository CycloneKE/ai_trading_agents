import subprocess
import sys
import os
import time
import tempfile
import json
import base64
import urllib.request

# Known monitoring password for the smoke run; the agent's /health endpoint is
# authenticated (it exposes component statuses), so the probe authenticates too.
MONITORING_PASSWORD = 'smoke-test-password'


def wait_for_health(url, timeout=20, password=None):
    req_headers = {}
    if password:
        token = base64.b64encode(f":{password}".encode()).decode()
        req_headers['Authorization'] = f'Basic {token}'
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=req_headers)
            # Per-request timeout must exceed the server's health_check_timeout
            # (a blocking component check makes /health take up to that long).
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status == 200:
                    data = resp.read().decode()
                    if '"status": "ok"' in data or '"status": "running"' in data:
                        return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def test_agent_smoke_start_and_health_check():
    # Create a temporary config enabling monitoring on a fixed port
    base_config_path = os.path.join('config', 'run_config.json')
    with open(base_config_path, 'r') as f:
        base = json.load(f)

    # Use a dedicated port for the smoke test
    smoke_port = 18081
    base['monitoring'] = base.get('monitoring', {})
    base['monitoring']['enabled'] = True
    base['monitoring']['port'] = smoke_port
    # Force fallback-only mode to avoid hitting external APIs during smoke tests
    base.setdefault('data_manager', {})['use_fallback_only'] = True
    # validate_config requires a non-empty strategies section; run_config.json
    # ships with an empty one, so inject a minimal valid strategy for the smoke run.
    if not base.get('strategies'):
        base['strategies'] = {'momentum': {'type': 'technical', 'enabled': True, 'weight': 1.0}}

    tmp_fd, tmp_path = tempfile.mkstemp(prefix='smoke_config_', suffix='.json')
    os.close(tmp_fd)
    with open(tmp_path, 'w') as f:
        json.dump(base, f)

    env = os.environ.copy()
    # Provide minimal dummy secrets so _validate_secrets passes
    env.update({
        'COINBASE_API_KEY': 'dummy',
        'COINBASE_API_SECRET': 'dummy',
        'COINBASE_PASSPHRASE': 'dummy',
        'OANDA_API_KEY': 'dummy',
        'OANDA_ACCOUNT_ID': 'dummy',
        'TRADING_ALPHA_VANTAGE_API_KEY': 'dummy',
        'TRADING_FMP_API_KEY': 'dummy',
        'TRADING_FINNHUB_API_KEY': 'dummy',
        'PYTHONUNBUFFERED': '1',
        # Pin the monitoring password so the /health probe can authenticate
        # deterministically (takes precedence over SECRET_KEY from .env).
        'MONITORING_PASSWORD': MONITORING_PASSWORD,
    })

    # The entry point moved to src/agent/main.py and uses absolute ``src.``
    # imports, so it must run as a module from the repo root. Put the repo root
    # and the bare-name source dirs on PYTHONPATH so any transitive flat imports
    # (e.g. live_risk_manager) also resolve in the child process.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    extra_paths = [
        repo_root,
        os.path.join(repo_root, 'src', 'agent'),
        os.path.join(repo_root, 'scripts'),
    ]
    env['PYTHONPATH'] = os.pathsep.join(extra_paths + [env.get('PYTHONPATH', '')]).rstrip(os.pathsep)

    # Start the agent process
    proc = subprocess.Popen([sys.executable, '-m', 'src.agent.main', '--config', tmp_path],
                            cwd=repo_root, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        url = f'http://127.0.0.1:{smoke_port}/health'
        # 90s: a cold interpreter (first run after boot) can spend 20s+ just on
        # imports before the monitoring server binds; 25s flaked on cold starts.
        ok = wait_for_health(url, timeout=90, password=MONITORING_PASSWORD)
        # Capture some output for debugging if failed
        if not ok:
            out = ''
            try:
                out, _ = proc.communicate(timeout=1)
            except Exception:
                proc.kill()
            raise AssertionError(f"Health check did not become OK within timeout. Agent output:\n{out}")
        assert ok
    finally:
        # Terminate the process
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        # Clean up temp config
        try:
            os.remove(tmp_path)
        except Exception:
            pass