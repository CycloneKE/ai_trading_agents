import subprocess
import sys
import os
import time
import tempfile
import json
import urllib.request


def wait_for_health(url, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
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
        'PYTHONUNBUFFERED': '1'
    })

    # Start the agent process
    proc = subprocess.Popen([sys.executable, 'main.py', '--config', tmp_path], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        url = f'http://127.0.0.1:{smoke_port}/health'
        ok = wait_for_health(url, timeout=25)
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