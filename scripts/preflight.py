"""Pre-session readiness gate for trader testing sessions.

Runs every check a session operator would otherwise do by hand and prints a
single PASS/FAIL verdict. Run it before traders join:

    python scripts/preflight.py            # static checks only
    python scripts/preflight.py --live     # also probe a running backend

Exit code 0 = ready, 1 = at least one blocking check failed.
"""

import argparse
import base64
import json
import os
import socket
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

if os.name == 'nt':
    os.system('')  # enables ANSI escape processing in the legacy console
GREEN, RED, YELLOW, RESET = '\033[92m', '\033[91m', '\033[93m', '\033[0m'

REQUIRED_ENV = [
    ('SECRET_KEY', 'JWT signing key - API fails closed without it'),
]
RECOMMENDED_ENV = [
    ('TRADING_FMP_API_KEY', 'FMP market data'),
    ('TRADING_ALPHA_VANTAGE_API_KEY', 'Alpha Vantage market data'),
    ('TRADING_FINNHUB_API_KEY', 'Finnhub market data'),
    ('TRADING_ALPACA_API_KEY', 'Alpaca paper broker'),
    ('TRADING_ALPACA_API_SECRET', 'Alpaca paper broker'),
]

results = []  # (level, name, ok, detail); level 'block' fails the run


def check(level, name, ok, detail=''):
    results.append((level, name, ok, detail))
    if ok:
        mark = f'{GREEN}PASS{RESET}'
    elif level == 'block':
        mark = f'{RED}FAIL{RESET}'
    else:
        mark = f'{YELLOW}WARN{RESET}'
    line = f'  [{mark}] {name}'
    if detail and not ok:
        line += f'  ({detail})'
    print(line)


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


def http_get(url, auth_password=None, timeout=6):
    headers = {}
    if auth_password:
        token = base64.b64encode(f':{auth_password}'.encode()).decode()
        headers['Authorization'] = f'Basic {token}'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def static_checks():
    print('\n-- Environment --')
    env_path = os.path.join(ROOT, '.env')
    check('block', '.env exists', os.path.exists(env_path))

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, '.env'))
    except ImportError:
        pass

    for var, why in REQUIRED_ENV:
        check('block', f'{var} set', bool(os.environ.get(var)), why)
    for var, why in RECOMMENDED_ENV:
        check('warn', f'{var} set', bool(os.environ.get(var)), why)

    print('\n-- Configuration --')
    cfg_path = os.path.join(ROOT, 'config', 'config.json')
    cfg = {}
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        check('block', 'config/config.json parses', True)
    except Exception as e:
        check('block', 'config/config.json parses', False, str(e))

    if cfg:
        brokers = cfg.get('brokers', {})
        live = [
            name for name, b in brokers.items()
            if b.get('enabled', True)
            and b.get('type') != 'paper'
            and not (b.get('paper') or b.get('is_paper'))
        ]
        check('block', 'all enabled brokers are paper-mode', not live,
              f'LIVE brokers configured: {", ".join(live)}')
        check('warn', 'API enabled in config', cfg.get('api', {}).get('enabled', False))

    print('\n-- Accounts --')
    users_path = os.path.join(ROOT, 'users.json')
    try:
        with open(users_path) as f:
            users = json.load(f)
        check('block', 'users.json has at least one account', len(users) > 0)
        traders = [u for u in users if u != 'admin']
        check('warn', 'trader accounts exist (besides admin)', len(traders) > 0,
              'run: python scripts/add_traders.py <names...>')
    except Exception as e:
        check('block', 'users.json readable', False, str(e))

    print('\n-- Frontend --')
    build_id = os.path.join(ROOT, 'frontend', '.next', 'BUILD_ID')
    check('block', 'frontend production build exists', os.path.exists(build_id),
          'run: cd frontend && npm run build')
    check('block', 'frontend deps installed',
          os.path.isdir(os.path.join(ROOT, 'frontend', 'node_modules')),
          'run: cd frontend && npm install')


def live_checks():
    print('\n-- Live backend --')
    api_up = port_in_use(5001)
    check('block', 'API listening on 5001', api_up, 'start the backend first')
    if api_up:
        try:
            status, body = http_get('http://127.0.0.1:5001/api/health')
            check('block', 'API /api/health returns ok',
                  status == 200 and '"ok"' in body)
        except Exception as e:
            check('block', 'API /api/health returns ok', False, str(e))

    mon_up = port_in_use(8080)
    check('warn', 'monitoring listening on 8080', mon_up)
    if mon_up:
        password = os.environ.get('MONITORING_PASSWORD', os.environ.get('SECRET_KEY'))
        try:
            status, body = http_get('http://127.0.0.1:8080/health', auth_password=password)
            ready = status == 200 and '"ready": true' in body
            check('warn', 'monitoring reports ready', ready,
                  'components may still be starting')
        except Exception as e:
            check('warn', 'monitoring reports ready', False, str(e))

    check('warn', 'dashboard listening on 3001', port_in_use(3001),
          'run: cd frontend && npm run start')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--live', action='store_true',
                        help='also probe the running backend/dashboard')
    args = parser.parse_args()

    print('Premarket preflight check')
    static_checks()
    if args.live:
        live_checks()

    blockers = [name for level, name, ok, _ in results if level == 'block' and not ok]
    warns = [name for level, name, ok, _ in results if level == 'warn' and not ok]

    print()
    if blockers:
        print(f'{RED}NOT READY{RESET} - {len(blockers)} blocking issue(s):')
        for name in blockers:
            print(f'  - {name}')
        return 1
    verdict = f'{GREEN}READY{RESET}'
    if warns:
        verdict += f' ({len(warns)} warning(s) - see above)'
    if not args.live:
        verdict += '  [static checks only; rerun with --live once the stack is up]'
    print(verdict)
    return 0


if __name__ == '__main__':
    sys.exit(main())
