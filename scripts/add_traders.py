"""Bulk-provision trader accounts for dashboard/API access.

Adds one users.json entry per trader with a bcrypt-hashed, randomly generated
password, and prints each credential exactly once so it can be handed to the
trader out-of-band. Existing users are never overwritten unless --reset is
given.

Usage:
    python scripts/add_traders.py alice bob carol
    python scripts/add_traders.py alice --reset      # regenerate alice's password
    python scripts/add_traders.py --list             # show existing usernames
"""

import argparse
import json
import os
import secrets
import string
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt

USERS_FILE = 'users.json'
PASSWORD_ALPHABET = string.ascii_letters + string.digits
PASSWORD_LENGTH = 16


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def generate_password() -> str:
    return ''.join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_users(users: dict) -> None:
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('usernames', nargs='*', help='trader usernames to add')
    parser.add_argument('--reset', action='store_true',
                        help='regenerate passwords for usernames that already exist')
    parser.add_argument('--list', action='store_true', dest='list_users',
                        help='list existing usernames and exit')
    args = parser.parse_args()

    users = load_users()

    if args.list_users:
        for name in sorted(users):
            print(name)
        return 0

    if not args.usernames:
        parser.error('provide at least one username (or --list)')

    created = []
    for username in args.usernames:
        username = username.strip().lower()
        if not username.replace('_', '').replace('-', '').isalnum():
            print(f"SKIP  {username}: usernames must be alphanumeric (plus - or _)")
            continue
        if username in users and not args.reset:
            print(f"SKIP  {username}: already exists (use --reset to regenerate password)")
            continue
        password = generate_password()
        users[username] = hash_password(password)
        created.append((username, password))

    if not created:
        print('No accounts changed.')
        return 1

    save_users(users)

    print(f"\n{len(created)} account(s) written to {USERS_FILE}.")
    print('Credentials below are shown ONCE - hand each to its trader privately:\n')
    for username, password in created:
        print(f"    {username}: {password}")
    print('\nTraders log in with these at the dashboard; tokens expire after 24h.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
