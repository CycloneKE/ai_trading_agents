"""
Authentication and authorization module for the AI Trading Agent API.
"""

import jwt
from functools import wraps
from flask import request, jsonify, g
import bcrypt
import json
import os
from datetime import datetime, timedelta

# Roles. 'operator' is full control (kill switch, config, decision internals);
# 'viewer' sees outcomes only (positions, P&L, price tape).
ROLES = ('operator', 'viewer')
DEFAULT_ROLE = 'operator'  # legacy bare-hash accounts are operators

# Configuration
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise RuntimeError("CRITICAL: SECRET_KEY environment variable is not set. Cannot start without JWT secret.")
# Absolute path so auth works regardless of the process working directory
# (systemd/Docker may set a different CWD than the repo root). Override with
# USERS_FILE env for a path outside the image.
USERS_FILE = os.environ.get(
    'USERS_FILE',
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'users.json'))

def hash_password(password: str) -> str:
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(stored_password: str, provided_password: str) -> bool:
    """Verifies a provided password against a stored hash."""
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))


def user_hash_and_role(record) -> tuple:
    """A users.json entry is either a bare bcrypt hash (legacy -> operator)
    or {'password': hash, 'role': ...}. Returns (hash, role)."""
    if isinstance(record, dict):
        role = record.get('role', DEFAULT_ROLE)
        return record.get('password', ''), (role if role in ROLES else 'viewer')
    return record, DEFAULT_ROLE


def get_user_role(username: str) -> str:
    """Authoritative current role for a username, read from the file so role
    changes take effect without forcing a re-login. 'viewer' if unknown."""
    try:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
        if username in users:
            return user_hash_and_role(users[username])[1]
    except Exception:
        pass
    return 'viewer'

# Token lifetime: default 8h (one trading session). Override with
# TOKEN_TTL_HOURS; shorter is safer on a shared LAN without TLS.
TOKEN_TTL_HOURS = float(os.environ.get('TOKEN_TTL_HOURS', '8'))


def create_token(username: str, role: str = DEFAULT_ROLE) -> str:
    """Creates a JWT token for a user, carrying their role."""
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS),
        'iat': datetime.utcnow(),
        'sub': username,
        'role': role if role in ROLES else 'viewer',
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def token_required(f):
    """Decorator to protect routes with JWT authentication. Stashes the
    caller's identity + current role on flask.g for downstream checks."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            try:
                token = request.headers['Authorization'].split(" ")[1]
            except IndexError:
                return jsonify({'message': 'Token is missing!'}), 401

        if not token:
            return jsonify({'message': 'Token is missing!'}), 401

        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])

            with open(USERS_FILE, 'r') as uf:
                users = json.load(uf)
            if data['sub'] not in users:
                return jsonify({'message': 'User not found!'}), 401

            # Role read from the file (authoritative/current), not the token,
            # so revoking a guest to viewer takes effect immediately.
            g.current_user = data['sub']
            g.current_role = get_user_role(data['sub'])

        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(*args, **kwargs)

    return decorated


def role_required(*allowed_roles):
    """Gate a route to specific roles. Must wrap a token_required'd route so
    g.current_role is set. Fails closed (403) for anything not allowed."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if getattr(g, 'current_role', None) not in allowed_roles:
                return jsonify({'error': 'Insufficient privileges for this action'}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator
