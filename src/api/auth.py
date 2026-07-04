"""
Authentication and authorization module for the AI Trading Agent API.
"""

import jwt
from functools import wraps
from flask import request, jsonify
import bcrypt
import json
import os
from datetime import datetime, timedelta

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

# Token lifetime: default 8h (one trading session). Override with
# TOKEN_TTL_HOURS; shorter is safer on a shared LAN without TLS.
TOKEN_TTL_HOURS = float(os.environ.get('TOKEN_TTL_HOURS', '8'))


def create_token(username: str) -> str:
    """Creates a JWT token for a user."""
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS),
        'iat': datetime.utcnow(),
        'sub': username
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

def token_required(f):
    """Decorator to protect routes with JWT authentication."""
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
            
            # You can add a check here to see if the user exists in your database
            with open(USERS_FILE, 'r') as uf:
                users = json.load(uf)
            if data['sub'] not in users:
                return jsonify({'message': 'User not found!'}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired!'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Invalid token!'}), 401

        return f(*args, **kwargs)

    return decorated
