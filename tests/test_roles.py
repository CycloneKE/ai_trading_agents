"""Tests for role parsing and gating (operator vs viewer)."""
import os
import sys

os.environ.setdefault('SECRET_KEY', 'test-secret-for-roles')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import auth


def test_legacy_bare_hash_is_operator():
    h, role = auth.user_hash_and_role('$2b$12$somehash')
    assert h == '$2b$12$somehash'
    assert role == 'operator'


def test_dict_record_carries_role():
    h, role = auth.user_hash_and_role({'password': '$2b$hash', 'role': 'viewer'})
    assert h == '$2b$hash'
    assert role == 'viewer'


def test_unknown_role_downgraded_to_viewer():
    _, role = auth.user_hash_and_role({'password': 'x', 'role': 'superadmin'})
    assert role == 'viewer'


def test_token_carries_role_and_roundtrips():
    import jwt
    tok = auth.create_token('joe', 'viewer')
    decoded = jwt.decode(tok, auth.SECRET_KEY, algorithms=['HS256'])
    assert decoded['sub'] == 'joe'
    assert decoded['role'] == 'viewer'


def test_create_token_sanitizes_bad_role():
    import jwt
    tok = auth.create_token('joe', 'root')
    decoded = jwt.decode(tok, auth.SECRET_KEY, algorithms=['HS256'])
    assert decoded['role'] == 'viewer'


def test_role_required_blocks_and_allows():
    from flask import Flask, g, jsonify
    app = Flask(__name__)

    @app.route('/op')
    @auth.role_required('operator')
    def op_only():
        return jsonify(ok=True)

    # viewer -> 403
    with app.test_request_context('/op'):
        g.current_role = 'viewer'
        resp = app.view_functions['op_only']()
        assert resp[1] == 403

    # operator -> passes through
    with app.test_request_context('/op'):
        g.current_role = 'operator'
        resp = app.view_functions['op_only']()
        assert resp.json['ok'] is True

    # missing role -> 403 (fails closed)
    with app.test_request_context('/op'):
        resp = app.view_functions['op_only']()
        assert resp[1] == 403
