# [site-login] - START
"""Sign-in of the review site on the LAN: first visit with the setup code, sign-in, sessions, account
page, password change, and what stays the same for agents (the access token in a header) and on
127.0.0.1 (no sign-in)."""
from __future__ import annotations

import pytest

from nnslr_tools.pages import asset

from nnslr_tools import review_auth
from review_helpers import ADMIN, PASSWORD, ROUTE, TOKEN, RunningServer, json_of, make_report, sign_in


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    # The real cost takes ~0.1 s per check.
    monkeypatch.setattr(review_auth, 'SCRYPT_N', 2 ** 10)


@pytest.fixture
def lan(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path, token=TOKEN, require_token=True)
    yield server, tmp_path
    server.close()


def cookie_of(response) -> dict:
    value = response.getheader('Set-Cookie').split(';', 1)[0]
    return {'Cookie': value}


def test_first_visit_creates_the_administrator_with_the_setup_code(lan):
    server, root = lan
    status, response, data = server.request('GET', '/api/auth/state')
    assert json_of(data) == {'configured': False, 'signed_in': False, 'local': False}
    status, response, data = server.request('POST', '/api/auth/setup/check', body={'code': 'wrong-code-000000'})
    assert status == 403 and 'Wrong setup code' in json_of(data)['message']
    assert server.request('POST', '/api/auth/setup/check', body={'code': TOKEN})[0] == 200
    status, response, data = server.request('POST', '/api/auth/setup',
                                            body={'code': TOKEN, 'user': ADMIN, 'password': 'short'})
    assert status == 400 and '12 characters' in json_of(data)['message']
    status, response, data = server.request('POST', '/api/auth/setup',
                                            body={'code': TOKEN, 'user': ADMIN, 'password': PASSWORD})
    assert status == 200
    cookie = response.getheader('Set-Cookie')
    assert cookie.startswith(server.app.cookie_name + '=') and 'HttpOnly' in cookie and 'SameSite=Strict' in cookie
    assert TOKEN not in cookie
    session = cookie_of(response)
    assert server.request('GET', '/api/routes', headers=session)[0] == 200
    # The setup code works once: then the administrator exists.
    status, response, data = server.request('POST', '/api/auth/setup',
                                            body={'code': TOKEN, 'user': 'other', 'password': PASSWORD})
    assert status == 409
    files = sorted(path.name for path in (root / 'web' / 'auth').iterdir())
    assert files == ['admin.json', 'sessions.json']
    assert ((root / 'web' / 'auth' / 'admin.json').stat().st_mode & 0o777) == 0o600
    assert PASSWORD not in (root / 'web' / 'auth' / 'admin.json').read_text()


def test_sign_in_sign_out_and_wrong_attempts(lan):
    server, root = lan
    sign_in(server)
    status, response, data = server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': 'nope nope nope'})
    assert status == 401 and json_of(data)['message'] == 'Wrong name or password.'
    status, response, data = server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': PASSWORD})
    assert status == 200
    session = cookie_of(response)
    state = json_of(server.request('GET', '/api/auth/state', headers=session)[2])
    assert state['signed_in'] and state['user'] == ADMIN and state['sessions'] == 2
    status, response, data = server.request('GET', '/account', headers=session)
    # The account page is a Vue app: its form is in the account script.
    assert status == 200 and asset('js/account.js').encode() in data and b'password-form' in data
    status, response, data = server.request('POST', '/api/auth/logout', body={}, headers=session)
    assert status == 200 and 'Max-Age=0' in response.getheader('Set-Cookie')
    assert server.request('GET', '/api/routes', headers=session)[0] == 401
    # The good sign-in above cleared this address: five new wrong answers lock it.
    for _ in range(5):
        server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': 'wrong password'})
    status, response, data = server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': PASSWORD})
    assert status == 429 and 'Too many wrong attempts' in json_of(data)['message']


def test_password_change_signs_out_every_other_browser(lan):
    server, root = lan
    first = sign_in(server)
    status, response, data = server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': PASSWORD})
    second = cookie_of(response)
    status, response, data = server.request('POST', '/api/auth/password',
                                            body={'current': 'wrong', 'password': 'a brand new password'},
                                            headers=second)
    assert status == 403
    status, response, data = server.request('POST', '/api/auth/password',
                                            body={'current': PASSWORD, 'password': 'a brand new password'},
                                            headers=second)
    assert status == 200 and json_of(data)['state']['sessions'] == 1
    assert server.request('GET', '/api/routes', headers=first)[0] == 401
    assert server.request('GET', '/api/routes', headers=second)[0] == 401
    assert server.request('GET', '/api/routes', headers=cookie_of(response))[0] == 200


def test_sign_in_calls_only_from_this_site_and_as_json(lan):
    server, root = lan
    status, response, data = server.request('POST', '/api/auth/setup/check', body={'code': TOKEN},
                                            headers={'Origin': 'http://evil.example'})
    assert status == 403
    status, response, data = server.request('POST', '/api/auth/setup/check', body=b'code=x',
                                            headers={'Content-Type': 'application/x-www-form-urlencoded'})
    assert status == 415
    assert server.request('GET', '/api/auth/login')[0] == 401


def test_agents_keep_the_token_in_a_header(lan):
    server, root = lan
    sign_in(server)
    status, response, data = server.request('GET', f'/route/{ROUTE}/', headers={'X-Review-Token': TOKEN})
    assert status == 200 and b'"server": true' in data
    status, response, data = server.request('GET', '/api/v1/routes', headers={'Authorization': 'Bearer ' + TOKEN})
    assert status == 200
    # A browser session is not an API key.
    status, response, data = server.request('GET', '/api/v1/routes', headers=sign_in_cookie(server))
    assert status == 401


def sign_in_cookie(server) -> dict:
    status, response, data = server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': PASSWORD})
    return cookie_of(response)


def test_loopback_has_no_sign_in(tmp_path):
    make_report(tmp_path)
    server = RunningServer(tmp_path)
    try:
        assert server.app.auth is None
        assert server.request('GET', '/')[0] == 200
        assert json_of(server.request('GET', '/api/auth/state')[2])['local'] is True
        # The Account & Settings app, as on the LAN: on 127.0.0.1 it shows the settings without the sign-in cards.
        status, response, data = server.request('GET', '/account')
        assert status == 200 and asset('js/account.js').encode() in data and b'id="app"' in data
        assert server.request('POST', '/api/auth/login', body={'user': ADMIN, 'password': PASSWORD})[0] == 404
    finally:
        server.close()
# [site-login] - END
