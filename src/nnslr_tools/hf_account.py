"""The Hugging Face account the portal publishes with (Settings > Hugging Face; scripts/publish_networks.sh).

A person pastes a token in the wizard of Settings; it is checked on Hugging Face (whose it is, whether it can write)
and kept in <data root>/web/private/huggingface.json, mode 600 in a folder of mode 700, outside git. It never goes
back to a browser, into a log, a report or a command line: a publishing job gets it in its environment (HF_TOKEN).
Without one, the job uses the Hugging Face login of this server's account (``hf auth login``), when there is one.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from nnslr_tools.review_auth import read_private, write_private
from nnslr_tools.review_db import ReviewRejected

FILE = Path('web') / 'private' / 'huggingface.json'
TOKEN = re.compile(r'hf_[A-Za-z0-9]{20,200}')
WHOAMI = 'https://huggingface.co/api/whoami-v2'
# Where the Hugging Face library keeps the login of `hf auth login`.
SERVER_LOGIN = Path(os.environ.get('HF_TOKEN_PATH') or Path(os.environ.get('HF_HOME') or Path.home() / '.cache' /
                                                           'huggingface') / 'token')
# The page where a person creates a token that can write.
NEW_TOKEN_URL = 'https://huggingface.co/settings/tokens/new?tokenType=write'


def path(root: Path) -> Path:
    return Path(root) / FILE


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _load(root: Path) -> dict:
    text = read_private(path(root))
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) and isinstance(data.get('token'), str) else {}


def can_write(answer: dict) -> bool:
    """Whether a whoami-v2 answer is of a token that can create and write the user's repositories: a classic write
    token, or a fine-grained one with repo.write on the user."""
    access = (answer.get('auth') or {}).get('accessToken') or {}
    role = access.get('role')
    if role == 'write':
        return True
    if role != 'fineGrained':
        return False
    grained = access.get('fineGrained') or {}
    for scope in grained.get('scoped') or []:
        entity = scope.get('entity') or {}
        if entity.get('type') == 'user' and entity.get('name') == answer.get('name') \
                and 'repo.write' in (scope.get('permissions') or []):
            return True
    return False


def check_token(token: str, *, timeout: float = 15) -> dict:
    """{'user', 'role', 'can_write'} of a token, asked to Hugging Face (whoami-v2). The token is in the header only."""
    request = urllib.request.Request(WHOAMI, headers={'Authorization': 'Bearer ' + token,
                                                      'User-Agent': 'sunnypilot-speedlimits'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.loads(response.read(1 << 20).decode('utf-8'))
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ReviewRejected(400, 'invalid_token', 'Hugging Face does not accept this token: copy it again, '
                                                       'whole.') from None
        raise ReviewRejected(502, 'hf_error', f'Hugging Face answered HTTP {exc.code}: try again later.') from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise ReviewRejected(502, 'hf_unreachable', 'Hugging Face cannot be reached from this server now: '
                                                    'try again later.') from None
    if not isinstance(answer, dict) or not isinstance(answer.get('name'), str):
        raise ReviewRejected(502, 'hf_error', 'Hugging Face gave an answer this site does not understand.')
    access = (answer.get('auth') or {}).get('accessToken') or {}
    return {'user': answer['name'], 'role': access.get('role') or 'unknown', 'can_write': can_write(answer)}


def public_state(root: Path) -> dict:
    """What the page may know: whose token is kept and what it can do, or whether the server has a login of its own.
    Never the token."""
    data = _load(root)
    if data:
        return {'connected': True, 'source': 'portal', 'user': data.get('user'), 'role': data.get('role'),
                'can_write': bool(data.get('can_write')), 'checked_at': data.get('checked_at'),
                'new_token_url': NEW_TOKEN_URL}
    server = SERVER_LOGIN.is_file() and SERVER_LOGIN.stat().st_size > 0
    return {'connected': server, 'source': 'server' if server else None, 'user': None, 'role': None,
            'can_write': None, 'checked_at': None, 'new_token_url': NEW_TOKEN_URL}


def save(root: Path, token) -> dict:
    """Check a pasted token on Hugging Face and keep it: refused when it is not a token, not accepted, or read-only."""
    if not isinstance(token, str) or not TOKEN.fullmatch(token.strip()):
        raise ReviewRejected(400, 'invalid_token', 'That is not a Hugging Face token: it starts with hf_.')
    token = token.strip()
    checked = check_token(token)
    if not checked['can_write']:
        raise ReviewRejected(400, 'read_only_token', f'This token of {checked["user"]} can only read: create one '
                                                     'with Write access.')
    write_private(path(root), json.dumps({'token': token, **checked, 'checked_at': _now()}, indent=2) + '\n')
    try:
        os.chmod(path(root).parent, 0o700)
    except OSError:
        pass
    return public_state(root)


def remove(root: Path) -> dict:
    try:
        path(root).unlink()
    except FileNotFoundError:
        pass
    return public_state(root)


def token(root: Path) -> str | None:
    """The token kept by the portal, for the environment of a publishing job; None without one."""
    return _load(root).get('token')
