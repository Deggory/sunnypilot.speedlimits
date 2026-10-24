# [site-login] - START
"""The administrator of the review site: one account, a password and the signed-in browsers.

On the LAN the browser signs in with a name and a password. The first visit creates the account: the page
asks for the setup code (NNSLR_REVIEW_TOKEN, the token of /etc/speedlimits-web.env and of the repo .env),
then a name and a password. From then on the pages, the panel and their calls need a signed-in session;
the setup code no longer opens the site in a browser. It stays the token of the REST API /api/v1, of MCP
/mcp and of the agents that send it in a header (X-Review-Token), never in the address.

Password: scrypt (n=2**15, r=8, p=1, 16-byte random salt, 32-byte key), never stored or logged in clear. A
wrong name costs the same scrypt as a wrong password, so the answer time does not tell which one was wrong.
Sessions: 32 random bytes in an HttpOnly SameSite=Strict cookie; the server keeps only their SHA-256, so
the session file cannot be replayed. A session ends after 7 days without use or 30 days in all, on
sign-out, and for every browser when the password changes.

Attempts: 5 wrong answers from one address in 15 minutes lock that address for 15 minutes, and 50 from
anywhere lock sign-in for everyone for 15 minutes (sessions already open keep working). At most two
password checks run at once, so a flood of attempts cannot take all the CPU.

Files (<data root>/web/auth/, folder 700, files 600, written all at once): admin.json and sessions.json.
To start over (a forgotten password) stop the site and move admin.json away: the next visit asks for the
setup code again. Standard library only. Ported from openpilot.distill's web/auth.py (same author, MIT).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path

from nnslr_tools.review_db import ReviewRejected

USERNAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._@-]{0,63}$')
MIN_PASSWORD = 12
MAX_PASSWORD = 256
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
SESSION_IDLE_SECONDS = 7 * 24 * 3600
SESSION_MAX_SECONDS = 30 * 24 * 3600
# A session last-use time is written to disk at most this often.
SESSION_SAVE_SECONDS = 3600
MAX_SESSIONS = 20
FAIL_WINDOW_SECONDS = 15 * 60
FAILS_PER_ADDRESS = 5
FAILS_IN_ALL = 50
CONCURRENT_CHECKS = 2

WRONG_LOGIN = 'Wrong name or password.'
WRONG_CODE = 'Wrong setup code.'
ALREADY = 'The administrator already exists: sign in.'


def auth_dir(root: Path) -> Path:
    return Path(root) / 'web' / 'auth'


def read_private(path: Path) -> str:
    """The text of a file, '' when it does not exist. Any other read error is refused, so a file that
    cannot be read is never overwritten."""
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ''
    except (OSError, UnicodeDecodeError) as exc:
        raise ReviewRejected(500, 'unreadable', 'Cannot read the saved sign-in data on this server ('
                             + type(exc).__name__ + ').') from None


def write_private(path: Path, text: str) -> None:
    """Writes a file readable by its owner only (600), all at once: a temporary file in the same folder,
    flushed to disk, then renamed over the old one. A link is never written through."""
    if path.is_symlink():
        raise ReviewRejected(409, 'file_is_a_link', 'The sign-in file on this server is a link: '
                             'fix it by hand.')
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=str(path.parent))
    except OSError as exc:
        raise ReviewRejected(500, 'cannot_save', 'Cannot save it on this server ('
                             + type(exc).__name__ + ').') from None
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise ReviewRejected(500, 'cannot_save', 'Cannot save it on this server ('
                             + type(exc).__name__ + ').') from None


def hash_password(password: str, salt: bytes | None = None, n: int | None = None, r: int = SCRYPT_R,
                  p: int = SCRYPT_P) -> dict:
    if salt is None:
        salt = secrets.token_bytes(16)
    if n is None:
        n = SCRYPT_N
    key = hashlib.scrypt(password.encode('utf-8'), salt=salt, n=n, r=r, p=p, maxmem=SCRYPT_MAXMEM, dklen=32)
    return {'n': n, 'r': r, 'p': p, 'salt': base64.b64encode(salt).decode('ascii'),
            'hash': base64.b64encode(key).decode('ascii')}


def password_matches(password: str, stored: dict) -> bool:
    try:
        salt = base64.b64decode(stored['salt'], validate=True)
        expected = base64.b64decode(stored['hash'], validate=True)
        n = int(stored['n'])
        r = int(stored['r'])
        p = int(stored['p'])
    except (KeyError, TypeError, ValueError):
        return False
    if n > 2 ** 20 or r > 32 or p > 16:
        return False
    found = hash_password(password, salt, n, r, p)
    return hmac.compare_digest(base64.b64decode(found['hash']), expected)


def _session_key(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def check_new_password(password, username: str) -> str:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise ReviewRejected(400, 'weak_password', 'The password must have at least ' + str(MIN_PASSWORD)
                             + ' characters.')
    if len(password) > MAX_PASSWORD:
        raise ReviewRejected(400, 'weak_password', 'The password can have at most ' + str(MAX_PASSWORD)
                             + ' characters.')
    if password.strip() == '' or len(set(password)) < 4:
        raise ReviewRejected(400, 'weak_password', 'Choose a password with more different characters.')
    if password.lower() == username.lower():
        raise ReviewRejected(400, 'weak_password', 'The password cannot be the same as the name.')
    return password


class Throttle:
    """Wrong answers per address and in all over the last FAIL_WINDOW_SECONDS."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._by_address: dict[str, list[float]] = {}
        self._all: list[float] = []

    def _trim(self, now: float) -> None:
        start = now - FAIL_WINDOW_SECONDS
        self._all = [moment for moment in self._all if moment > start]
        for address in list(self._by_address):
            kept = [moment for moment in self._by_address[address] if moment > start]
            if kept:
                self._by_address[address] = kept
            else:
                del self._by_address[address]

    def check(self, address: str) -> None:
        with self._lock:
            now = self._clock()
            self._trim(now)
            mine = self._by_address.get(address, [])
            wait = 0.0
            if len(mine) >= FAILS_PER_ADDRESS:
                wait = mine[-FAILS_PER_ADDRESS] + FAIL_WINDOW_SECONDS - now
            if len(self._all) >= FAILS_IN_ALL:
                wait = max(wait, self._all[-FAILS_IN_ALL] + FAIL_WINDOW_SECONDS - now)
            if wait > 0:
                minutes = max(1, int(wait // 60) + 1)
                raise ReviewRejected(429, 'too_many_attempts', 'Too many wrong attempts: try again in '
                                     + str(minutes) + ' minutes.')

    def failed(self, address: str) -> None:
        with self._lock:
            now = self._clock()
            self._all.append(now)
            self._by_address.setdefault(address, []).append(now)

    def succeeded(self, address: str) -> None:
        with self._lock:
            self._by_address.pop(address, None)


class AdminAuth:
    def __init__(self, directory: Path, setup_code: str | None = None, clock=time.time):
        self.directory = directory
        self.admin_path = directory / 'admin.json'
        self.sessions_path = directory / 'sessions.json'
        self.setup_code = setup_code
        self._clock = clock
        self._lock = threading.Lock()
        self._checks = threading.BoundedSemaphore(CONCURRENT_CHECKS)
        self.throttle = Throttle()
        self._admin = self._load_admin()
        self._sessions = self._load_sessions()
        # A wrong name still pays for one scrypt.
        self._dummy = hash_password(secrets.token_urlsafe(16))

    # -- files -------------------------------------------------------------------------------------
    def _load_admin(self) -> dict | None:
        text = read_private(self.admin_path)
        if not text:
            return None
        try:
            data = json.loads(text)
        except ValueError:
            raise ReviewRejected(500, 'unreadable', 'The administrator file of this server is damaged.') \
                from None
        if not isinstance(data, dict) or not isinstance(data.get('user'), str) or not isinstance(
                data.get('password'), dict):
            raise ReviewRejected(500, 'unreadable', 'The administrator file of this server is damaged.')
        return data

    def _load_sessions(self) -> dict:
        text = read_private(self.sessions_path)
        if not text:
            return {}
        try:
            data = json.loads(text)
        except ValueError:
            return {}
        result = {}
        if isinstance(data, dict):
            for key, value in data.items():
                if (re.fullmatch(r'[0-9a-f]{64}', key) and isinstance(value, dict)
                        and isinstance(value.get('created'), (int, float))
                        and isinstance(value.get('seen'), (int, float))):
                    result[key] = {'created': float(value['created']), 'seen': float(value['seen'])}
        return result

    def _save_admin(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private(self.admin_path, json.dumps(self._admin, indent=1) + '\n')

    def _save_sessions(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private(self.sessions_path, json.dumps(self._sessions, indent=1) + '\n')

    # -- state -------------------------------------------------------------------------------------
    def configured(self) -> bool:
        with self._lock:
            return self._admin is not None

    def username(self) -> str | None:
        with self._lock:
            return None if self._admin is None else self._admin['user']

    @staticmethod
    def _expired(entry: dict, now: float) -> bool:
        return now - entry['seen'] > SESSION_IDLE_SECONDS or now - entry['created'] > SESSION_MAX_SECONDS

    def _new_session(self, now: float) -> str:
        # Under self._lock.
        for key in [key for key, entry in self._sessions.items() if self._expired(entry, now)]:
            del self._sessions[key]
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda key: self._sessions[key]['seen'])
            del self._sessions[oldest]
        token = secrets.token_urlsafe(32)
        self._sessions[_session_key(token)] = {'created': now, 'seen': now}
        self._save_sessions()
        return token

    def session_valid(self, token) -> bool:
        if not isinstance(token, str) or not 20 <= len(token) <= 100:
            return False
        key = _session_key(token)
        with self._lock:
            if self._admin is None:
                return False
            entry = self._sessions.get(key)
            if entry is None:
                return False
            now = self._clock()
            if self._expired(entry, now):
                del self._sessions[key]
                self._save_sessions()
                return False
            previous = entry['seen']
            entry['seen'] = now
            if now - previous > SESSION_SAVE_SECONDS:
                self._save_sessions()
            return True

    # -- actions -----------------------------------------------------------------------------------
    def _checking(self):
        if not self._checks.acquire(timeout=5):
            raise ReviewRejected(503, 'busy', 'The server is checking other attempts: try again in a moment.')
        return self._checks

    def check_setup_code(self, code, address: str) -> None:
        """Step 1 of the first visit: is this the setup code? (the account is not created yet)."""
        if self.configured():
            raise ReviewRejected(409, 'already_configured', ALREADY)
        self.throttle.check(address)
        if not self._code_matches(code):
            self.throttle.failed(address)
            raise ReviewRejected(403, 'wrong_setup_code', WRONG_CODE)

    def _code_matches(self, code) -> bool:
        if self.setup_code is None or not isinstance(code, str) or len(code) > 256:
            return False
        return hmac.compare_digest(code.strip().encode('utf-8'), self.setup_code.encode('utf-8'))

    def setup(self, code, username, password, address: str) -> str:
        self.check_setup_code(code, address)
        if not isinstance(username, str) or not USERNAME_RE.fullmatch(username):
            raise ReviewRejected(400, 'invalid_name', 'The name can have letters, digits and . _ @ - (at most 64).')
        check_new_password(password, username)
        stored = hash_password(password)
        with self._lock:
            if self._admin is not None:
                raise ReviewRejected(409, 'already_configured', ALREADY)
            self._admin = {'version': 1, 'user': username, 'password': stored, 'changed': self._clock()}
            self._save_admin()
            self._sessions = {}
            token = self._new_session(self._clock())
        self.throttle.succeeded(address)
        print('Sito: creato l’amministratore "' + username + '"', flush=True)
        return token

    def login(self, username, password, address: str) -> str:
        if not self.configured():
            raise ReviewRejected(409, 'not_configured', 'Create the administrator first.')
        self.throttle.check(address)
        if not isinstance(username, str) or not isinstance(password, str) or len(password) > MAX_PASSWORD \
                or len(username) > 64:
            self.throttle.failed(address)
            raise ReviewRejected(401, 'wrong_login', WRONG_LOGIN)
        with self._lock:
            admin = dict(self._admin)
        slot = self._checking()
        try:
            name_ok = hmac.compare_digest(username.strip().lower().encode('utf-8'),
                                          admin['user'].lower().encode('utf-8'))
            stored = admin['password'] if name_ok else self._dummy
            password_ok = password_matches(password, stored)
        finally:
            slot.release()
        if not (name_ok and password_ok):
            self.throttle.failed(address)
            print('Sito: tentativo di accesso sbagliato', flush=True)
            raise ReviewRejected(401, 'wrong_login', WRONG_LOGIN)
        self.throttle.succeeded(address)
        with self._lock:
            return self._new_session(self._clock())

    def logout(self, token, everywhere: bool = False) -> None:
        with self._lock:
            if everywhere:
                self._sessions = {}
            elif isinstance(token, str):
                self._sessions.pop(_session_key(token), None)
            self._save_sessions()

    def change_password(self, token: str, current, new, address: str) -> str:
        """Every other browser is signed out; the answer is the new session of this one."""
        self.throttle.check(address)
        with self._lock:
            admin = dict(self._admin)
        if not isinstance(current, str) or len(current) > MAX_PASSWORD:
            current = ''
        slot = self._checking()
        try:
            ok = password_matches(current, admin['password'])
        finally:
            slot.release()
        if not ok:
            self.throttle.failed(address)
            raise ReviewRejected(403, 'wrong_password', 'The current password is wrong.')
        check_new_password(new, admin['user'])
        stored = hash_password(new)
        with self._lock:
            self._admin = dict(self._admin, password=stored, changed=self._clock())
            self._save_admin()
            self._sessions = {}
            fresh = self._new_session(self._clock())
        print('Sito: password dell’amministratore cambiata, gli altri browser sono usciti', flush=True)
        return fresh

    def sessions_count(self) -> int:
        with self._lock:
            now = self._clock()
            return sum(1 for entry in self._sessions.values() if not self._expired(entry, now))
# [site-login] - END
