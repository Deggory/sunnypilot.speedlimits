"""Is the comma reachable from this server? The comma pill of the top bar asks (GET /api/panel/comma).

The comma is the address set in the comma panel of the site (``<data root>/web/comma.json``, set_address), else the
ssh host of helpers/fetch_comma_segments.sh: COMMA_HOST (default "comma", the Host of ~/.ssh/config), resolved once
with ``ssh -G`` to its address and port. The panel's download gets the same address (--host). The check only opens a
TCP connection to that port and closes it. When the comma answers, a login is tried too, every LOGIN_SECONDS: ssh with
this server's key in batch mode, running ``true`` (nothing changes on the comma). The comma takes the public keys of
the GitHub user set in its Settings → Network → SSH Keys, so a refused key is fixed on GitHub (server_key). The time of
the last login that worked is kept (``<data root>/web/comma-seen.json``): a comma that is only switched off is not
offered the setup from zero again, the panel says when it was last connected. A thread
repeats the checks while the site asks, so a page never waits for a comma that is away (car off, not on the Wi-Fi).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time

CHECK_SECONDS = 15
LOGIN_SECONDS = 300
TIMEOUT_SECONDS = 2.0
# The thread stops when nobody asked for this long; the next question starts it again.
IDLE_SECONDS = 120

_lock = threading.Lock()
_state: dict = {'online': None, 'checked': None, 'login': None}
_login_at = 0.0
_asked = 0.0
_thread: threading.Thread | None = None
_address: tuple[str, int] | None = None
_root: Path | None = None
# An IP or a host name, with the ssh user before it if not "comma" (helpers/fetch_comma_segments.sh adds "comma@").
ADDRESS = re.compile(r'^(?:[a-z_][a-z0-9_-]{0,31}@)?[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$')


class AddressError(ValueError):
    """An address the comma panel refuses."""


def settings_path(root: Path) -> Path:
    return Path(root) / 'web' / 'comma.json'


def seen_path(root: Path) -> Path:
    """When this server last logged in to the comma: a comma that is only switched off is not one to set up again."""
    return Path(root) / 'web' / 'comma-seen.json'


def last_login(root: Path) -> str | None:
    """ISO time of the last login that worked, or None (never, or the file unreadable)."""
    try:
        seen = json.loads(seen_path(root).read_text(encoding='utf-8')).get('login')
    except (OSError, ValueError, AttributeError):
        return None
    return seen if isinstance(seen, str) else None


def _remember_login(root: Path | None, when: str) -> None:
    if root is None:
        return
    path = seen_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.' + path.name + '.tmp')
        temporary.write_text(json.dumps({'login': when}) + '\n', encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        pass


def configured(root: Path) -> str | None:
    """The address set in the comma panel, or None (then the ssh host "comma" of ~/.ssh/config)."""
    try:
        address = json.loads(settings_path(root).read_text(encoding='utf-8')).get('address')
    except (OSError, ValueError, AttributeError):
        return None
    return address if isinstance(address, str) and ADDRESS.match(address) else None


def set_address(root: Path, address: str) -> str | None:
    """Keeps the address of the comma ('' goes back to the ssh host "comma"); the next check uses it."""
    global _address
    address = (address or '').strip()
    if address and not ADDRESS.match(address):
        raise AddressError('An IP address or a host name, like 192.168.1.20 or comma.local.')
    path = settings_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps({'address': address or None}) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    with _lock:
        _address = None
        _state.update(online=None, checked=None, login=None)
    return address or None


def _resolve() -> tuple[str, int] | None:
    """The address and port to try: the one set in the panel (port 22), else COMMA_HOST as ssh would connect it
    (user@ is dropped)."""
    address = configured(_root) if _root is not None else None
    if address:
        return address.rsplit('@', 1)[-1], 22
    host = os.environ.get('COMMA_HOST', 'comma').rsplit('@', 1)[-1]
    try:
        out = subprocess.run(['ssh', '-G', host], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    values = dict(line.split(' ', 1) for line in out.splitlines() if ' ' in line)
    try:
        return values.get('hostname', host), int(values.get('port', '22'))
    except ValueError:
        return None


def _check() -> bool:
    global _address
    if _address is None:
        _address = _resolve()
    if _address is None:
        return False
    try:
        with socket.create_connection(_address, timeout=TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def _ssh_target() -> str:
    """Who ssh logs in as: the address set in the panel ("comma@" added, as helpers/fetch_comma_segments.sh does), else
    COMMA_HOST (the Host of ~/.ssh/config, with its user)."""
    address = configured(_root) if _root is not None else None
    if address:
        return address if '@' in address else 'comma@' + address
    return os.environ.get('COMMA_HOST', 'comma')


def _login() -> bool:
    """Can this server log in to the comma with its key? (true is the only command run.)"""
    try:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', '-o', 'StrictHostKeyChecking=accept-new',
                                 _ssh_target(), 'true'], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def server_key() -> str | None:
    """This server's public ssh key, to add to the GitHub user of the comma (public: it is no secret)."""
    for name in ('id_ed25519.pub', 'id_ecdsa.pub', 'id_rsa.pub'):
        try:
            return (Path.home() / '.ssh' / name).read_text(encoding='utf-8').strip()
        except OSError:
            continue
    return None


def _loop() -> None:
    global _thread, _login_at
    while True:
        online = _check()
        login = _state.get('login')
        if not online:
            login = None
        elif login is None or time.monotonic() - _login_at > LOGIN_SECONDS:
            login, _login_at = _login(), time.monotonic()
            if login:
                _remember_login(_root, datetime.now(timezone.utc).isoformat(timespec='seconds'))
        with _lock:
            _state.update(online=online, login=login, checked=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            if time.monotonic() - _asked > IDLE_SECONDS:
                _thread = None
                return
        time.sleep(CHECK_SECONDS)


def check_now(root: Path) -> dict:
    """The checks done now, not at the next round of the thread (the Test button of the panel): answers the ssh port,
    and if so, is this server's key accepted."""
    global _root, _login_at
    with _lock:
        _root = Path(root)
    online = _check()
    login = _login() if online else None
    checked = datetime.now(timezone.utc).isoformat(timespec='seconds')
    if login:
        _remember_login(Path(root), checked)
    with _lock:
        _login_at = time.monotonic()
        _state.update(online=online, login=login, checked=checked)
    return state(root)


def state(root: Path) -> dict:
    """{'online': True / False / None (not checked yet), 'login': whether this server's key is accepted (None while
    the comma is away), 'checked': ISO time of the last check, 'address': what is tried, 'configured': whether it was
    set in the panel, 'server_key': this server's public key, 'last_login': ISO time of the last login that worked}."""
    global _asked, _thread, _root
    with _lock:
        _root = Path(root)
        _asked = time.monotonic()
        if _thread is None:
            _thread = threading.Thread(target=_loop, name='comma-link', daemon=True)
            _thread.start()
        tried = dict(_state)
    address = configured(root)
    if address is None:
        resolved = _address
        address = os.environ.get('COMMA_HOST', 'comma') + (' (' + resolved[0] + ')' if resolved else '')
    return {**tried, 'address': address, 'configured': configured(root) is not None, 'server_key': server_key(),
            'last_login': last_login(root)}
