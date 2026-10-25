"""Is the comma reachable from this server? The comma pill of the top bar asks (GET /api/panel/comma).

The comma is the address set in the comma panel of the site (``<data root>/web/comma.json``, set_address), else the
ssh host of helpers/fetch_comma_segments.sh: COMMA_HOST (default "comma", the Host of ~/.ssh/config), resolved once
with ``ssh -G`` to its address and port. The panel's download gets the same address (--host). The check only opens a
TCP connection to that port and closes it: no login, nothing runs on the comma. A thread repeats it every
CHECK_SECONDS while the site asks, so a page never waits for a comma that is away (the car off, or not on the Wi-Fi).
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
TIMEOUT_SECONDS = 2.0
# The thread stops when nobody asked for this long; the next question starts it again.
IDLE_SECONDS = 120

_lock = threading.Lock()
_state: dict = {'online': None, 'checked': None}
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
        _state.update(online=None, checked=None)
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


def _loop() -> None:
    global _thread
    while True:
        online = _check()
        with _lock:
            _state.update(online=online, checked=datetime.now(timezone.utc).isoformat(timespec='seconds'))
            if time.monotonic() - _asked > IDLE_SECONDS:
                _thread = None
                return
        time.sleep(CHECK_SECONDS)


def state(root: Path) -> dict:
    """{'online': True / False / None (not checked yet), 'checked': ISO time of the last check, 'address': what is
    tried, 'configured': whether it was set in the panel}."""
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
    return {**tried, 'address': address, 'configured': configured(root) is not None}
