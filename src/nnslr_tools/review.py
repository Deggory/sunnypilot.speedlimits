# [preannotation] - START
"""Review page and review server.

The server reads the pre-annotation reports (read-only) and writes the reviewed
frames into the review database (:mod:`nnslr_tools.review_db`). It listens on
127.0.0.1 by default; on any other address a browser signs in as the administrator
(:mod:`nnslr_tools.review_auth`, name and password; the first visit creates it with
the access token as setup code), and agents send the access token in a header.

Besides the page it serves the REST API ``/api/v1`` and, when the optional MCP
SDK is installed, the MCP Streamable HTTP endpoint ``/mcp``
(:mod:`nnslr_tools.review_mcp`). Both use the operations of
:mod:`nnslr_tools.review_service`, as the page does, and always need the token
in a header (``Authorization: Bearer`` or ``X-Review-Token``), also on 127.0.0.1.
"""
from __future__ import annotations

import html
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from nnslr_tools.manifest import parse_route_id
from nnslr_tools.preannotate import PreannotationError, _inside, resolve_data_root
# [site-login] - START
from nnslr_tools.review_auth import SESSION_MAX_SECONDS, AdminAuth, auth_dir
# [site-login] - END
from nnslr_tools.review_hints import empty_stats
from nnslr_tools.review_service import ReviewService

TOKEN_ENV = 'NNSLR_REVIEW_TOKEN'
MAX_BODY_BYTES = 512 * 1024
# A connection that sends nothing for this many seconds is closed, and at most
# MAX_CONNECTIONS are open at once, so idle clients cannot use up threads.
REQUEST_TIMEOUT_SECONDS = 30
MAX_CONNECTIONS = 64
# [site-login] - START
LOGIN_MESSAGE = 'Accesso richiesto: ricarica la pagina e accedi con nome e password.'
# Sign-in calls that work without a session, and all of them.
PUBLIC_AUTH_ROUTES = ('/api/auth/login', '/api/auth/setup/check', '/api/auth/setup')
AUTH_ROUTES = PUBLIC_AUTH_ROUTES + ('/api/auth/logout', '/api/auth/logout-all', '/api/auth/password')
# The pages a browser opens: without a session they show the sign-in page instead.
PAGE_RE = re.compile(r'/(index\.html|pannello|account|route/[^/]+(/(index\.html)?)?)?')
# [site-login] - END
API_TOKEN_MESSAGE = ('Token mancante o errato: invialo nell’header "Authorization: Bearer <token>" '
                     '(oppure "X-Review-Token"), mai nell’indirizzo.')
DEFAULT_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def render_review(rows: list[dict], run_id: str, *, route_id: str = '', server: bool = False,
                  nonce: str = '', hints: dict | None = None, hint_stats: dict | None = None,
                  suspicious: dict | None = None, routes: list[dict] | None = None) -> str:
    """The review page with its data embedded. Hints and suspicious flags travel apart from the
    frame rows, so the JSONL export (which copies the rows) never contains them."""
    template=files('nnslr_tools').joinpath('review.html').read_text()
    if hints is None:
        hints = {}
    if hint_stats is None:
        hint_stats = empty_stats()
    if suspicious is None:
        suspicious = {}
    if routes is None:
        routes = []
    data={'run_id':run_id,'route_id':route_id,'server':server,'frames':rows,'hints':hints,
          'hint_stats':hint_stats,'suspicious':suspicious,'routes':routes}
    payload=json.dumps(data,ensure_ascii=True,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=template.replace('__NNSLR_NONCE__',nonce)
    return page.replace('__NNSLR_DATA__',payload)


def write_review(directory: Path, rows: list[dict], run_id: str) -> None:
    route_id=''
    if rows and isinstance(rows[0],dict) and isinstance(rows[0].get('route_id'),str):
        route_id=rows[0]['route_id']
    (directory/'index.html').write_text(render_review(rows,run_id,route_id=route_id),encoding='utf-8')


def public_file(directory: Path, url: str) -> Path | None:
    name=unquote(urlsplit(url).path)
    if name in {'/','/index.html'}:
        relative=Path('index.html')
    elif re.fullmatch(r'/images/[0-9]{8}\.jpg',name):
        relative=Path(name.lstrip('/'))
    else:
        return None
    path=directory/relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
        return None
    return path


def latest_report(root: Path, route: str) -> Path:
    parse_route_id(route)
    base=_inside(root,root/'derived/preannotations'/route)
    pointer=_inside(base,base/'latest.json')
    if not pointer.is_file():
        raise PreannotationError(f'no report; run: nnslr preannotate --route {route}')
    run_id=json.loads(pointer.read_text()).get('run_id','')
    if not isinstance(run_id,str) or not re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}',run_id):
        raise PreannotationError('invalid latest report pointer')
    directory=_inside(base,base/run_id)
    if public_file(directory,'/') is None:
        raise PreannotationError('report index is missing')
    return directory


# [reviewed-dataset] - START
def report_rows(directory: Path) -> list[dict]:
    from nnslr_tools.annotations import load_jsonl
    source = _inside(directory, directory / 'preannotations.jsonl')
    return load_jsonl(source)


def refresh_review(directory: Path) -> None:
    """Upgrade the presentation of an existing run, preserving proposals and run identity."""
    metadata = _inside(directory, directory / 'run.json')
    run_id = json.loads(metadata.read_text())['run_id']
    write_review(directory, report_rows(directory), run_id)
# [reviewed-dataset] - END


def list_reports(root: Path) -> list[tuple[str, Path]]:
    """(route, current report directory) for every route with a valid pre-annotation report."""
    base = root / 'derived' / 'preannotations'
    found = []
    if not base.is_dir():
        return found
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            directory = latest_report(root, entry.name)
        except (PreannotationError, ValueError, OSError):
            continue
        found.append((entry.name, directory))
    return found


def terminal_text(value) -> str:
    """Text from a request made safe for the server terminal: no control characters, bounded length."""
    text = str(value)
    if len(text) > 100:
        text = text[:100] + '…'
    result = []
    for character in text:
        if character.isprintable():
            result.append(character)
        else:
            result.append('?')
    return ''.join(result)


def is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return False


def review_token() -> str:
    """The access token: NNSLR_REVIEW_TOKEN when set, otherwise a new random one."""
    configured = os.environ.get(TOKEN_ENV, '')
    if configured:
        if not re.fullmatch(r'[A-Za-z0-9_-]{16,}', configured):
            raise PreannotationError(f'{TOKEN_ENV} must have at least 16 characters among letters, digits, "-" and "_"')
        return configured
    return secrets.token_urlsafe(24)


def lan_addresses() -> list[str]:
    """IPv4 addresses of this machine that other computers can use (best effort, no traffic sent)."""
    found = []
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        address = info[4][0]
        if not is_loopback(address) and address not in found:
            found.append(address)
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # A UDP connect only selects the outgoing interface; nothing is sent.
        probe.connect(('10.255.255.255', 1))
        address = probe.getsockname()[0]
        if not is_loopback(address) and address not in found:
            found.append(address)
    except OSError:
        pass
    finally:
        probe.close()
    return found


def _print_flush(text: str) -> None:
    print(text, flush=True)


class ReviewApp(ReviewService):
    """State shared by the request threads: the review operations plus access token and panel.

    ``api_token`` protects /api/v1 and /mcp. It is the page token when there is
    one; on 127.0.0.1 without a page token it is NNSLR_REVIEW_TOKEN or a random one.
    """

    def __init__(self, root: Path, store, *, token: str | None, require_token: bool, api_token: str | None = None):
        from nnslr_tools.panel import Panel
        super().__init__(root, store, log=_print_flush)
        self.token = token
        self.require_token = require_token
        self.api_token = api_token or token
        self.cookie_name = 'nnslr_review'
        # [site-login] - START
        # On the LAN a browser signs in; the access token is the setup code of the first visit.
        self.auth = AdminAuth(auth_dir(root), token) if require_token else None
        # [site-login] - END
        self._mcp = None
        self._mcp_failed = False
        self._mcp_lock = threading.Lock()
        # [control-panel] - START
        self.panel = Panel(root)
        # [control-panel] - END

    def token_matches(self, supplied) -> bool:
        if self.token is None or not isinstance(supplied, str):
            return False
        return secrets.compare_digest(supplied.encode('utf-8'), self.token.encode('utf-8'))

    def api_token_matches(self, supplied) -> bool:
        if self.api_token is None or not isinstance(supplied, str):
            return False
        return secrets.compare_digest(supplied.encode('utf-8'), self.api_token.encode('utf-8'))

    def mcp_bridge(self):
        """The MCP Streamable HTTP bridge, started at the first /mcp request; None without the MCP SDK."""
        with self._mcp_lock:
            if self._mcp is None:
                try:
                    from nnslr_tools.review_mcp import HttpBridge, sdk_available
                except ImportError:
                    return None
                if self._mcp_failed or not sdk_available():
                    return None
                try:
                    self._mcp = HttpBridge(self, read_only=os.environ.get('NNSLR_MCP_READ_ONLY', '') == '1')
                except Exception as exc:
                    # An incompatible SDK must not take the site and /api/v1 down: /mcp answers 501.
                    self._mcp_failed = True
                    self.log(f'MCP non disponibile: {type(exc).__name__}: {terminal_text(exc)}')
                    return None
            return self._mcp

    def close(self) -> None:
        with self._mcp_lock:
            if self._mcp is not None:
                self._mcp.close()
                self._mcp = None

    def host_allowed(self, host_header) -> bool:
        # With a token the cookie protects every request. Without one (loopback
        # only) the Host header must be a loopback name, against DNS rebinding.
        if self.require_token:
            return True
        if not isinstance(host_header, str) or not host_header:
            return False
        name = host_header.strip().lower()
        if name.startswith('['):
            end = name.find(']')
            if end < 0:
                return False
            name = name[1:end]
        elif ':' in name:
            name = name.rsplit(':', 1)[0]
        return is_loopback(name)

    def route_links(self) -> list[dict]:
        """Every route with a report, for the route menu of the review page."""
        links = []
        for route, directory in list_reports(self.root):
            links.append({'route_id': route, 'url': '/route/' + route + '/'})
        return links

    def route_summaries(self) -> list[dict]:
        summaries = []
        for route, directory in list_reports(self.root):
            try:
                directory, rows, by_key = self.report(route)
            except (PreannotationError, ValueError, OSError):
                continue
            with_proposals = 0
            for row in rows:
                detections = row.get('detections')
                if isinstance(detections, list) and detections:
                    with_proposals = with_proposals + 1
            hints_by_frame, hint_stats = self.hints(route, rows)
            suspicious = self.suspicious(route, directory.name, rows, hints_by_frame)
            summaries.append({
                'route_id': route,
                'run_id': directory.name,
                'frame_count': len(rows),
                'frames_with_proposals': with_proposals,
                'confirmed_count': self.store.confirmed_count(route, directory.name),
                'suspicious_count': len(suspicious),
                'hint_frames': hint_stats['frames'],
                'url': '/route/' + route + '/',
            })
        return summaries


def _site_style() -> str:
    """Style of the route index, sign-in, account and error pages: site.css, read on every request so a
    change to it is live at the next page load. Same tokens and top bar as the review page and the panel."""
    return files('nnslr_tools').joinpath('site.css').read_text()


def _plural(count: int, one: str, many: str) -> str:
    if count == 1:
        return '1 ' + one
    return str(count) + ' ' + many


# Dark (the default, as distill-web and perception-web) or the light look chosen on the Account page,
# read before the first paint. The pages served from html files carry the same line.
THEME_SCRIPT = ('<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\');'
                'if(t===\'light\'||t===\'auto\')document.documentElement.dataset.theme=t}}catch(e){{}}</script>')
# The tabs of the site, the same on every page: (current key, address, label).
SITE_TABS = (('route', '/', 'Route'), ('panel', '/pannello', 'Pannello'), ('status', '/stato', 'Stato'))
# At the right of the top bar on every page: the dark / light switch and the Account icon (not a tab).
THEME_TOGGLE = '<button class="icon-btn theme-toggle" type="button" id="theme-toggle" aria-pressed="false" aria-label="Passa al tema chiaro" title="Passa al tema chiaro"><svg class="i-sun" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="3.6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M10 1.8v2.2M10 16v2.2M1.8 10H4M16 10h2.2M4.2 4.2l1.6 1.6M14.2 14.2l1.6 1.6M4.2 15.8l1.6-1.6M14.2 5.8l1.6-1.6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg><svg class="i-moon" viewBox="0 0 20 20" aria-hidden="true"><path d="M16.5 12.6A7 7 0 0 1 7.4 3.5a7 7 0 1 0 9.1 9.1z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg></button>'
ACCOUNT_LINK = '<a class="icon-btn account-btn" href="/account" aria-label="Account" title="Account"><svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="7" r="3.4" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M3.6 17.2c.9-3.1 3.4-4.8 6.4-4.8s5.5 1.7 6.4 4.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg></a>'
THEME_TOGGLE_SCRIPT = '<script nonce="{nonce}">(function(){{var b=document.getElementById(\'theme-toggle\');if(!b)return;var r=document.documentElement;function dark(){{var t=r.dataset.theme;if(t===\'light\')return false;if(t===\'auto\')return!matchMedia(\'(prefers-color-scheme: light)\').matches;return true}}function show(){{var d=dark();b.classList.toggle(\'is-light\',!d);b.setAttribute(\'aria-pressed\',String(!d));b.title=d?\'Passa al tema chiaro\':\'Passa al tema scuro\';b.setAttribute(\'aria-label\',b.title)}}b.addEventListener(\'click\',function(){{var next=dark()?\'light\':\'dark\';try{{localStorage.setItem(\'nnslr-theme\',next)}}catch(e){{}}if(next===\'dark\')delete r.dataset.theme;else r.dataset.theme=next;show()}});show()}})();</script>'


def _page_tail(nonce: str = '') -> str:
    """End of the server pages: the script of the dark / light switch."""
    return (THEME_TOGGLE_SCRIPT.format(nonce=nonce) if nonce else '') + '</body></html>'


def _page_head(title: str, current: str = '', nonce: str = '') -> list[str]:
    """Head, top bar and tabs of the server pages. current is the key of the tab to mark, '' for none."""
    tabs = ''
    for key, href, label in SITE_TABS:
        tabs += '<a href="' + href + '"' + (' aria-current="page"' if key == current else '') + '>' + label + '</a>'
    theme = THEME_SCRIPT.format(nonce=nonce) if nonce else ''
    return [
        '<!doctype html><html lang="it"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="color-scheme" content="light dark">',
        theme,
        '<title>' + html.escape(title) + '</title>',
        '<style>' + _site_style() + '</style></head><body>',
        '<header class="topbar"><div class="topbar-inner">'
        '<a class="brand" href="/" title="Elenco delle route"><span class="brand-mark" aria-hidden="true">50</span>'
        '<span class="brand-name">Speed limits</span></a>'
        + THEME_TOGGLE + ACCOUNT_LINK.replace('<a ', '<a aria-current="page" ' if current == 'account' else '<a ', 1)
        + '</div><nav class="tabs" aria-label="Sezioni">' + tabs + '</nav></header>',
    ]


# [control-panel] - START
def render_panel(nonce: str) -> str:
    """The control panel page; it loads its data from /api/panel/status."""
    return files('nnslr_tools').joinpath('panel.html').read_text().replace('__NNSLR_NONCE__', nonce)


def render_status(nonce: str) -> str:
    """The Stato page (the machine and the jobs, as distill-web's Status); data from /api/panel/status."""
    return files('nnslr_tools').joinpath('status.html').read_text().replace('__NNSLR_NONCE__', nonce)
# [control-panel] - END


# [site-login] - START
def render_login(nonce: str) -> str:
    """The sign-in page (also the first visit, which creates the administrator)."""
    page = files('nnslr_tools').joinpath('login.html').read_text()
    return page.replace('__NNSLR_STYLE__', _site_style()).replace('__NNSLR_NONCE__', nonce)


def render_account(nonce: str, signed_in_browser: bool) -> str:
    """The account page: who is signed in, sign out, change the password."""
    lines = _page_head('Speed limits · Account', 'account', nonce)
    if not signed_in_browser:
        lines.append('<main class="page"><div class="card state-box center"><h1>Account</h1><p>Questo sito è avviato '
                     'solo per questo computer (127.0.0.1): non chiede di accedere. Sulla rete di casa (servizio '
                     '<code>speedlimits-web</code>) si entra con nome e password.</p></div></main>' + _page_tail(nonce))
        return '\n'.join(lines)
    lines.append(files('nnslr_tools').joinpath('account.html').read_text().replace('__NNSLR_NONCE__', nonce))
    return '\n'.join(lines)
# [site-login] - END


# Groups of the route list, in this order: the work in progress first, finished routes last.
ROUTE_GROUPS = (('doing', 'In corso'), ('todo', 'Da iniziare'), ('done', 'Completate'))


def _route_state(confirmed: int, total: int) -> str:
    if total and confirmed >= total:
        return 'done'
    if confirmed == 0:
        return 'todo'
    return 'doing'


def _route_row(summary: dict) -> str:
    total = summary['frame_count']
    confirmed = summary['confirmed_count']
    percent = 0
    if total:
        percent = round(confirmed * 100 / total)
    state = _route_state(confirmed, total)
    # What opening the route means: carry on (the one obvious action), start, or look again at a finished one.
    action_class = 'btn small open'
    action = 'Inizia →'
    if state == 'doing':
        action_class = 'btn small primary open'
        action = 'Riprendi →'
    elif state == 'done':
        action_class = 'btn small open again'
        action = 'Ricontrolla'
    suspicious = summary.get('suspicious_count', 0)
    hint_frames = summary.get('hint_frames', 0)
    stats = ['<span>' + _plural(total, 'fotogramma', 'fotogrammi') + '</span>',
             '<span>' + str(summary['frames_with_proposals']) + ' con proposte</span>']
    if suspicious:
        stats.append('<span class="sus">' + _plural(suspicious, 'sospetto', 'sospetti') + '</span>')
    else:
        stats.append('<span>0 sospetti</span>')
    if hint_frames:
        stats.append('<span>' + str(hint_frames) + ' con seconda lettura</span>')
    route = html.escape(summary['route_id'])
    return ('<li><a class="card-row ' + state + '" href="' + html.escape(summary['url']) + '">'
            + '<span class="state-icon" aria-hidden="true">' + ('✓' if state == 'done' else '') + '</span>'
            + '<span class="route" title="' + route + '">' + route + '</span>'
            + '<span class="count">' + str(confirmed) + ' di ' + str(total)
            + ' <span class="unit">confermati</span></span>'
            + '<span class="bar" aria-hidden="true"><span style="width:' + str(percent) + '%"></span></span>'
            + '<span class="stats">' + ''.join(stats) + '</span>'
            + '<span class="' + action_class + '">' + action + '</span></a></li>')


def render_index(summaries: list[dict], database: Path, nonce: str = '') -> str:
    lines = _page_head('Speed limits · Route', 'route', nonce)
    lines.append('<main class="page">')
    lines.append('<header class="page-head"><h1 class="page-title">Route</h1>'
                 '<p class="page-sub">Scegli una route da revisionare. Ogni conferma viene salvata subito '
                 'sul server.</p></header>')
    if not summaries:
        lines.append('<div class="card state-box"><h2>Nessuna route da revisionare</h2><p>Prepara una route con '
                     '<code>nnslr extract-frames</code> e <code>nnslr preannotate</code>, poi ricarica questa '
                     'pagina.</p></div>')
    else:
        grouped = {key: [] for key, _ in ROUTE_GROUPS}
        for summary in summaries:
            grouped[_route_state(summary['confirmed_count'], summary['frame_count'])].append(summary)
        for key, title in ROUTE_GROUPS:
            if not grouped[key]:
                continue
            lines.append('<section class="group" aria-labelledby="group-' + key + '">'
                         '<h2 class="group-title" id="group-' + key + '">' + title
                         + ' <span class="n">' + str(len(grouped[key])) + '</span></h2><ul class="cards">')
            for summary in grouped[key]:
                lines.append(_route_row(summary))
            lines.append('</ul></section>')
    lines.append('<details class="foot"><summary>Dettagli tecnici</summary>'
                 '<p>Database delle revisioni: <code>' + html.escape(str(database)) + '</code></p></details>')
    lines.append('</main>' + _page_tail(nonce))
    return '\n'.join(lines)


def render_problem(status: int, message: str, nonce: str = '') -> str:
    if status == 404:
        title = 'Pagina non trovata'
    elif status == 403:
        title = 'Accesso negato'
    else:
        title = 'Errore'
    lines = _page_head('Speed limits · ' + title, '', nonce)
    lines.append('<main class="page"><div class="card state-box center">')
    lines.append('<p class="code">Codice HTTP ' + str(status) + '</p>')
    lines.append('<h1>' + html.escape(title) + '</h1>')
    # "Pagina non trovata." under the title "Pagina non trovata" would only repeat it.
    if message.rstrip('.') != title:
        lines.append('<p>' + html.escape(message) + '</p>')
    # Every 403 page (token missing or wrong, host not allowed) would come back here from the list.
    if status != 403:
        lines.append('<a class="btn back" href="/">← Elenco delle route</a>')
    lines.append('</div></main>' + _page_tail(nonce))
    return '\n'.join(lines)


def make_handler(app: ReviewApp):
    from nnslr_tools.annotations import strict_json_loads
    from nnslr_tools.panel import PanelError
    from nnslr_tools.review_db import ReviewRejected, db_path, public_review, save_frame_review
    from nnslr_tools.review_service import review_schema

    class Handler(BaseHTTPRequestHandler):
        server_version = 'speedlimits-web'
        sys_version = ''
        # Socket timeout: a client that stops sending is disconnected, even before the token check.
        timeout = REQUEST_TIMEOUT_SECONDS

        def do_GET(self):
            self._dispatch('GET')

        def do_POST(self):
            self._dispatch('POST')

        def do_DELETE(self):
            self._dispatch('DELETE')

        def do_OPTIONS(self):
            self._dispatch('OPTIONS')

        def log_message(self, *args):
            pass

        def _dispatch(self, method):
            try:
                self._handle(method)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except ReviewRejected as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
            except PanelError as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
            except Exception as exc:
                self._problem(500, 'server_error', 'Errore interno del server: ' + type(exc).__name__ + '.',
                              self.path.startswith('/api/') or self.path.startswith('/mcp'))

        def _handle(self, method):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            api = path.startswith('/api/')
            if not app.host_allowed(self.headers.get('Host')):
                self._problem(403, 'host_not_allowed', 'Nome host non consentito per questo server.', api)
                return
            query = parse_qs(parts.query, keep_blank_values=True)
            if path == '/mcp' or path == '/api/v1' or path.startswith('/api/v1/'):
                self._external(method, path, query)
                return
            if method not in ('GET', 'POST'):
                self._problem(405, 'method_not_allowed', 'Metodo non consentito.', api)
                return
            # [site-login] - START
            if 'token' in query:
                # An old link with ?token=: the token no longer opens the site in a browser (it is the
                # setup code of the first visit); drop it from the address, the sign-in page follows.
                self._redirect(self._clean_target(path), None)
                return
            if not self._authorized():
                self._public(method, path, api)
                return
            # [site-login] - END
            if method == 'POST':
                self._post(path)
            else:
                self._get(path)

        # [site-login] - START
        @staticmethod
        def _clean_target(path):
            # Our own paths only, never "//host" or a backslash.
            if not re.fullmatch(r'/[A-Za-z0-9._/-]*', path) or path.startswith('//'):
                return '/'
            return path

        def _authorized(self):
            """A browser signed in as the administrator, or an agent with the access token in a header
            (X-Review-Token, as for /api/v1: a page cannot make another site send it)."""
            if not app.require_token:
                return True
            if app.token_matches(self.headers.get('X-Review-Token')):
                return True
            return self._session() is not None

        def _session(self):
            """The session token of this request when it is signed in, otherwise None."""
            if app.auth is None:
                return None
            # Plain "name=value; name=value" split: the browser also sends the
            # cookies of other web apps on this host (any port), and a value our
            # parser does not like must not hide the review cookie.
            for raw in self.headers.get_all('Cookie', []):
                for part in raw.split(';'):
                    name, separator, value = part.strip().partition('=')
                    if separator and name.strip() == app.cookie_name and app.auth.session_valid(value.strip()):
                        return value.strip()
            return None

        def _session_cookie(self, token):
            parts = [app.cookie_name + '=' + token, 'Path=/', 'HttpOnly', 'SameSite=Strict',
                     'Max-Age=' + str(SESSION_MAX_SECONDS)]
            origin = (self.headers.get('Origin') or '').strip().lower()
            proto = (self.headers.get('X-Forwarded-Proto') or '').strip().lower()
            if origin.startswith('https://') or proto == 'https':
                parts.append('Secure')
            return '; '.join(parts)

        def _expired_cookie(self):
            return app.cookie_name + '=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0'

        def _auth_state(self):
            if app.auth is None:
                return {'configured': False, 'signed_in': True, 'local': True}
            signed_in = self._session() is not None
            state = {'configured': app.auth.configured(), 'signed_in': signed_in, 'local': False}
            if signed_in:
                state['user'] = app.auth.username()
                state['sessions'] = app.auth.sessions_count()
            return state

        def _public(self, method, path, api):
            """A request without a session: the sign-in page (in place of any page), and the sign-in calls."""
            if method == 'GET' and not api and PAGE_RE.fullmatch(path):
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200 if path in ('/', '/index.html') else 401, 'text/html; charset=utf-8',
                           render_login(nonce).encode('utf-8'), csp)
                return
            if method == 'GET' and path == '/api/auth/state':
                self._json(200, self._auth_state())
                return
            if method == 'POST' and path in PUBLIC_AUTH_ROUTES:
                self._post(path)
                return
            self._problem(401, 'login_required', LOGIN_MESSAGE, api)

        def _auth_post(self, path, body):
            if app.auth is None:
                self._problem(404, 'not_found', 'Su 127.0.0.1 il sito non chiede di accedere.', True)
                return
            if not isinstance(body, dict):
                raise ReviewRejected(400, 'invalid_json', 'Serve un oggetto JSON.')
            address = self.client_address[0]
            if path == '/api/auth/setup/check':
                app.auth.check_setup_code(body.get('code'), address)
                self._json(200, {'ok': True})
            elif path == '/api/auth/setup':
                token = app.auth.setup(body.get('code'), body.get('user'), body.get('password'), address)
                self._json(200, {'ok': True}, [('Set-Cookie', self._session_cookie(token))])
            elif path == '/api/auth/login':
                token = app.auth.login(body.get('user'), body.get('password'), address)
                print('Sito: amministratore entrato', flush=True)
                self._json(200, {'ok': True}, [('Set-Cookie', self._session_cookie(token))])
            elif path == '/api/auth/password':
                session = self._session()
                if session is None:
                    self._problem(401, 'login_required', LOGIN_MESSAGE, True)
                    return
                token = app.auth.change_password(session, body.get('current'), body.get('password'), address)
                # Every other browser is signed out: this one, with its new session, is the only one.
                state = {'configured': True, 'signed_in': True, 'local': False, 'user': app.auth.username(),
                         'sessions': 1}
                self._json(200, {'ok': True, 'state': state}, [('Set-Cookie', self._session_cookie(token))])
            else:
                app.auth.logout(self._session(), everywhere=path == '/api/auth/logout-all')
                self._json(200, {'ok': True}, [('Set-Cookie', self._expired_cookie())])
        # [site-login] - END

        def _get(self, path):
            # [site-login] - START
            if path == '/api/auth/state':
                self._json(200, self._auth_state())
                return
            if path == '/account':
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200, 'text/html; charset=utf-8',
                           render_account(nonce, app.auth is not None).encode('utf-8'), csp)
                return
            if path in AUTH_ROUTES:
                self._problem(405, 'method_not_allowed', 'Usa POST.', True)
                return
            # [site-login] - END
            if path == '/' or path == '/index.html':
                nonce = secrets.token_urlsafe(18)
                page = render_index(app.route_summaries(), db_path(app.root), nonce)
                self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'),
                           "default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                           "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                return
            if path == '/api/routes':
                self._json(200, {'routes': app.route_summaries()})
                return
            # [control-panel] - START
            if path == '/pannello':
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200, 'text/html; charset=utf-8', render_panel(nonce).encode('utf-8'), csp)
                return
            if path == '/stato':
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200, 'text/html; charset=utf-8', render_status(nonce).encode('utf-8'), csp)
                return
            if path == '/api/panel/status':
                self._json(200, app.panel.status())
                return
            match = re.fullmatch(r'/api/panel/job/([0-9a-z_-]+)/log', path)
            if match:
                offset = parse_qs(urlsplit(self.path).query).get('offset', ['0'])[0]
                self._json(200, app.panel.log(match.group(1), int(offset) if offset.isdigit() else 0))
                return
            # [control-panel] - END
            match = re.fullmatch(r'/api/route/([^/]+)/reviews', path)
            if match:
                self._send_reviews(match.group(1))
                return
            if re.fullmatch(r'/api/route/[^/]+/frame/[^/]+/[^/]+', path):
                self._problem(405, 'method_not_allowed', 'Usa POST per salvare un fotogramma.', True)
                return
            match = re.fullmatch(r'/route/([^/]+)', path)
            if match:
                route = self._route_or_none(match.group(1))
                if route is not None:
                    self._redirect('/route/' + quote(route) + '/', None)
                return
            match = re.fullmatch(r'/route/([^/]+)/(index\.html)?', path)
            if match:
                self._send_route_page(match.group(1))
                return
            match = re.fullmatch(r'/route/([^/]+)(/images/[0-9]{8}\.jpg)', path)
            if match:
                self._send_image(match.group(1), match.group(2))
                return
            self._problem(404, 'not_found', 'Pagina non trovata.', path.startswith('/api/'))

        # ---------- REST API /api/v1 and MCP /mcp (token in a header, never a cookie) ----------

        def _external(self, method, path, query):
            if 'token' in query:
                self._problem(400, 'token_in_url', API_TOKEN_MESSAGE, True)
                return
            # MCP transport rule: an Origin header that is not this server is refused (DNS rebinding).
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Richiesta da un’altra origine rifiutata.', True)
                return
            if not self._api_authorized():
                self.close_connection = True
                body = json.dumps({'ok': False, 'error': 'unauthorized', 'message': API_TOKEN_MESSAGE},
                                  ensure_ascii=False).encode('utf-8')
                self._send(401, 'application/json; charset=utf-8', body, DEFAULT_CSP,
                           [('WWW-Authenticate', 'Bearer realm="nnslr-review"')])
                return
            if path == '/mcp':
                self._mcp(method)
                return
            if method == 'POST':
                self._api_post(path)
            elif method == 'GET':
                self._api_get(path, query)
            else:
                self._problem(405, 'method_not_allowed', 'Metodo non consentito.', True)

        def _api_authorized(self):
            supplied = self.headers.get('Authorization')
            if isinstance(supplied, str) and supplied[:7].lower() == 'bearer ':
                return app.api_token_matches(supplied[7:].strip())
            return app.api_token_matches(self.headers.get('X-Review-Token'))

        def _api_get(self, path, query):
            values = {}
            for name, items in query.items():
                values[name] = items[0]
            if path in ('/api/v1', '/api/v1/'):
                self._json(200, {'ok': True, 'api': 'nnslr-review', 'version': 1,
                                 'mcp': app.mcp_bridge() is not None, 'documentation': 'docs/API_MCP.md'})
                return
            if path == '/api/v1/schema':
                self._json(200, review_schema())
                return
            if path == '/api/v1/routes':
                self._json(200, {'routes': app.list_routes()})
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)', path)
            if match:
                self._json(200, app.progress(match.group(1), suspicious_limit=values.get('suspicious_limit', 100)))
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/frames', path)
            if match:
                self._json(200, app.list_frames(match.group(1), frame_filter=values.get('filter'),
                                                cursor=values.get('cursor'), limit=values.get('limit')))
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/next', path)
            if match:
                self._json(200, app.next_frame(match.group(1), frame_filter=values.get('filter'),
                                               after=values.get('after')))
                return
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/frames/([0-9]{1,9})/([0-9]{1,9})(/[a-z]+)?', path)
            if match is None:
                self._problem(404, 'not_found', 'Indirizzo dell’API non valido.', True)
                return
            route = match.group(1)
            key = match.group(2) + '/' + match.group(3)
            action = match.group(4)
            if action is None:
                self._json(200, app.get_frame(route, key))
            elif action == '/context':
                self._json(200, app.frame_context(route, key, before=values.get('before'), after=values.get('after')))
            elif action == '/history':
                self._json(200, app.history(route, key))
            elif action == '/image':
                image = app.image(route, key, variant=values.get('variant'), crop=values.get('crop'),
                                  long_side=values.get('long_side'))
                self._send(200, image['mime_type'], image['data'])
            elif action == '/review':
                self._problem(405, 'method_not_allowed', 'Usa POST per salvare un fotogramma.', True)
            else:
                self._problem(404, 'not_found', 'Indirizzo dell’API non valido.', True)

        def _api_post(self, path):
            match = re.fullmatch(r'/api/v1/routes/([^/]+)/frames/([0-9]{1,9})/([0-9]{1,9})/review', path)
            if match is None:
                self._problem(404, 'not_found', 'Indirizzo dell’API non valido.', True)
                return
            content_type = self.headers.get('Content-Type', '')
            if content_type.split(';')[0].strip().lower() != 'application/json':
                self._problem(415, 'unsupported_media_type', 'Sono accettati solo dati JSON.', True)
                return
            body = self._read_json()
            key = match.group(2) + '/' + match.group(3)
            self._json(200, app.save_review(match.group(1), key, body, client=self.client_address[0],
                                            channel='rest'))

        def _mcp(self, method):
            bridge = app.mcp_bridge()
            if bridge is None:
                self._problem(501, 'mcp_unavailable', 'Server MCP non disponibile: installa le dipendenze '
                                                      'opzionali con  pip install -e ".[mcp]"  e riavvia il server.',
                              True)
                return
            if method != 'POST':
                # Stateless JSON server: no server stream (GET) and no session to end (DELETE). The MCP
                # transport lets a server answer both with 405.
                body = json.dumps({'ok': False, 'error': 'method_not_allowed',
                                   'message': 'Questo server MCP accetta solo POST.'}).encode('utf-8')
                self._send(405, 'application/json; charset=utf-8', body, DEFAULT_CSP, [('Allow', 'POST')])
                return
            if self.headers.get('Transfer-Encoding') is not None:
                raise ReviewRejected(411, 'length_required', 'Richiesta senza Content-Length.')
            raw_length = self.headers.get('Content-Length')
            if raw_length is None or not raw_length.strip().isdigit():
                raise ReviewRejected(411, 'length_required', 'Richiesta senza Content-Length.')
            length = int(raw_length)
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise ReviewRejected(413, 'too_large', 'Richiesta troppo grande (massimo 512 KiB).')
            body = self.rfile.read(length)
            headers = []
            for name, value in self.headers.items():
                # The token stays in this process: the MCP layer never sees it.
                if name.lower() in ('authorization', 'x-review-token', 'cookie'):
                    continue
                headers.append((name, value))
            status, response_headers, data = bridge.handle(method, headers, body, self.client_address,
                                                           self.server.server_address)
            extra = []
            content_type = 'application/json'
            for name, value in response_headers:
                lowered = name.lower()
                if lowered == 'content-type':
                    content_type = value
                elif lowered in ('mcp-session-id', 'mcp-protocol-version', 'allow'):
                    extra.append((name, value))
            self._send(status, content_type, data, DEFAULT_CSP, extra)

        def _route_or_none(self, route):
            # Unknown routes, malformed ids and paths outside the data root all end as 404.
            try:
                parse_route_id(route)
                app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Route sconosciuta o senza report di pre-annotazione.',
                              False)
                return None
            return route

        def _send_route_page(self, route):
            try:
                directory, rows, by_key = app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Route sconosciuta o senza report di pre-annotazione.', False)
                return
            hints_by_frame, hint_stats = app.hints(route, rows)
            suspicious = app.suspicious(route, directory.name, rows, hints_by_frame)
            nonce = secrets.token_urlsafe(18)
            page = render_review(rows, directory.name, route_id=route, server=True, nonce=nonce,
                                 hints=hints_by_frame, hint_stats=hint_stats, suspicious=suspicious,
                                 routes=app.route_links())
            csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                   "img-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'), csp)

        def _send_image(self, route, name):
            try:
                directory, rows, by_key = app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Route sconosciuta o senza report di pre-annotazione.', False)
                return
            path = public_file(directory, name)
            if path is None:
                self._problem(404, 'not_found', 'Immagine non trovata.', False)
                return
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(path.stat().st_size))
            self._security_headers(DEFAULT_CSP)
            self.end_headers()
            with path.open('rb') as source:
                while block := source.read(65536):
                    self.wfile.write(block)

        def _send_reviews(self, route):
            try:
                directory, rows, by_key = app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Route sconosciuta o senza report di pre-annotazione.', True)
                return
            reviews = []
            confirmed = 0
            for review in app.store.reviews(route, directory.name):
                reviews.append(public_review(review))
                if review['confirmed']:
                    confirmed = confirmed + 1
            hints_by_frame, hint_stats = app.hints(route, rows)
            suspicious = app.suspicious(route, directory.name, rows, hints_by_frame)
            self._json(200, {'route_id': route, 'run_id': directory.name, 'frame_count': len(rows),
                             'confirmed_count': confirmed, 'reviews': reviews, 'suspicious': suspicious})

        def _post(self, path):
            # [site-login] - START
            if path in AUTH_ROUTES:
                # Same protections as saving a frame: same origin, JSON only (no form can send it).
                if not self._same_origin():
                    self._problem(403, 'cross_origin', 'Richiesta da un’altra origine rifiutata.', True)
                    return
                content_type = self.headers.get('Content-Type', '')
                if content_type.split(';')[0].strip().lower() != 'application/json':
                    self._problem(415, 'unsupported_media_type', 'Sono accettati solo dati JSON.', True)
                    return
                self._auth_post(path, self._read_json())
                return
            # [site-login] - END
            # [control-panel] - START
            run = re.fullmatch(r'/api/panel/run/([a-z_]+)', path)
            stop = re.fullmatch(r'/api/panel/job/([0-9a-z_-]+)/stop', path)
            if run or stop:
                # Same protections as saving a frame: same origin, JSON only (no form can send it).
                if not self._same_origin():
                    self._problem(403, 'cross_origin', 'Richiesta da un’altra origine rifiutata.', True)
                    return
                content_type = self.headers.get('Content-Type', '')
                if content_type.split(';')[0].strip().lower() != 'application/json':
                    self._problem(415, 'unsupported_media_type', 'Sono accettati solo dati JSON.', True)
                    return
                self._read_json()
                if run:
                    job = app.panel.start(run.group(1))
                    print(f'Pannello: avviata {job["label"]} ({job["id"]})', flush=True)
                    self._json(200, {'ok': True, **job})
                else:
                    self._json(200, {'ok': True, **app.panel.stop(stop.group(1))})
                return
            # [control-panel] - END
            match = re.fullmatch(r'/api/route/([^/]+)/frame/([0-9]{1,9})/([0-9]{1,9})', path)
            if match is None:
                self._problem(404, 'not_found', 'Indirizzo di salvataggio non valido.', True)
                return
            if not self._same_origin():
                self._problem(403, 'cross_origin', 'Richiesta da un’altra origine rifiutata.', True)
                return
            content_type = self.headers.get('Content-Type', '')
            if content_type.split(';')[0].strip().lower() != 'application/json':
                self._problem(415, 'unsupported_media_type', 'Sono accettati solo dati JSON.', True)
                return
            body = self._read_json()
            route = match.group(1)
            frame_key = str(int(match.group(2))) + '/' + str(int(match.group(3)))
            try:
                directory, rows, by_key = app.report(route)
            except (PreannotationError, ValueError, OSError):
                self._problem(404, 'unknown_route', 'Route sconosciuta o senza report di pre-annotazione.', True)
                return
            report_row = by_key.get(frame_key)
            if report_row is None:
                self._problem(404, 'unknown_frame', 'Fotogramma non presente nel report di questa route.', True)
                return
            review = save_frame_review(app.store, app.root, route_id=route, run_id=directory.name,
                                       report_row=report_row, body=body, client=self.client_address[0])
            reviewer = terminal_text(review['reviewer'])
            if review['confirmed']:
                print(f'Salvato: {route} fotogramma {frame_key}, revisione {review["revision"]} '
                      f'({reviewer})', flush=True)
            else:
                print(f'Conferma annullata: {route} fotogramma {frame_key} ({reviewer})', flush=True)
            # A confirmation can change which neighbours are suspicious (rule d): send the new flags.
            hints_by_frame, hint_stats = app.hints(route, rows)
            suspicious = app.suspicious(route, directory.name, rows, hints_by_frame)
            self._json(200, {'ok': True, 'review': public_review(review),
                             'confirmed_count': app.store.confirmed_count(route, directory.name),
                             'suspicious': suspicious})

        def _same_origin(self):
            fetch_site = self.headers.get('Sec-Fetch-Site')
            if fetch_site is not None and fetch_site not in ('same-origin', 'none'):
                return False
            origin = self.headers.get('Origin')
            if origin is None:
                return True
            host = self.headers.get('Host')
            if not host:
                return False
            # Same host name and port as the request; https as well, for a reverse proxy
            # (Traefik) that terminates TLS and keeps the Host header.
            supplied = origin.strip().lower()
            if supplied == ('http://' + host.strip()).lower():
                return True
            if supplied == ('https://' + host.strip()).lower():
                return True
            return False

        def _read_json(self):
            if self.headers.get('Transfer-Encoding') is not None:
                raise ReviewRejected(411, 'length_required', 'Richiesta senza Content-Length.')
            raw_length = self.headers.get('Content-Length')
            if raw_length is None or not raw_length.strip().isdigit():
                raise ReviewRejected(411, 'length_required', 'Richiesta senza Content-Length.')
            length = int(raw_length)
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                raise ReviewRejected(413, 'too_large', 'Revisione troppo grande (massimo 512 KiB).')
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ReviewRejected(400, 'invalid_json', 'Richiesta incompleta.')
            try:
                return strict_json_loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, ValueError, TypeError) as exc:
                raise ReviewRejected(400, 'invalid_json', 'Dati JSON non validi.') from exc

        def _security_headers(self, csp):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            self.send_header('Cross-Origin-Resource-Policy', 'same-origin')
            self.send_header('Content-Security-Policy', csp)

        def _send(self, status, content_type, body, csp=DEFAULT_CSP, extra=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self._security_headers(csp)
            if extra is not None:
                for name, value in extra:
                    self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload, extra=None):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._send(status, 'application/json; charset=utf-8', body, DEFAULT_CSP, extra)

        def _problem(self, status, reason, message, api):
            if api:
                self._json(status, {'ok': False, 'error': reason, 'message': message})
            else:
                nonce = secrets.token_urlsafe(18)
                self._send(status, 'text/html; charset=utf-8', render_problem(status, message, nonce).encode('utf-8'),
                           "default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                           "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

        def _redirect(self, location, cookie):
            extra = [('Location', location)]
            if cookie is not None:
                extra.append(('Set-Cookie', cookie))
            self._send(303, 'text/plain; charset=utf-8', b'', DEFAULT_CSP, extra)

    return Handler


class _ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class):
        self._connection_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(server_address, handler_class)

    def process_request(self, request, client_address):
        if not self._connection_slots.acquire(blocking=False):
            # Every slot is busy: close this connection now instead of starting another thread.
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            # The thread did not start, so it will never release its slot.
            self._connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class _ReviewServerIPv6(_ReviewServer):
    address_family = socket.AF_INET6


def create_review_server(root: Path, *, host: str = '127.0.0.1', port: int = 8765, token: str | None = None,
                         require_token: bool | None = None):
    """Bind the server (not started). The token is required whenever the address is not loopback."""
    from nnslr_tools.review_db import ReviewStore, db_path
    if require_token is None:
        require_token = not is_loopback(host)
    if require_token and token is None:
        token = review_token()
    store = ReviewStore(db_path(root))
    # /api/v1 and /mcp always need a token, also when the page on 127.0.0.1 does not.
    api_token = token
    if api_token is None:
        api_token = review_token()
    app = ReviewApp(root, store, token=token, require_token=require_token, api_token=api_token)
    if ':' in host:
        server = _ReviewServerIPv6((host, port), make_handler(app))
    else:
        server = _ReviewServer((host, port), make_handler(app))
    # One cookie per port, so two review servers on one machine do not overwrite each other.
    app.cookie_name = 'nnslr_review_' + str(server.server_port)
    return server, app


def _base_urls(host: str, port: int) -> list[str]:
    if host in ('0.0.0.0', '', '::'):
        names = lan_addresses()
        if not names:
            names = [socket.gethostname()]
    elif ':' in host:
        names = ['[' + host + ']']
    else:
        names = [host]
    urls = []
    for name in names:
        urls.append(f'http://{name}:{port}')
    return urls


def serve_review(route: str | None = None, *, data_root: Path | None=None, host: str='127.0.0.1',
                 port: int=8765) -> None:
    root=resolve_data_root(data_root)
    if route is not None:
        latest_report(root,route)
    import sqlite3
    try:
        server, app = create_review_server(root, host=host, port=port)
    except (OSError, sqlite3.Error) as exc:
        raise PreannotationError(f'cannot start review server: {exc}') from exc
    from nnslr_tools.review_db import db_path
    # A token fixed in NNSLR_REVIEW_TOKEN (the systemd service) is never printed:
    # the output may end in a log. A random token must be printed, or nobody could use it.
    token_from_env = bool(os.environ.get(TOKEN_ENV))
    with server:
        port = server.server_port
        # [site-login] - START
        # The address alone: on the LAN the browser signs in (the token is the setup code of the first visit).
        suffix = ''
        # [site-login] - END
        path = '/'
        if route is not None:
            path = '/route/' + route + '/'
        print(f'Revisione NNSLR in ascolto su {host}:{port}', flush=True)
        print('Apri nel browser:', flush=True)
        for base in _base_urls(host, port):
            print(f'  {base}{path}{suffix}', flush=True)
        if route is not None:
            print(f'Elenco di tutte le route: {_base_urls(host, port)[0]}/{suffix}', flush=True)
        # [site-login] - START
        if app.require_token and app.auth.configured():
            print(f'Accedi come amministratore ({app.auth.username()}).', flush=True)
        elif app.require_token and token_from_env:
            print(f'Prima visita: crea l’amministratore con il codice di configurazione, cioè il token di '
                  f'{TOKEN_ENV} (per il servizio: /etc/speedlimits-web.env), che non viene stampato.', flush=True)
        elif app.require_token:
            print(f'Prima visita: crea l’amministratore con questo codice di configurazione: {app.token} '
                  f'(vale finché il server è acceso; fissalo con {TOKEN_ENV}).', flush=True)
        # [site-login] - END
        else:
            print(f'Solo da questo computer. Da un altro: ssh -L {port}:127.0.0.1:{port} <server>, '
                  'oppure riavvia con --host 0.0.0.0', flush=True)
        print(f'Database revisioni: {db_path(root)}', flush=True)
        if app.token is None and not token_from_env:
            # Loopback without a page token: the API token is random and must be shown to be usable.
            print(f'API REST /api/v1 e MCP /mcp: header "Authorization: Bearer {app.api_token}" '
                  f'(fissalo con {TOKEN_ENV}).', flush=True)
        else:
            print(f'API REST /api/v1 e MCP /mcp: header "Authorization: Bearer <{TOKEN_ENV}>" (docs/API_MCP.md).',
                  flush=True)
        print('Ogni “Conferma” viene salvata subito nel database. Ctrl+C per fermare.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()
# [preannotation] - END
