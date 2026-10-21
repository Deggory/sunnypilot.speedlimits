# [preannotation] - START
"""Review page and review server.

The server reads the pre-annotation reports (read-only) and writes the reviewed
frames into the review database (:mod:`nnslr_tools.review_db`). It listens on
127.0.0.1 by default; on any other address every request needs the access token.
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
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from nnslr_tools.manifest import parse_route_id
from nnslr_tools.preannotate import PreannotationError, _inside, resolve_data_root

TOKEN_ENV = 'NNSLR_REVIEW_TOKEN'
MAX_BODY_BYTES = 512 * 1024
TOKEN_MESSAGE = 'Accesso negato: apri l’indirizzo con il token stampato all’avvio del server.'
DEFAULT_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


def render_review(rows: list[dict], run_id: str, *, route_id: str = '', server: bool = False,
                  nonce: str = '') -> str:
    template=files('nnslr_tools').joinpath('review.html').read_text()
    data={'run_id':run_id,'route_id':route_id,'server':server,'frames':rows}
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


class ReviewApp:
    """State shared by the request threads: data root, review store, access token, report cache."""

    def __init__(self, root: Path, store, *, token: str | None, require_token: bool):
        self.root = root
        self.store = store
        self.token = token
        self.require_token = require_token
        self.cookie_name = 'nnslr_review'
        self._reports = {}
        self._reports_lock = threading.Lock()

    def token_matches(self, supplied) -> bool:
        if self.token is None or not isinstance(supplied, str):
            return False
        return secrets.compare_digest(supplied.encode('utf-8'), self.token.encode('utf-8'))

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

    def report(self, route: str) -> tuple[Path, list[dict], dict]:
        """Current report of a route: directory, frame rows, rows by frame_key. Run directories never change."""
        directory = latest_report(self.root, route)
        key = str(directory)
        with self._reports_lock:
            cached = self._reports.get(key)
        if cached is None:
            rows = report_rows(directory)
            by_key = {}
            for row in rows:
                by_key[row.get('frame_key')] = row
            cached = (rows, by_key)
            with self._reports_lock:
                self._reports[key] = cached
        return directory, cached[0], cached[1]

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
            summaries.append({
                'route_id': route,
                'run_id': directory.name,
                'frame_count': len(rows),
                'frames_with_proposals': with_proposals,
                'confirmed_count': self.store.confirmed_count(route, directory.name),
                'url': '/route/' + route + '/',
            })
        return summaries


INDEX_STYLE = (
    ':root{color-scheme:dark;font:16px system-ui;background:#111722;color:#edf2f8}'
    'body{margin:0;padding:24px;max-width:1100px;margin:auto}h1{font-size:24px;margin:0 0 8px}'
    'p{line-height:1.5;color:#b7c7d9}a{color:#8fd3ff}code{color:#edf2f8;word-break:break-all}'
    '.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:#192332;border-radius:8px}'
    'th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #2c3b50;white-space:nowrap}'
    'td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.done{color:#76e3bd}'
    '@media(max-width:850px){body{padding:12px}}'
)


def render_index(summaries: list[dict], database: Path) -> str:
    lines = [
        '<!doctype html><html lang="it"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>NNSLR · Route da revisionare</title>',
        '<style>' + INDEX_STYLE + '</style>',
        '<h1>NNSLR · Route da revisionare</h1>',
        '<p>Ogni “Conferma fotogramma” viene salvata subito nel database del server: <code>'
        + html.escape(str(database)) + '</code></p>',
    ]
    if not summaries:
        lines.append('<p>Nessun report di pre-annotazione. Prepara una route con <code>nnslr extract-frames</code> '
                     'e <code>nnslr preannotate</code>, poi ricarica questa pagina.</p>')
    else:
        lines.append('<div class="wrap"><table><thead><tr><th>Route</th><th class="n">Fotogrammi</th>'
                     '<th class="n">Con proposte</th><th class="n">Confermati</th><th></th></tr></thead><tbody>')
        for summary in summaries:
            if summary['frame_count'] and summary['confirmed_count'] >= summary['frame_count']:
                done = ' class="n done"'
            else:
                done = ' class="n"'
            lines.append(
                '<tr><td><code>' + html.escape(summary['route_id']) + '</code></td>'
                + '<td class="n">' + str(summary['frame_count']) + '</td>'
                + '<td class="n">' + str(summary['frames_with_proposals']) + '</td>'
                + '<td' + done + '>' + str(summary['confirmed_count']) + ' / ' + str(summary['frame_count']) + '</td>'
                + '<td><a href="' + html.escape(summary['url']) + '">Apri la revisione →</a></td></tr>')
        lines.append('</tbody></table></div>')
    lines.append('</html>')
    return '\n'.join(lines)


def render_problem(status: int, message: str) -> str:
    return ('<!doctype html><html lang="it"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>NNSLR · Revisione</title><style>' + INDEX_STYLE + '</style>'
            '<h1>NNSLR · Revisione</h1><p>' + html.escape(message) + '</p>'
            '<p>Codice HTTP ' + str(status) + '.</p></html>')


def make_handler(app: ReviewApp):
    from nnslr_tools.annotations import strict_json_loads
    from nnslr_tools.review_db import ReviewRejected, db_path, public_review, save_frame_review

    class Handler(BaseHTTPRequestHandler):
        server_version = 'nnslr-review'
        sys_version = ''

        def do_GET(self):
            self._dispatch('GET')

        def do_POST(self):
            self._dispatch('POST')

        def log_message(self, *args):
            pass

        def _dispatch(self, method):
            try:
                self._handle(method)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except ReviewRejected as exc:
                self._problem(exc.status, exc.reason, exc.message, True)
            except Exception as exc:
                self._problem(500, 'server_error', 'Errore interno del server: ' + type(exc).__name__ + '.',
                              self.path.startswith('/api/'))

        def _handle(self, method):
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            api = path.startswith('/api/')
            if not app.host_allowed(self.headers.get('Host')):
                self._problem(403, 'host_not_allowed', 'Nome host non consentito per questo server.', api)
                return
            query = parse_qs(parts.query, keep_blank_values=True)
            if 'token' in query:
                self._accept_token(method, path, query['token'][0], api)
                return
            if not self._authorized():
                self._problem(403, 'token_required', TOKEN_MESSAGE, api)
                return
            if method == 'POST':
                self._post(path)
            else:
                self._get(path)

        def _authorized(self):
            if not app.require_token:
                return True
            if app.token_matches(self.headers.get('X-Review-Token')):
                return True
            raw = self.headers.get('Cookie')
            if raw is None:
                return False
            jar = SimpleCookie()
            try:
                jar.load(raw)
            except CookieError:
                return False
            morsel = jar.get(app.cookie_name)
            if morsel is None:
                return False
            return app.token_matches(morsel.value)

        def _accept_token(self, method, path, supplied, api):
            # Clean target: our own paths only, never "//host" or a backslash.
            target = path
            if not re.fullmatch(r'/[A-Za-z0-9._/-]*', target) or target.startswith('//'):
                target = '/'
            if not app.require_token:
                self._redirect(target, None)
                return
            if method != 'GET' or not app.token_matches(supplied):
                self._problem(403, 'invalid_token',
                              'Token non valido: usa l’indirizzo stampato all’avvio del server.', api)
                return
            cookie = app.cookie_name + '=' + app.token + '; Path=/; HttpOnly; SameSite=Strict'
            self._redirect(target, cookie)

        def _get(self, path):
            if path == '/' or path == '/index.html':
                page = render_index(app.route_summaries(), db_path(app.root))
                self._send(200, 'text/html; charset=utf-8', page.encode('utf-8'),
                           "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
                           "form-action 'none'; frame-ancestors 'none'")
                return
            if path == '/api/routes':
                self._json(200, {'routes': app.route_summaries()})
                return
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
            nonce = secrets.token_urlsafe(18)
            page = render_review(rows, directory.name, route_id=route, server=True, nonce=nonce)
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
            self._json(200, {'route_id': route, 'run_id': directory.name, 'frame_count': len(rows),
                             'confirmed_count': confirmed, 'reviews': reviews})

        def _post(self, path):
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
            if review['confirmed']:
                print(f'Salvato: {route} fotogramma {frame_key}, revisione {review["revision"]} '
                      f'({review["reviewer"]})', flush=True)
            else:
                print(f'Conferma annullata: {route} fotogramma {frame_key} ({review["reviewer"]})', flush=True)
            self._json(200, {'ok': True, 'review': public_review(review),
                             'confirmed_count': app.store.confirmed_count(route, directory.name)})

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
            return origin.strip().lower() == ('http://' + host.strip()).lower()

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

        def _json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self._send(status, 'application/json; charset=utf-8', body)

        def _problem(self, status, reason, message, api):
            if api:
                self._json(status, {'ok': False, 'error': reason, 'message': message})
            else:
                self._send(status, 'text/html; charset=utf-8', render_problem(status, message).encode('utf-8'),
                           "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
                           "form-action 'none'; frame-ancestors 'none'")

        def _redirect(self, location, cookie):
            extra = [('Location', location)]
            if cookie is not None:
                extra.append(('Set-Cookie', cookie))
            self._send(303, 'text/plain; charset=utf-8', b'', DEFAULT_CSP, extra)

    return Handler


class _ReviewServer(ThreadingHTTPServer):
    daemon_threads = True


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
    app = ReviewApp(root, store, token=token, require_token=require_token)
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
    with server:
        port = server.server_port
        suffix = ''
        if app.require_token:
            suffix = '?token=' + app.token
        path = '/'
        if route is not None:
            path = '/route/' + route + '/'
        print(f'Revisione NNSLR in ascolto su {host}:{port}', flush=True)
        print('Apri nel browser:', flush=True)
        for base in _base_urls(host, port):
            print(f'  {base}{path}{suffix}', flush=True)
        if route is not None:
            print(f'Elenco di tutte le route: {_base_urls(host, port)[0]}/{suffix}', flush=True)
        if app.require_token:
            print(f'Il token resta valido finché il server è acceso (fissalo con {TOKEN_ENV}).', flush=True)
        else:
            print(f'Solo da questo computer. Da un altro: ssh -L {port}:127.0.0.1:{port} <server>, '
                  'oppure riavvia con --host 0.0.0.0', flush=True)
        print(f'Database revisioni: {db_path(root)}', flush=True)
        print('Ogni “Conferma fotogramma” viene salvata subito nel database. Ctrl+C per fermare.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
# [preannotation] - END
