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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from nnslr_tools.manifest import parse_route_id
from nnslr_tools.preannotate import PreannotationError, _inside, resolve_data_root
from nnslr_tools.review_hints import empty_stats, frame_sizes_of, load_hints, suspicious_frames

TOKEN_ENV = 'NNSLR_REVIEW_TOKEN'
MAX_BODY_BYTES = 512 * 1024
# A connection that sends nothing for this many seconds is closed, and at most
# MAX_CONNECTIONS are open at once, so idle clients cannot use up threads.
REQUEST_TIMEOUT_SECONDS = 30
MAX_CONNECTIONS = 64
TOKEN_MESSAGE = 'Accesso negato: apri l’indirizzo con il token stampato all’avvio del server.'
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


class ReviewApp:
    """State shared by the request threads: data root, review store, access token, report cache."""

    def __init__(self, root: Path, store, *, token: str | None, require_token: bool):
        from nnslr_tools.panel import Panel
        self.root = root
        self.store = store
        self.token = token
        self.require_token = require_token
        self.cookie_name = 'nnslr_review'
        self._reports = {}
        self._reports_lock = threading.Lock()
        # [control-panel] - START
        self.panel = Panel(root)
        # [control-panel] - END

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

    def hints(self, route: str, rows: list[dict]) -> tuple[dict, dict]:
        """Second-opinion hints of a route, read again at every call (the files may change)."""
        return load_hints(self.root, route, frame_sizes_of(rows))

    def suspicious(self, route: str, run_id: str, rows: list[dict], hints_by_frame: dict) -> dict:
        """Suspicious frames of a route given the reviews confirmed so far."""
        confirmed = {}
        for review in self.store.reviews(route, run_id, confirmed_only=True):
            confirmed[review['frame_key']] = review['payload'].get('detections', [])
        return suspicious_frames(rows, hints_by_frame, confirmed, hints_available=bool(hints_by_frame))

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


# Style of the route index and of the error pages: the same tokens, top bar and cards as the
# review page. Light only, on white.
INDEX_STYLE = """
:root {
  --bg: #ffffff;
  --surface: #ffffff;
  --surface-2: #f5f5f7;
  --fill: rgba(118, 118, 128, 0.12);
  --fill-strong: rgba(118, 118, 128, 0.22);
  --seg-on: #ffffff;
  --fg: #1d1d1f;
  --muted: #6e6e73;
  --line: rgba(0, 0, 0, 0.08);
  --bar: rgba(251, 251, 253, 0.78);
  --stage: #0b0b0c;
  --accent: #0071e3;
  --button: #0071e3;
  --on-accent: #ffffff;
  --yellow: #f5b800;
  --todo: #86868b;
  --ok: #1f8a3b;
  --ok-fill: #34c759;
  --ok-tint: rgba(52, 199, 89, 0.14);
  --warn: #b85000;
  --warn-fill: #ff9f0a;
  --warn-tint: rgba(255, 159, 10, 0.15);
  --danger: #d70015;
  --danger-fill: #ff3b30;
  --danger-tint: rgba(255, 59, 48, 0.1);
  --sus: #8944ab;
  --sus-fill: #af52de;
  --sus-tint: rgba(175, 82, 222, 0.1);
  --hint: #0066cc;
  --hint-tint: rgba(0, 113, 227, 0.07);
  --focus: #0071e3;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.03), 0 6px 20px rgba(0, 0, 0, 0.05);
  --card-border: rgba(0, 0, 0, 0.08);
  --sign-face: #ffffff;
  --sign-ink: #111111;
  --sign-ring: #e3001b;
  /* Overlay colours sit on the dark photo stage. */
  --ov-speed: #ff5964;
  --ov-cancel: #d4d9df;
  --ov-other: #ffa94d;
  --ov-none: #8a929c;
  --ov-selected: #ffd60a;
  --ov-hint: #64d2ff;
  --ov-text: #fff3b0;
  --ov-halo: #0d0f12;
  --font-ui: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
  --font-data: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --radius: 18px;
  --control: 32px;
  color-scheme: light;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; text-size-adjust: 100%; }
html, body { min-height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-ui);
  font-size: 15px;
  line-height: 1.45;
  letter-spacing: -0.006em;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}
body::before {
  content: "";
  position: fixed;
  inset: 0 0 auto;
  height: env(safe-area-inset-top, 0px);
  z-index: 10;
  background: var(--bar);
}
:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }

/* Top bar: the same as on the review page, brand in the same place. */
.topbar {
  position: sticky;
  top: env(safe-area-inset-top, 0px);
  z-index: 5;
  background: var(--bar);
  -webkit-backdrop-filter: saturate(180%) blur(20px);
  backdrop-filter: saturate(180%) blur(20px);
  border-bottom: 1px solid var(--line);
  padding-left: max(20px, env(safe-area-inset-left, 0px));
  padding-right: max(20px, env(safe-area-inset-right, 0px));
}
.topbar-inner { display: flex; align-items: center; gap: 18px; height: 52px; max-width: 1600px; margin: 0 auto; }
.brand { display: flex; align-items: center; gap: 10px; min-width: 0; color: inherit; text-decoration: none; flex: none; }
.brand-mark {
  width: 26px; height: 26px; border-radius: 50%;
  background: var(--sign-face); border: 3.5px solid var(--sign-ring);
  display: grid; place-items: center; flex: none;
  color: var(--sign-ink); font-weight: 700; font-size: 10px; font-variant-numeric: tabular-nums;
}
.brand-name { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
.topnav { display: inline-flex; gap: 2px; background: var(--fill); border-radius: 9px; padding: 2px; flex: none; }
.topnav a {
  display: inline-flex; align-items: center; min-height: 28px; padding: 4px 14px; border-radius: 7px;
  font-size: 13px; font-weight: 500; color: var(--fg); text-decoration: none; white-space: nowrap;
}
.topnav a:hover { background: var(--fill); }
.topnav a[aria-current="page"] {
  background: var(--seg-on); box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12), 0 0 0 0.5px rgba(0, 0, 0, 0.04); font-weight: 600;
}

/* Page */
.page {
  max-width: 1600px;
  margin: 0 auto;
  padding-block: 40px 56px;
  padding-left: max(20px, env(safe-area-inset-left, 0px));
  padding-right: max(20px, env(safe-area-inset-right, 0px));
}
.page-head { margin: 0 0 28px; }
.page-title { margin: 0; font-size: 34px; line-height: 1.12; font-weight: 700; letter-spacing: -0.022em; }
.lead { margin: 8px 0 0; font-size: 17px; color: var(--muted); max-width: 62ch; }
code { font-family: var(--font-data); font-size: 12.5px; background: var(--fill); border-radius: 6px; padding: 2px 6px; overflow-wrap: anywhere; }

/* Route cards: the whole card is the link. */
.cards {
  list-style: none; margin: 0; padding: 0;
  display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 320px), 1fr)); gap: 20px;
}
.card {
  display: flex; flex-direction: column; height: 100%;
  background: var(--surface); border: 1px solid var(--card-border); border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 22px 24px 20px; color: inherit; text-decoration: none;
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05), 0 14px 36px rgba(0, 0, 0, 0.1); }
.route { font-family: var(--font-data); font-size: 12.5px; color: var(--muted); word-break: break-all; }
.count { margin-top: 10px; font-size: 26px; line-height: 1.15; font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
.count .unit { color: var(--muted); font-weight: 500; }
.bar { display: block; height: 4px; margin-top: 14px; border-radius: 2px; background: var(--fill); overflow: hidden; }
.bar span { display: block; height: 100%; background: var(--ok-fill); border-radius: 2px; }
/* Each figure keeps its words together; the dot in front of the first figure of a line is clipped. */
.stats {
  display: flex; flex-wrap: wrap; column-gap: 16px; row-gap: 2px; overflow: hidden;
  margin-top: 12px; font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums;
}
.stats span { margin-left: -16px; white-space: nowrap; }
.stats span::before { content: "·"; display: inline-block; width: 16px; text-align: center; color: var(--muted); font-weight: 400; }
.stats .sus { color: var(--sus); font-weight: 600; }
.open { margin-top: 18px; font-size: 15px; font-weight: 600; color: var(--accent); }
.open.again { color: var(--muted); font-weight: 500; }

/* Empty index and error pages */
.state-box {
  background: var(--surface); border: 1px solid var(--card-border); border-radius: var(--radius); box-shadow: var(--shadow);
  padding: 32px 28px; max-width: 720px; display: flex; flex-direction: column; gap: 12px;
}
.state-box.center { margin: 32px auto; }
.state-box h1, .state-box h2 { margin: 0; font-size: 24px; line-height: 1.2; font-weight: 600; letter-spacing: -0.02em; }
.state-box p { margin: 0; color: var(--muted); max-width: 62ch; }
.state-box .code { font-size: 13px; }
.back { align-self: flex-start; margin-top: 6px; color: var(--accent); font-weight: 500; text-decoration: none; }
.back:hover { text-decoration: underline; }

/* Technical details, out of the way */
.foot { margin-top: 48px; font-size: 13px; color: var(--muted); }
.foot summary { display: inline-flex; align-items: center; gap: 6px; padding: 4px 0; cursor: pointer; list-style: none; }
.foot summary::-webkit-details-marker { display: none; }
.foot summary::after { content: "›"; font-size: 16px; line-height: 1; transition: transform 0.2s ease; }
.foot[open] summary::after { transform: rotate(90deg); }
.foot summary:hover { color: var(--fg); }
.foot p { margin: 8px 0 0; }

@media (max-width: 640px) {
  .brand-name { display: none; }
  .page { padding-block: 28px 40px; }
  .page-head { margin-bottom: 20px; }
  .page-title { font-size: 28px; }
  .lead { font-size: 15px; }
  .cards { gap: 14px; }
  .card { padding: 18px 20px; }
  .state-box { padding: 24px 20px; }
  .state-box.center { margin: 8px auto; }
}
@media (pointer: coarse) {
  .topnav a { min-height: 36px; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
"""


def _plural(count: int, one: str, many: str) -> str:
    if count == 1:
        return '1 ' + one
    return str(count) + ' ' + many


def _page_head(title: str, current: str = '') -> list[str]:
    """Head and top bar of the server pages. current is 'route' on the route list, '' elsewhere."""
    route_link = '<a href="/">Route</a>'
    if current == 'route':
        route_link = '<a href="/" aria-current="page">Route</a>'
    return [
        '<!doctype html><html lang="it"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="color-scheme" content="light">',
        '<title>' + html.escape(title) + '</title>',
        '<style>' + INDEX_STYLE + '</style></head><body>',
        '<header class="topbar"><div class="topbar-inner">'
        '<a class="brand" href="/" title="Elenco delle route"><span class="brand-mark" aria-hidden="true">50</span>'
        '<span class="brand-name">Revisione cartelli</span></a>'
        '<nav class="topnav" aria-label="Sezioni">' + route_link + '<a href="/pannello">Pannello</a></nav>'
        '</div></header>',
    ]


# [control-panel] - START
def render_panel(nonce: str) -> str:
    """The control panel page; it loads its data from /api/panel/status."""
    return files('nnslr_tools').joinpath('panel.html').read_text().replace('__NNSLR_NONCE__', nonce)
# [control-panel] - END


def render_index(summaries: list[dict], database: Path) -> str:
    lines = _page_head('Revisione cartelli · Route', 'route')
    lines.append('<main class="page">')
    lines.append('<div class="page-head"><h1 class="page-title">Route</h1>'
                 '<p class="lead">Scegli una route da revisionare. Ogni conferma viene salvata subito '
                 'sul server.</p></div>')
    if not summaries:
        lines.append('<div class="state-box"><h2>Nessuna route da revisionare</h2><p>Prepara una route con '
                     '<code>nnslr extract-frames</code> e <code>nnslr preannotate</code>, poi ricarica questa '
                     'pagina.</p></div>')
    else:
        lines.append('<ul class="cards">')
        for summary in summaries:
            total = summary['frame_count']
            confirmed = summary['confirmed_count']
            percent = 0
            if total:
                percent = round(confirmed * 100 / total)
            # What opening the route means: start, carry on, or look again at a finished one.
            card_class = 'card'
            action_class = 'open'
            action = 'Riprendi →'
            if confirmed == 0:
                action = 'Inizia →'
            elif total and confirmed >= total:
                card_class = 'card done'
                action_class = 'open again'
                action = 'Ricontrolla →'
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
            lines.append(
                '<li><a class="' + card_class + '" href="' + html.escape(summary['url']) + '">'
                + '<span class="route">' + html.escape(summary['route_id']) + '</span>'
                + '<span class="count">' + str(confirmed) + ' di ' + str(total)
                + ' <span class="unit">confermati</span></span>'
                + '<span class="bar" aria-hidden="true"><span style="width:' + str(percent) + '%"></span></span>'
                + '<span class="stats">' + ''.join(stats) + '</span>'
                + '<span class="' + action_class + '">' + action + '</span></a></li>')
        lines.append('</ul>')
    lines.append('<details class="foot"><summary>Dettagli tecnici</summary>'
                 '<p>Database delle revisioni: <code>' + html.escape(str(database)) + '</code></p></details>')
    lines.append('</main></body></html>')
    return '\n'.join(lines)


def render_problem(status: int, message: str) -> str:
    if status == 404:
        title = 'Pagina non trovata'
    elif status == 403:
        title = 'Accesso negato'
    else:
        title = 'Errore'
    lines = _page_head('Revisione cartelli · ' + title)
    lines.append('<main class="page"><div class="state-box center">')
    lines.append('<h1>' + html.escape(title) + '</h1>')
    # "Pagina non trovata." under the title "Pagina non trovata" would only repeat it.
    if message.rstrip('.') != title:
        lines.append('<p>' + html.escape(message) + '</p>')
    lines.append('<p class="code">Codice HTTP ' + str(status) + '</p>')
    # Every 403 page (token missing or wrong, host not allowed) would come back here from the list.
    if status != 403:
        lines.append('<a class="back" href="/">← Elenco delle route</a>')
    lines.append('</div></main></body></html>')
    return '\n'.join(lines)


def make_handler(app: ReviewApp):
    from nnslr_tools.annotations import strict_json_loads
    from nnslr_tools.panel import PanelError
    from nnslr_tools.review_db import ReviewRejected, db_path, public_review, save_frame_review

    class Handler(BaseHTTPRequestHandler):
        server_version = 'speedlimits-web'
        sys_version = ''
        # Socket timeout: a client that stops sending is disconnected, even before the token check.
        timeout = REQUEST_TIMEOUT_SECONDS

        def do_GET(self):
            self._dispatch('GET')

        def do_POST(self):
            self._dispatch('POST')

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
            # Plain "name=value; name=value" split: the browser also sends the
            # cookies of other web apps on this host (any port), and a value our
            # parser does not like must not hide the review cookie.
            for raw in self.headers.get_all('Cookie', []):
                for part in raw.split(';'):
                    name, separator, value = part.strip().partition('=')
                    if not separator:
                        continue
                    if name.strip() == app.cookie_name and app.token_matches(value.strip()):
                        return True
            return False

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
            # [control-panel] - START
            if path == '/pannello':
                nonce = secrets.token_urlsafe(18)
                csp = ("default-src 'none'; script-src 'nonce-" + nonce + "'; style-src 'unsafe-inline'; "
                       "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
                self._send(200, 'text/html; charset=utf-8', render_panel(nonce).encode('utf-8'), csp)
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
    # A token fixed in NNSLR_REVIEW_TOKEN (the systemd service) is never printed:
    # the output may end in a log. A random token must be printed, or nobody could use it.
    token_from_env = bool(os.environ.get(TOKEN_ENV))
    with server:
        port = server.server_port
        suffix = ''
        if app.require_token:
            if token_from_env:
                suffix = '?token=<' + TOKEN_ENV + '>'
            else:
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
        if app.require_token and token_from_env:
            print(f'Il token è quello di {TOKEN_ENV} (per il servizio: /etc/speedlimits-web.env) '
                  'e non viene stampato.', flush=True)
        elif app.require_token:
            print(f'Il token resta valido finché il server è acceso (fissalo con {TOKEN_ENV}).', flush=True)
        else:
            print(f'Solo da questo computer. Da un altro: ssh -L {port}:127.0.0.1:{port} <server>, '
                  'oppure riavvia con --host 0.0.0.0', flush=True)
        print(f'Database revisioni: {db_path(root)}', flush=True)
        print('Ogni “Conferma” viene salvata subito nel database. Ctrl+C per fermare.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
# [preannotation] - END
