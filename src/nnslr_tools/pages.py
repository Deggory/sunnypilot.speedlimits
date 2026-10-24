"""The HTML pages of the review site.

Templates, styles and scripts live in ``nnslr_tools/web``: ``shared.css`` (tokens and frame, the look of
distill-web), ``site.css`` (route list, error pages, sign-in, account), the page templates and
``js/<page>.js``, built from the TypeScript of ``web/src`` by ``helpers/build_web.sh``. Every page gets
its script inline with the nonce of its Content-Security-Policy. Files are read on every request: a
change to them is live at the next page load, a change to this module after a restart.

Placeholders of the templates:

- ``__NNSLR_THEME__``: the inline script that applies the dark / light choice before the first paint;
- ``__NNSLR_SHARED_CSS__``, ``__NNSLR_SITE_CSS__``: the two style sheets;
- ``__NNSLR_TOPBAR__``: brand, live state, theme switch, Account icon and tabs;
- ``__NNSLR_BAR_END__``: only the theme switch and the Account icon (the review page has its own bar);
- ``__NNSLR_SCRIPT__``: the page script; ``__NNSLR_NONCE__``: the nonce, where a template needs it;
- ``__NNSLR_DATA__``: the frames of the review page.
"""
from __future__ import annotations

import html
import json
from importlib.resources import files
from pathlib import Path

from nnslr_tools.review_hints import empty_stats

# The tabs of the site: (key of the current page, address, label). Account is the icon at the right.
SITE_TABS = (('route', '/', 'Route'), ('panel', '/pannello', 'Pannello'), ('status', '/stato', 'Stato'))
# Dark (the default) or the choice kept in this browser, applied before the first paint
# (web/src/common/theme.ts changes it).
THEME_HEAD = ('<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\');'
              'if(t===\'light\'||t===\'auto\')document.documentElement.dataset.theme=t}}catch(e){{}}</script>')
THEME_SWITCH = (
    '<button class="icon-btn theme-toggle" type="button" id="theme-toggle" aria-pressed="false"'
    ' aria-label="Passa al tema chiaro" title="Passa al tema chiaro">'
    '<svg class="i-sun" viewBox="0 0 20 20" aria-hidden="true">'
    '<circle cx="10" cy="10" r="3.6" fill="none" stroke="currentColor" stroke-width="1.8"/>'
    '<path d="M10 1.8v2.2M10 16v2.2M1.8 10H4M16 10h2.2M4.2 4.2l1.6 1.6M14.2 14.2l1.6 1.6M4.2 15.8l1.6-1.6'
    'M14.2 5.8l1.6-1.6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>'
    '<svg class="i-moon" viewBox="0 0 20 20" aria-hidden="true">'
    '<path d="M16.5 12.6A7 7 0 0 1 7.4 3.5a7 7 0 1 0 9.1 9.1z" fill="none" stroke="currentColor"'
    ' stroke-width="1.8" stroke-linejoin="round"/></svg></button>')
ACCOUNT_ICON = (
    '<a class="icon-btn account-btn" href="/account"{current} aria-label="Account" title="Account">'
    '<svg viewBox="0 0 20 20" aria-hidden="true">'
    '<circle cx="10" cy="7" r="3.4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
    '<path d="M3.6 17.2c.9-3.1 3.4-4.8 6.4-4.8s5.5 1.7 6.4 4.8" fill="none" stroke="currentColor"'
    ' stroke-width="1.8" stroke-linecap="round"/></svg></a>')
LIVE = ('<span class="live" role="status"><span class="dot" id="live-dot"></span>'
        '<span id="live-text">connessione…</span></span>')
# Groups of the route list, in this order: the work in progress first, finished routes last.
ROUTE_GROUPS = (('doing', 'In corso'), ('todo', 'Da iniziare'), ('done', 'Completate'))


def asset(name: str) -> str:
    """A file of nnslr_tools/web, read now (the server picks up a change at the next request)."""
    return files('nnslr_tools').joinpath('web').joinpath(name).read_text(encoding='utf-8')


def page_script(page: str, nonce: str) -> str:
    """The built script of a page (web/src/pages/<page>.ts), inline with the nonce."""
    script = asset('js/' + page + '.js')
    # Inline, a "</script" in a string would end the element early.
    if '</script' in script.lower():
        raise ValueError('web/js/' + page + '.js contains "</script": escape it in the TypeScript source')
    return '<script nonce="' + nonce + '">\n' + script + '</script>'


def bar_end(current: str = '') -> str:
    """The theme switch and the Account icon, at the right of every top bar."""
    return THEME_SWITCH + ACCOUNT_ICON.format(current=' aria-current="page"' if current == 'account' else '')


def top_bar(current: str = '', *, live: bool = False, signed_in: bool = True) -> str:
    """Brand, live state, theme switch, Account icon and tabs. current: the key of the tab to mark, or
    'account'. The sign-in page (signed_in=False) shows only the brand and the switch."""
    end = bar_end(current) if signed_in else THEME_SWITCH
    bar = ('<header class="topbar"><div class="topbar-inner">'
           '<a class="brand" href="/" title="Elenco delle route"><span class="brand-mark" aria-hidden="true">50</span>'
           '<span class="brand-name">Speed limits</span></a>' + (LIVE if live else '') + end + '</div>')
    if signed_in:
        tabs = ''.join('<a href="' + href + '"' + (' aria-current="page"' if key == current else '') + '>'
                       + label + '</a>' for key, href, label in SITE_TABS)
        bar += '<nav class="tabs" aria-label="Sezioni">' + tabs + '</nav>'
    return bar + '</header>'


def fill(template: str, nonce: str, page: str, current: str = '', *, live: bool = False,
         signed_in: bool = True, data: str = '') -> str:
    """A template with its placeholders replaced (see the module docstring)."""
    values = {
        '__NNSLR_THEME__': THEME_HEAD.format(nonce=nonce),
        '__NNSLR_SHARED_CSS__': asset('shared.css'),
        '__NNSLR_SITE_CSS__': asset('site.css'),
        '__NNSLR_TOPBAR__': top_bar(current, live=live, signed_in=signed_in),
        '__NNSLR_BAR_END__': bar_end(current),
        '__NNSLR_SCRIPT__': page_script(page, nonce),
        '__NNSLR_NONCE__': nonce,
        # Last: the frames are data and must never be scanned for placeholders.
        '__NNSLR_DATA__': data,
    }
    for name, value in values.items():
        if name in template:
            template = template.replace(name, value)
    return template


def render_review(rows: list[dict], run_id: str, *, route_id: str = '', server: bool = False,
                  nonce: str = '', hints: dict | None = None, hint_stats: dict | None = None,
                  suspicious: dict | None = None, routes: list[dict] | None = None) -> str:
    """The review page with its data embedded. Hints and suspicious flags travel apart from the
    frame rows, so the JSONL export (which copies the rows) never contains them."""
    data = {'run_id': run_id, 'route_id': route_id, 'server': server, 'frames': rows,
            'hints': hints if hints is not None else {},
            'hint_stats': hint_stats if hint_stats is not None else empty_stats(),
            'suspicious': suspicious if suspicious is not None else {},
            'routes': routes if routes is not None else []}
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return fill(asset('review.html'), nonce, 'review', 'route', data=payload)


# [control-panel] - START
def render_panel(nonce: str) -> str:
    """The control panel; it loads its data from /api/panel/status."""
    return fill(asset('panel.html'), nonce, 'panel', 'panel', live=True)


def render_status(nonce: str) -> str:
    """The Stato page (the machine and the jobs, as distill-web's Status); data from /api/panel/status."""
    return fill(asset('status.html'), nonce, 'status', 'status', live=True)
# [control-panel] - END


# [site-login] - START
def render_login(nonce: str) -> str:
    """The sign-in page (also the first visit, which creates the administrator)."""
    return fill(asset('login.html'), nonce, 'login', signed_in=False)


def render_account(nonce: str, signed_in_browser: bool) -> str:
    """The account page: who is signed in, sign out, change the password, the appearance."""
    if signed_in_browser:
        return fill(asset('account.html'), nonce, 'account', 'account')
    return simple_page('Account', 'account', nonce,
                       '<main class="page"><div class="card state-box center"><h1>Account</h1><p>Questo sito è '
                       'avviato solo per questo computer (127.0.0.1): non chiede di accedere. Sulla rete di casa '
                       '(servizio <code>speedlimits-web</code>) si entra con nome e password.</p></div></main>')
# [site-login] - END


def simple_page(title: str, current: str, nonce: str, main: str) -> str:
    """A page without a template of its own: the frame, the content, the theme switch script."""
    return '\n'.join([
        '<!doctype html><html lang="it"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        '<meta name="color-scheme" content="light dark">',
        THEME_HEAD.format(nonce=nonce),
        '<title>Speed limits · ' + html.escape(title) + '</title>',
        '<style>' + asset('shared.css') + asset('site.css') + '</style></head><body>',
        top_bar(current),
        main,
        page_script('site', nonce),
        '</body></html>',
    ])


def _plural(count: int, one: str, many: str) -> str:
    if count == 1:
        return '1 ' + one
    return str(count) + ' ' + many


def route_state(confirmed: int, total: int) -> str:
    """doing, todo or done: the group of a route in the list."""
    if total and confirmed >= total:
        return 'done'
    if confirmed == 0:
        return 'todo'
    return 'doing'


def _route_row(summary: dict) -> str:
    total = summary['frame_count']
    confirmed = summary['confirmed_count']
    percent = round(confirmed * 100 / total) if total else 0
    state = route_state(confirmed, total)
    # What opening the route means: carry on (the one obvious action), start, or look again at a finished one.
    action_class, action = {'doing': ('btn small primary open', 'Riprendi →'),
                            'todo': ('btn small open', 'Inizia →'),
                            'done': ('btn small open again', 'Ricontrolla')}[state]
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
    """The route list, grouped: in progress, to start, done."""
    main = ['<main class="page">',
            '<header class="page-head"><h1 class="page-title">Route</h1>'
            '<p class="page-sub">Scegli una route da revisionare. Ogni conferma viene salvata subito '
            'sul server.</p></header>']
    if not summaries:
        main.append('<div class="card state-box"><h2>Nessuna route da revisionare</h2><p>Prepara una route con '
                    '<code>nnslr extract-frames</code> e <code>nnslr preannotate</code>, poi ricarica questa '
                    'pagina.</p></div>')
    grouped: dict[str, list[dict]] = {key: [] for key, _ in ROUTE_GROUPS}
    for summary in summaries:
        grouped[route_state(summary['confirmed_count'], summary['frame_count'])].append(summary)
    for key, title in ROUTE_GROUPS:
        if not grouped[key]:
            continue
        main.append('<section class="group" aria-labelledby="group-' + key + '">'
                    '<h2 class="group-title" id="group-' + key + '">' + title
                    + ' <span class="n">' + str(len(grouped[key])) + '</span></h2><ul class="cards">')
        main.extend(_route_row(summary) for summary in grouped[key])
        main.append('</ul></section>')
    main.append('<details class="foot"><summary>Dettagli tecnici</summary>'
                '<p>Database delle revisioni: <code>' + html.escape(str(database)) + '</code></p></details>')
    main.append('</main>')
    return simple_page('Route', 'route', nonce, '\n'.join(main))


def render_problem(status: int, message: str, nonce: str = '') -> str:
    """An error page (404, 403, ...)."""
    title = {404: 'Pagina non trovata', 403: 'Accesso negato'}.get(status, 'Errore')
    main = ['<main class="page"><div class="card state-box center">',
            '<p class="code">Codice HTTP ' + str(status) + '</p>',
            '<h1>' + html.escape(title) + '</h1>']
    # "Pagina non trovata." under the title "Pagina non trovata" would only repeat it.
    if message.rstrip('.') != title:
        main.append('<p>' + html.escape(message) + '</p>')
    # Every 403 page (token missing or wrong, host not allowed) would come back here from the list.
    if status != 403:
        main.append('<a class="btn back" href="/">← Elenco delle route</a>')
    main.append('</div></main>')
    return simple_page(title, '', nonce, '\n'.join(main))
