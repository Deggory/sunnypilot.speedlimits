"""The HTML pages of the review site.

Templates, styles and scripts live in ``nnslr_tools/web``: ``shared.css`` (tokens and frame, the look of
distill-web), ``site.css`` (route list, error pages, sign-in, account), the page templates and
``js/<page>.js``, built from the Vue and TypeScript of ``web/src`` by ``helpers/build_web.sh``; a page with
a script renders its content in ``<div id="app">``. Every page gets
its script inline with the nonce of its Content-Security-Policy. Files are read on every request: a
change to them is live at the next page load, a change to this module after a restart.

Placeholders of the templates:

- ``__NNSLR_THEME__``: the inline script that applies the dark / light choice before the first paint;
- ``__NNSLR_SHARED_CSS__``, ``__NNSLR_SITE_CSS__``: the two style sheets;
- ``__NNSLR_TOPBAR__``: brand, live state, Account menu and tabs;
- ``__NNSLR_BAR_END__``: only the Account menu (the review page has its own bar);
- ``__NNSLR_SCRIPT__``: the page script; ``__NNSLR_NONCE__``: the nonce, where a template needs it;
- ``__NNSLR_DATA__``: the frames of the review page.
"""
from __future__ import annotations

import html
import json
from importlib.resources import files
from pathlib import Path

from nnslr_tools.gps import country_names
from nnslr_tools.review_hints import empty_stats

# The logo: a Swiss 50 sign in the yellow corners of the box the review selects (the model reading a sign), on
# the dark tile .brand-mark of shared.css; the digits are paths, the same on every system. The same drawing as
# web/src/components/BrandMark.vue, for the Vue pages.
BRAND_MARK = ('<span class="brand-mark" aria-hidden="true"><svg viewBox="0 0 32 32" focusable="false">'
              '<path d="M4.2 9V5.6c0-.8.6-1.4 1.4-1.4H9M23 4.2h3.4c.8 0 1.4.6 1.4 1.4V9M27.8 23v3.4c0 .8-.6 1.4-1.4 '
              '1.4H23M9 27.8H5.6c-.8 0-1.4-.6-1.4-1.4V23" fill="none" stroke="#ffd60a" stroke-width="1.6" '
              'stroke-linecap="round"/><circle cx="16" cy="16" r="9" fill="#fff" stroke="#e3001b" stroke-width="2.7"/>'
              '<g fill="none" stroke="#111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
              'transform="matrix(.88 0 0 .88 1.92 1.92)"><path d="M15.2 12.2H11.4l-.3 3.5c.6-.5 1.3-.7 2-.7 1.4 0 2.3 '
              '1 2.3 2.4 0 1.5-1 2.4-2.4 2.4-1 0-1.8-.5-2.2-1.2"/><rect x="17.2" y="12.2" width="3.9" height="7.6" '
              'rx="1.95"/></g></svg></span>')
# The machine in small (GPU, its memory and temperature, CPU, RAM, disk) at the right of the tab row, on every
# page with the tabs (web/src/components/TopMachine.vue, mounted here by the page script). data-feed: the
# control pages have it from their status, the others ask /api/panel/system.
TABS_MACHINE = '<div class="tabs-machine" id="tabs-machine" data-feed="{feed}"></div>'

# The tabs of the site: (key of the current page, address, label). Account is the icon at the right.
# In the order of the work: prepare the routes, review them, train on the reviews, evaluate the runs.
SITE_TABS = (('overview', '/', 'Overview'), ('prepare', '/prepare', 'Prepare'), ('review', '/routes', 'Review'),
             ('train', '/train', 'Train'), ('evaluate', '/evaluate', 'Evaluate'))
# The tabs that are one Vue app (web/src/control), by address: the app shows the tab of the address. The
# route list (/routes) is reached from the Overview.
CONTROL_TABS = {'/': 'overview', '/prepare': 'prepare', '/train': 'train', '/evaluate': 'evaluate'}
# Earlier addresses of these pages, kept working.
OLD_ADDRESSES = {'/panoramica': '/', '/pannello': '/', '/stato': '/', '/prepara': '/prepare', '/allena': '/train'}
# Dark (the default) or the choice kept in this browser, applied before the first paint
# (web/src/common/theme.ts changes it).
THEME_HEAD = ('<script nonce="{nonce}">try{{var t=localStorage.getItem(\'nnslr-theme\');'
              'if(t===\'light\'||t===\'auto\')document.documentElement.dataset.theme=t}}catch(e){{}}</script>')
# The Account menu at the right of every top bar (web/src/components/AccountMenu.vue, mounted here by the
# page script): who is signed in, the appearance (dark, light, automatic), the settings and sign out. The
# icon is already in the HTML, so nothing moves while the script starts.
ACCOUNT_MENU = (
    '<div class="account-menu" id="account-menu" data-current="{current}">'
    '<button class="icon-btn account-btn" type="button" id="account-btn" aria-haspopup="dialog" aria-expanded="false"'
    ' aria-controls="account-pop" aria-label="Account and settings" title="Account">'
    '<svg viewBox="0 0 20 20" aria-hidden="true">'
    '<circle cx="10" cy="7" r="3.4" fill="none" stroke="currentColor" stroke-width="1.8"/>'
    '<path d="M3.6 17.2c.9-3.1 3.4-4.8 6.4-4.8s5.5 1.7 6.4 4.8" fill="none" stroke="currentColor"'
    ' stroke-width="1.8" stroke-linecap="round"/></svg></button>'
    '<div class="account-pop" id="account-pop" role="dialog" aria-label="Account and settings" hidden></div></div>')
LIVE = ('<span class="live" role="status"><span class="dot" id="live-dot"></span>'
        '<span id="live-text">connecting…</span></span>')
# Groups of the route list, in this order: the work in progress first, finished routes last.
ROUTE_GROUPS = (('doing', 'In progress'), ('todo', 'To start'), ('done', 'Done'))


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
    """The Account menu, at the right of every top bar."""
    return ACCOUNT_MENU.format(current='true' if current == 'account' else 'false')


def top_bar(current: str = '', *, live: bool = False, signed_in: bool = True) -> str:
    """Brand, live state, Account menu and tabs. current: the key of the tab to mark, or 'account'.
    The sign-in page (signed_in=False) shows only the brand."""
    end = bar_end(current) if signed_in else ''
    bar = ('<header class="topbar"><div class="topbar-inner">'
           '<a class="brand" href="/" title="Overview">' + BRAND_MARK
           + '<span class="brand-name">Speed limits</span></a>' + (LIVE if live else '') + end + '</div>')
    if signed_in:
        tabs = ''.join('<a href="' + href + '"' + (' aria-current="page"' if key == current else '') + '>'
                       + label + '</a>' for key, href, label in SITE_TABS)
        bar += ('<nav class="tabs" aria-label="Sections">' + tabs
                + TABS_MACHINE.format(feed='status' if live else 'system') + '</nav>')
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
            'routes': routes if routes is not None else [],
            'countries': country_names()}
    payload = json.dumps(data, ensure_ascii=True, allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return fill(asset('review.html'), nonce, 'review', 'route', data=payload)


# [control-panel] - START
def render_control(nonce: str, tab: str) -> str:
    """Overview, Prepare, Train, Evaluate: one page (control.html, web/src/control), the tab marked in the bar;
    data from /api/panel/status (and /api/panel/runs for Evaluate)."""
    return fill(asset('control.html'), nonce, 'control', tab, live=True)
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
                       '<main class="page"><div class="card state-box center"><h1>Account</h1><p>This site was '
                       'started for this computer only (127.0.0.1): it does not ask you to sign in. On the home '
                       'network (service <code>speedlimits-web</code>) you sign in with a name and a password.</p>'
                       '</div></main>')
# [site-login] - END


def simple_page(title: str, current: str, nonce: str, main: str) -> str:
    """A page without a template of its own: the frame, the content, the script of the Account menu."""
    return '\n'.join([
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
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
    action_class, action = {'doing': ('btn small primary open', 'Resume →'),
                            'todo': ('btn small open', 'Start →'),
                            'done': ('btn small open again', 'Check again')}[state]
    suspicious = summary.get('suspicious_count', 0)
    hint_frames = summary.get('hint_frames', 0)
    stats = ['<span>' + _plural(total, 'frame', 'frames') + '</span>',
             '<span>' + str(summary['frames_with_proposals']) + ' with proposals</span>']
    if suspicious:
        stats.append('<span class="sus">' + _plural(suspicious, 'suspicious', 'suspicious') + '</span>')
    else:
        stats.append('<span>0 suspicious</span>')
    if hint_frames:
        stats.append('<span>' + str(hint_frames) + ' with second reading</span>')
    route = html.escape(summary['route_id'])
    return ('<li><a class="card-row ' + state + '" href="' + html.escape(summary['url']) + '">'
            + '<span class="state-icon" aria-hidden="true">' + ('✓' if state == 'done' else '') + '</span>'
            + '<span class="route" title="' + route + '">' + route + '</span>'
            + '<span class="count">' + str(confirmed) + ' of ' + str(total)
            + ' <span class="unit">confirmed</span></span>'
            + '<span class="bar" aria-hidden="true"><span style="width:' + str(percent) + '%"></span></span>'
            + '<span class="stats">' + ''.join(stats) + '</span>'
            + '<span class="' + action_class + '">' + action + '</span></a></li>')


def render_index(summaries: list[dict], database: Path, nonce: str = '') -> str:
    """The route list, grouped: in progress, to start, done."""
    main = ['<main class="page">',
            '<header class="page-head"><h1 class="page-title">Routes</h1>'
            '<p class="page-sub">Choose a route to review. Every confirmation is saved on the server '
            'at once.</p></header>']
    if not summaries:
        main.append('<div class="card state-box"><h2>No routes to review</h2><p>Prepare a route with '
                    '<code>nnslr extract-frames</code> and <code>nnslr preannotate</code>, then reload this '
                    'page.</p></div>')
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
    main.append('<details class="foot"><summary>Technical details</summary>'
                '<p>Review database: <code>' + html.escape(str(database)) + '</code></p></details>')
    main.append('</main>')
    return simple_page('Routes', 'review', nonce, '\n'.join(main))


def render_problem(status: int, message: str, nonce: str = '') -> str:
    """An error page (404, 403, ...)."""
    title = {404: 'Page not found', 403: 'Access denied'}.get(status, 'Error')
    main = ['<main class="page"><div class="card state-box center">',
            '<p class="code">HTTP code ' + str(status) + '</p>',
            '<h1>' + html.escape(title) + '</h1>']
    # "Page not found." under the title "Page not found" would only repeat it.
    if message.rstrip('.') != title:
        main.append('<p>' + html.escape(message) + '</p>')
    # Every 403 page (token missing or wrong, host not allowed) would come back here from the list.
    if status != 403:
        main.append('<a class="btn back" href="/routes">← All routes</a>')
    main.append('</div></main>')
    return simple_page(title, '', nonce, '\n'.join(main))
