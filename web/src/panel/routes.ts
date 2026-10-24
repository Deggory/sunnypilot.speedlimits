// The table of the routes: segments, frames, confirmed frames, objects in the dataset and state.
import {byId, node} from '../common/dom';
import {thousands} from '../common/format';
import type {RouteRow, RouteState} from '../common/panel-types';

// State of a route: label, pill classes, and a prefix dropped on phones.
const STATE: Record<RouteState, [label: string, kind: string, prefix?: string]> = {
  collegare: ['da collegare', 'todo'], estrarre: ['da estrarre', 'todo'], preannotare: ['da pre-annotare', 'todo'],
  revisionare: ['da revisionare', 'muted hollow'], importare: ['da importare', 'ok hollow', 'pronta '],
  pronta: ['nel dataset', 'ok'],
};

// A pill; the optional prefix is dropped on phones ("pronta da importare" -> "da importare").
function statePill(label: string, kind: string, prefix?: string): HTMLSpanElement {
  const element = node('span', 'pill ' + kind);
  if (prefix) element.appendChild(node('span', 'pill-long', prefix));
  element.append(label);
  return element;
}

// Route id as a link to its review; the leading zeros of "0000011e--…" go in their own span,
// hidden on phones so the part that tells routes apart stays visible.
function routeLink(route: string): HTMLAnchorElement {
  const link = node('a');
  link.href = '/route/' + encodeURIComponent(route) + '/';
  link.title = route;
  const pad = /^0+(?=[0-9a-f]+--)/.exec(route)?.[0];
  if (pad) link.append(node('span', 'rid-pad', pad), route.slice(pad.length));
  else link.textContent = route;
  return link;
}

/** Confirmed frames of the route's report: a bar and the count, or a dash before the report exists. */
function reviewedCell(row: RouteRow): HTMLTableCellElement {
  if (!row.run_id) return node('td', 'none', '–');
  const bar = node('div', 'bar ok');
  const fill = node('span');
  fill.style.width = (row.report_frames ? Math.min(100, row.confirmed * 100 / row.report_frames) : 0) + '%';
  bar.appendChild(fill);
  const box = node('div', 'reviewed');
  box.append(bar, node('span', 'num', thousands(row.confirmed) + ' / ' + thousands(row.report_frames)));
  const cell = node('td');
  cell.appendChild(box);
  return cell;
}

function stateCell(row: RouteRow): HTMLTableCellElement {
  const [label, kind, prefix] = STATE[row.state];
  const cell = node('td');
  cell.appendChild(statePill(label, kind, prefix));
  if (row.state === 'importare') cell.appendChild(node('span', 'note', thousands(row.new_reviews) + ' fotogrammi nuovi'));
  if (row.run_id && row.segments > row.extracted) {
    const late = row.segments - row.extracted;
    cell.appendChild(node('span', 'note', late + (late === 1 ? ' segmento arrivato' : ' segmenti arrivati') + ' dopo, non usati'));
  }
  return cell;
}

function routeRow(row: RouteRow): HTMLTableRowElement {
  const id = node('td', 'route');
  id.appendChild(routeLink(row.route));
  const tr = node('tr');
  tr.append(id,
    node('td', 'num', row.extracted + ' / ' + row.segments),
    node('td', 'num', thousands(row.frames)),
    reviewedCell(row),
    node('td', 'num', thousands(row.objects)),
    stateCell(row));
  return tr;
}

export function renderRoutes(routes: RouteRow[]): void {
  byId('routes-empty').hidden = routes.length > 0;
  byId('routes-wrap').hidden = routes.length === 0;
  byId('routes').replaceChildren(...routes.map(routeRow));
}
