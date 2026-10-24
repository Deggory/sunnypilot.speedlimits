// The three step cards: Prepara, Revisiona, Allena (figure, meter, text and buttons).
import {byId, node, setBar} from '../common/dom';
import {thousands} from '../common/format';
import type {PanelStatus, RouteRow, RouteState} from '../common/panel-types';
import {fillActions, type ActionContext} from './actions';

const TO_PREPARE: RouteState[] = ['collegare', 'estrarre', 'preannotare'];
const REVIEW_SUB = 'Confermi o correggi i cartelli: l\'unico passo a mano.';
const TRAIN_SUB = 'Revisioni → dataset → modello, sulla GPU.';

const sum = (rows: RouteRow[], value: (row: RouteRow) => number): number =>
  rows.reduce((total, row) => total + value(row), 0);

/** The figure of a card: the number, and under it what it counts. */
function big(id: string, figure: string, note: string): void {
  byId(id).replaceChildren(figure, node('span', 'big-note', note));
}

function renderPrepare(status: PanelStatus, context: ActionContext): void {
  const toPrepare = status.routes.filter((row) => TO_PREPARE.includes(row.state)).length;
  if (toPrepare) big('prepare-big', toPrepare + (toPrepare === 1 ? ' route nuova' : ' route nuove'), 'da trasformare in fotogrammi');
  else big('prepare-big', 'Tutto preparato', 'nessun video in attesa');
  fillActions('prepare-buttons', 'prepare-reason', ['fetch', 'prepare'], status.actions, context);
}

function renderReview(status: PanelStatus): void {
  const reported = status.routes.filter((row) => row.run_id);
  const frames = sum(reported, (row) => row.report_frames);
  const confirmed = sum(reported, (row) => Math.min(row.confirmed, row.report_frames));
  const toReview = status.routes.filter((row) => row.state === 'revisionare').length;
  if (frames) big('review-big', thousands(confirmed) + ' di ' + thousands(frames), 'fotogrammi confermati');
  else byId('review-big').textContent = 'Niente da revisionare';
  setBar('review-bar', frames ? confirmed * 100 / frames : 0, 'ok');
  byId('review-sub').textContent = !frames ? REVIEW_SUB
    : toReview ? toReview + ' route ancora da revisionare.' : 'Tutte le route sono revisionate.';
}

function renderTrain(status: PanelStatus, context: ActionContext): void {
  // reviewed_routes: routes with at least one confirmed frame; min_routes: what the split needs.
  const reviewed = status.reviewed_routes;
  const needed = status.min_routes;
  byId('train-goal').replaceChildren(
    ...Array.from({length: needed}, (_, index) => node('span', index < reviewed ? 'on' : '')));
  if (reviewed >= needed) big('train-big', reviewed + ' route revisionate', 'ne bastano ' + needed + ' per allenare');
  else big('train-big', reviewed + ' di ' + needed + ' route', 'revisionate, ne servono ' + needed + ' per allenare');
  const fresh = sum(status.routes, (row) => row.new_reviews);
  byId('train-sub').textContent = (fresh ? thousands(fresh) + ' fotogrammi confermati da importare. ' : '')
    + (reviewed < needed ? 'Lo split train/validation/test ne vuole almeno ' + needed + ' indipendenti.' : TRAIN_SUB);
  fillActions('train-buttons', 'train-reason', ['dataset', 'train', 'train_continue'], status.actions, context);
}

export function renderSteps(status: PanelStatus, context: ActionContext): void {
  renderPrepare(status, context);
  renderReview(status);
  renderTrain(status, context);
}
