// "Prossimo passo": the card on top, coloured as the step it points to, with its one obvious action.
import {byId, node} from '../common/dom';
import type {PanelStatus} from '../common/panel-types';
import {actionButton, type ActionContext} from './actions';

type NextAction = NonNullable<PanelStatus['next_action']>;
/** The colour of a step (k-<kind> in the page CSS). */
type Kind = 'prepare' | 'review' | 'train' | 'done';

const HERO: Record<NextAction, [title: string, text: string]> = {
  prepare: ['Prepara le route nuove', 'Ci sono video nuovi da trasformare in fotogrammi con i cartelli proposti.'],
  review: ['Revisiona i cartelli', 'Apri le route da revisionare e conferma o correggi i cartelli.'],
  train: ['Allena il modello', 'Le revisioni sono pronte: importale e allena detector e reader sulla GPU.'],
  fetch: ['Scarica nuovi giri', 'Per allenare servono più route: scarica altri giri dal comma.'],
};
// The step card that needs attention; fetching has none (it is a button of step 1, not its main work).
const ATTENTION: Partial<Record<NextAction, string>> = {
  prepare: 'step-prepare', review: 'step-review', train: 'step-train',
};
const STEP_CARDS = ['step-prepare', 'step-review', 'step-train'];

export function renderHero(status: PanelStatus, context: ActionContext): void {
  const action = status.next_action;
  let kind: Kind = 'done';
  let [title, text] = ['Tutto in ordine', status.next_step];
  if (action) {
    [title, text] = HERO[action];
    kind = action === 'fetch' ? 'prepare' : action;
  }
  if (action === 'train' && status.next_step.startsWith('tutto pronto')) {
    title = 'Tutto pronto';
    text = 'Il dataset è aggiornato: puoi allenare un nuovo modello quando vuoi.';
    kind = 'done';
  }
  byId('hero').className = 'card hero k-' + kind;
  byId('hero-title').textContent = title;
  byId('hero-text').textContent = text;

  const actions = byId('hero-actions');
  actions.replaceChildren();
  if (action === 'review') {
    const link = node('a', 'btn primary', 'Apri la revisione →');
    link.href = '/';
    actions.appendChild(link);
  } else if (action) {
    const state = status.actions[action];
    const button = actionButton(action, state, true, context);
    actions.appendChild(button);
    if (!state.allowed && state.reason) {
      const reason = node('p', 'reason', state.reason);
      reason.id = 'hero-reason';
      button.setAttribute('aria-describedby', reason.id);
      actions.appendChild(reason);
    }
  }

  const attention = action ? ATTENTION[action] : undefined;
  for (const id of STEP_CARDS) byId(id).classList.toggle('attention', id === attention);
}
