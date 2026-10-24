// "Prossimo passo": what the card on top says, its colour and the step card that needs attention.
import type {PanelStatus} from '../common/panel-types';

export type NextAction = NonNullable<PanelStatus['next_action']>;
/** The colour of a step (k-<kind> in the page CSS). */
export type StepKind = 'prepare' | 'review' | 'train' | 'done';
/** The three step cards. */
export type StepName = 'prepare' | 'review' | 'train';

export interface NextStep {
  /** null before the first answer: no colour. */
  kind: StepKind | null;
  title: string;
  text: string;
  action: NextAction | null;
  /** The step card to mark; fetching has none (it is a button of step 1, not its main work). */
  attention: StepName | null;
}

const HERO: Record<NextAction, [title: string, text: string]> = {
  prepare: ['Prepara le route nuove', 'Ci sono video nuovi da trasformare in fotogrammi con i cartelli proposti.'],
  review: ['Revisiona i cartelli', 'Apri le route da revisionare e conferma o correggi i cartelli.'],
  train: ['Allena il modello', 'Le revisioni sono pronte: importale e allena detector e reader sulla GPU.'],
  fetch: ['Scarica nuovi giri', 'Per allenare servono più route: scarica altri giri dal comma.'],
};

/** Before the first answer: the placeholder of the page. */
export const WAITING: NextStep = {kind: null, title: '…', text: '', action: null, attention: null};

export function nextStep(status: PanelStatus): NextStep {
  const action = status.next_action;
  if (!action) return {kind: 'done', title: 'Tutto in ordine', text: status.next_step, action, attention: null};
  const attention = action === 'fetch' ? null : action;
  if (action === 'train' && status.next_step.startsWith('tutto pronto')) {
    return {kind: 'done', title: 'Tutto pronto', text: 'Il dataset è aggiornato: puoi allenare un nuovo modello quando vuoi.',
      action, attention};
  }
  const [title, text] = HERO[action];
  return {kind: action === 'fetch' ? 'prepare' : action, title, text, action, attention};
}
