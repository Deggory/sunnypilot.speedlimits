// "Next step": what the card on top of the Overview says, its colour and the phase that needs attention.
import type {PanelStatus} from '../common/panel-types';

export type NextAction = NonNullable<PanelStatus['next_action']>;
/** The colour of a step (k-<kind> in the page CSS). */
export type StepKind = 'prepare' | 'review' | 'train' | 'done';
/** The three phases. */
export type StepName = 'prepare' | 'review' | 'train';

export interface NextStep {
  /** null before the first answer: no colour. */
  kind: StepKind | null;
  title: string;
  text: string;
  action: NextAction | null;
  /** The phase to mark; fetching has none (it is a button of the first phase, not its main work). */
  attention: StepName | null;
}

const TITLE: Record<NextAction, string> = {
  prepare: 'Prepare the new routes',
  review: 'Review the signs',
  train: 'Train the model',
  fetch: 'Download new drives',
};

/** Before the first answer: the placeholder of the page. */
export const WAITING: NextStep = {kind: null, title: '…', text: '', action: null, attention: null};

/** The card from the server's next_action, its English sentence (next_step_text) and all_ready. */
export function nextStep(status: PanelStatus): NextStep {
  const action = status.next_action;
  const text = status.next_step_text ?? '';
  if (!action) return {kind: 'done', title: 'All in order', text, action, attention: null};
  const attention = action === 'fetch' ? null : action;
  if (action === 'train' && status.all_ready) return {kind: 'done', title: 'All ready', text, action, attention};
  return {kind: action === 'fetch' ? 'prepare' : action, title: TITLE[action], text, action, attention};
}
