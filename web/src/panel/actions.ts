// The action buttons of the Pannello (start a job in the background) and the stop of a running job.
import {messageOf, post} from '../common/api';
import {byId, node} from '../common/dom';
import type {ActionName, ActionState, Job, StartedJob} from '../common/panel-types';
import {toast} from '../common/toast';

/** What a button needs from the page around it. */
export interface ActionContext {
  /** Reads /api/panel/status again and shows it. */
  refresh(): Promise<void>;
  /** A job was started from this page: its card stays in its step after it ends. */
  started(job: StartedJob): void;
}

export type Actions = Record<ActionName, ActionState>;

// Only the long or remote actions ask first; preparing and the dataset start at the click.
function confirmed(name: ActionName, label: string): boolean {
  if (name === 'train' || name === 'train_continue') {
    return window.confirm(label + ': il training usa la GPU per ore. Avvio?');
  }
  if (name === 'fetch') return window.confirm(label + ': scarica dal comma i giri nuovi. Avvio?');
  return true;
}

async function run(name: ActionName, state: ActionState, context: ActionContext): Promise<void> {
  if (!confirmed(name, state.label)) return;
  try {
    const job = await post<StartedJob>('/api/panel/run/' + name);
    context.started(job);
    toast('Partito: ' + job.label);
    await context.refresh();
  } catch (error) {
    toast(messageOf(error), true);
    void context.refresh();
  }
}

export async function stopJob(job: Job, context: ActionContext): Promise<void> {
  if (!window.confirm('Fermo "' + job.label + '"? Il lavoro già fatto resta, si riprende rilanciando.')) return;
  try {
    await post('/api/panel/job/' + job.id + '/stop');
    toast('Fermata: ' + job.label);
  } catch (error) {
    toast(messageOf(error), true);
  }
  void context.refresh();
}

export function actionButton(name: ActionName, state: ActionState, primary: boolean,
  context: ActionContext): HTMLButtonElement {
  const button = node('button', 'btn' + (primary ? ' primary' : ''), state.label);
  button.type = 'button';
  button.disabled = !state.allowed;
  button.addEventListener('click', () => void run(name, state, context));
  return button;
}

/**
 * Buttons of one card, and why the disabled ones are disabled, in the single reason line under them.
 * One reason shared by every disabled button is written once, otherwise "label: reason" each.
 */
export function fillActions(buttonsId: string, reasonId: string, names: ActionName[], actions: Actions,
  context: ActionContext): void {
  const blocked = names.filter((name) => !actions[name].allowed && actions[name].reason);
  const reasons = [...new Set(blocked.map((name) => actions[name].reason))];
  const only = reasons.length === 1 ? reasons[0] : undefined;
  const text = only !== undefined
    ? only.charAt(0).toUpperCase() + only.slice(1) + '.'
    : blocked.map((name) => actions[name].label + ': ' + actions[name].reason).join(' · ');
  const reason = byId(reasonId);
  reason.textContent = text;
  reason.title = text;
  byId(buttonsId).replaceChildren(...names.map((name) => {
    const button = actionButton(name, actions[name], false, context);
    if (blocked.includes(name)) button.setAttribute('aria-describedby', reasonId);
    return button;
  }));
}
