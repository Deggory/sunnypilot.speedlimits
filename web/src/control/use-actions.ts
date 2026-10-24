// Starting an action of the control pages (a job in the background) and stopping a running job.
import {messageOf, post} from '../common/api';
import type {ActionName, ActionState, Job, StartedJob} from '../common/panel-types';
import {toast} from '../composables/toast';

// The long, remote or disruptive actions ask first; preparing and the dataset start at the click.
function confirmed(name: ActionName, label: string): boolean {
  if (name === 'train' || name === 'train_continue') {
    return window.confirm(label + ': il training usa la GPU per ore. Avvio?');
  }
  if (name === 'fetch') return window.confirm(label + ': scarica dal comma i giri nuovi. Avvio?');
  if (name === 'update') return window.confirm(label + ': git pull e riavvio del sito (qualche secondo). Avvio?');
  return true;
}

/** run() and stop() for the buttons; refresh reads /api/panel/status again. */
export function usePanelActions(refresh: () => Promise<void>) {
  async function run(name: ActionName, state: ActionState): Promise<void> {
    if (!confirmed(name, state.label)) return;
    try {
      const job = await post<StartedJob>('/api/panel/run/' + name);
      toast('Partito: ' + job.label);
      await refresh();
    } catch (error) {
      toast(messageOf(error), true);
      void refresh();
    }
  }

  async function stop(job: Job): Promise<void> {
    if (!window.confirm('Fermo "' + job.label + '"? Il lavoro già fatto resta, si riprende rilanciando.')) return;
    try {
      await post('/api/panel/job/' + job.id + '/stop');
      toast('Fermato: ' + job.label);
    } catch (error) {
      toast(messageOf(error), true);
    }
    void refresh();
  }

  return {run, stop};
}
