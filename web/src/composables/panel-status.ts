// GET /api/panel/status 4 s after each answer, for the control pages (Overview, Prepare, Train), and the connection
// state of the top bar (composables/connection.ts).
import {shallowRef} from 'vue';
import {api} from '../common/api';
import type {PanelStatus} from '../common/panel-types';
import {showConnection} from './connection';
import {usePoll} from './poll';

const INTERVAL_MS = 4000;

/** The latest status (null before the first answer), refresh() to ask now, and onUpdate for side effects. */
export function usePanelStatus(onUpdate?: (status: PanelStatus) => void) {
  const status = shallowRef<PanelStatus | null>(null);
  async function ask(): Promise<void> {
    try {
      const answer = await api<PanelStatus>('/api/panel/status');
      status.value = answer;
      onUpdate?.(answer);
      showConnection(null);
    } catch (error) {
      showConnection(error);
    }
  }
  return {status, refresh: usePoll(ask, INTERVAL_MS)};
}
