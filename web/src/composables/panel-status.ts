// GET /api/panel/status every 4 s, for the control pages (Overview, Prepare, Train), and the connection state of the
// top bar (composables/connection.ts).
import {onBeforeUnmount, onMounted, shallowRef} from 'vue';
import {api} from '../common/api';
import type {PanelStatus} from '../common/panel-types';
import {showConnection} from './connection';

const INTERVAL_MS = 4000;

/** The latest status (null before the first answer), refresh() to ask now, and onUpdate for side effects. */
export function usePanelStatus(onUpdate?: (status: PanelStatus) => void) {
  const status = shallowRef<PanelStatus | null>(null);
  let timer: number | undefined;

  async function refresh(): Promise<void> {
    try {
      const answer = await api<PanelStatus>('/api/panel/status');
      status.value = answer;
      onUpdate?.(answer);
      showConnection(null);
    } catch (error) {
      showConnection(error);
    }
  }
  onMounted(() => {
    void refresh();
    timer = window.setInterval(() => void refresh(), INTERVAL_MS);
  });
  onBeforeUnmount(() => window.clearInterval(timer));
  return {status, refresh};
}
