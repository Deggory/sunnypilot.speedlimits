// GET /api/panel/status every 4 s, for the control pages (Panoramica, Prepara, Allena), and the live dot of the top bar
// (#live-dot, #live-text, rendered by pages.py outside the Vue app).
import {onBeforeUnmount, onMounted, shallowRef} from 'vue';
import {api, messageOf} from '../common/api';
import {clock} from '../common/format';
import type {PanelStatus} from '../common/panel-types';

const INTERVAL_MS = 4000;

function showLive(text: string, ok: boolean): void {
  document.querySelector('.live')?.classList.toggle('off', !ok);
  const dot = document.getElementById('live-dot');
  const label = document.getElementById('live-text');
  if (dot) {
    dot.className = ok ? 'dot' : 'dot off';
    dot.title = text;
  }
  if (label) label.textContent = text;
}

/** The latest status (null before the first answer), refresh() to ask now, and onUpdate for side effects. */
export function usePanelStatus(onUpdate?: (status: PanelStatus) => void) {
  const status = shallowRef<PanelStatus | null>(null);
  let timer: number | undefined;

  async function refresh(): Promise<void> {
    try {
      const answer = await api<PanelStatus>('/api/panel/status');
      status.value = answer;
      onUpdate?.(answer);
      showLive('aggiornato ' + clock(), true);
    } catch (error) {
      showLive('non raggiungibile: ' + messageOf(error), false);
    }
  }
  onMounted(() => {
    void refresh();
    timer = window.setInterval(() => void refresh(), INTERVAL_MS);
  });
  onBeforeUnmount(() => window.clearInterval(timer));
  return {status, refresh};
}
