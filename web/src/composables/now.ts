// The time now, ticking every second while some component uses it (one timer for the whole page).
import {onBeforeUnmount, onMounted, readonly, ref, type Ref} from 'vue';

const now = ref(Date.now());
let users = 0;
let timer: number | undefined;

export function useNow(): Readonly<Ref<number>> {
  onMounted(() => {
    if (users++ === 0) {
      now.value = Date.now();
      timer = window.setInterval(() => { now.value = Date.now(); }, 1000);
    }
  });
  onBeforeUnmount(() => {
    if (--users === 0) window.clearInterval(timer);
  });
  return readonly(now);
}
