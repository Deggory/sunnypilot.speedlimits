// The toast at the bottom right (#toast, .toast in shared.css).
import {byId} from './dom';

let timer: number | undefined;

export function toast(message: string, error = false): void {
  const element = byId('toast');
  element.textContent = message;
  element.className = 'toast show' + (error ? ' error' : '');
  window.clearTimeout(timer);
  timer = window.setTimeout(() => { element.className = 'toast'; }, error ? 7000 : 3500);
}
