// Small DOM helpers shared by every page.

/** The element with this id; a missing one is a bug of the page, so it throws. */
export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error('Elemento mancante nella pagina: #' + id);
  return element as T;
}

/** A new element with an optional class and text. */
export function node<K extends keyof HTMLElementTagNameMap>(
  tag: K, className?: string, text?: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

/** A status pill (.pill in shared.css): label and kind (ok, bad, run, warn, todo, muted...). */
export function pill(label: string, kind: string): HTMLSpanElement {
  return node('span', 'pill ' + kind, label);
}

/** Fills a bar (<div class="bar"><span id=...>) to a percent, with an optional colour kind. */
export function setBar(id: string, percent: number, kind = ''): void {
  const bar = byId(id);
  bar.style.width = Math.max(0, Math.min(100, percent)) + '%';
  if (bar.parentElement) bar.parentElement.className = 'bar' + (kind ? ' ' + kind : '');
}

/**
 * True when value is the same as at the previous call with this key. Lists and buttons are rebuilt only
 * when their data changes: rebuilding them at every refresh would lose a click and the keyboard focus.
 */
const shown = new Map<string, string>();
export function unchanged(key: string, value: unknown): boolean {
  const text = JSON.stringify(value);
  if (shown.get(key) === text) return true;
  shown.set(key, text);
  return false;
}
