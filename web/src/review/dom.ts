// Browser helpers of the review page: media queries, the data attributes of an event target, the small
// per-browser memory (localStorage), deep copies of JSON data.

/** The data attributes of an event target, when it is an element that has them. */
export function datasetOf(target: EventTarget | null): DOMStringMap | null {
  if (target instanceof HTMLElement || target instanceof SVGElement) return target.dataset;
  return null;
}

export function narrowLayout(): boolean {
  return window.matchMedia('(max-width: 980px)').matches;
}

export function coarsePointer(): boolean {
  return window.matchMedia('(pointer: coarse)').matches;
}

export function readLocal(key: string, fallback: string): string;
export function readLocal(key: string, fallback: null): string | null;
export function readLocal(key: string, fallback: string | null): string | null {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeLocal(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Per-viewer convenience only.
  }
}

/** A deep copy of JSON data. */
export function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** After a click on a control of the panel the focus goes back to the page, so that Enter confirms the frame
 * (the TypeScript page drew these controls anew at every change, which had the same effect). */
export function releaseFocus(event: Event): void {
  if (event.currentTarget instanceof HTMLElement) event.currentTarget.blur();
}
