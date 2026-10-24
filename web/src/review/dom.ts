// DOM helpers of the review page that common/dom.ts has not: SVG elements, selectors, media queries, the
// small per-browser memory (localStorage).

const SVG_NS = 'http://www.w3.org/2000/svg';

/** The SVG element with this id (byId of common/dom.ts is for HTML elements); a missing one is a bug. */
export function svgById<T extends SVGElement = SVGElement>(id: string): T {
  // getElementById is typed for HTML elements only.
  const element: Element | null = document.getElementById(id);
  if (!(element instanceof SVGElement)) throw new Error('Elemento SVG mancante nella pagina: #' + id);
  return element as T;
}

/** The first element matching a selector; a missing one is a bug of the page. */
export function one<T extends Element = HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (element === null) throw new Error('Elemento mancante nella pagina: ' + selector);
  return element;
}

/** A new SVG element with its attributes. */
export function svgNode<K extends keyof SVGElementTagNameMap>(
  tag: K, attributes: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, String(value));
  return element;
}

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
