// Dark (the default, as distill-web) or light. The choice is kept in this browser; the inline script in the
// <head> of every page (pages.py, THEME_HEAD) applies it before the first paint, this module changes it.

export type Theme = 'dark' | 'light' | 'auto';
const KEY = 'nnslr-theme';
const CHANGED = 'nnslr-theme';
const root = document.documentElement;

export function storedTheme(): Theme {
  try {
    const value = localStorage.getItem(KEY);
    if (value === 'light' || value === 'auto') return value;
  } catch {
    // Storage off (private mode): the default.
  }
  return 'dark';
}

/** True when the page shows the dark palette now (Automatico follows the system). */
export function isDark(): boolean {
  const theme = root.dataset['theme'];
  if (theme === 'light') return false;
  if (theme === 'auto') return !matchMedia('(prefers-color-scheme: light)').matches;
  return true;
}

export function setTheme(theme: Theme): void {
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // Storage off: the choice lasts until the page is left.
  }
  if (theme === 'dark') delete root.dataset['theme'];
  else root.dataset['theme'] = theme;
  document.dispatchEvent(new Event(CHANGED));
}

export function onThemeChange(listener: () => void): void {
  document.addEventListener(CHANGED, listener);
}

/** The dark / light switch at the right of the top bar (#theme-toggle), when the page has it. */
export function initThemeSwitch(): void {
  const button = document.getElementById('theme-toggle');
  if (button === null) return;
  const show = (): void => {
    const dark = isDark();
    const title = dark ? 'Passa al tema chiaro' : 'Passa al tema scuro';
    button.classList.toggle('is-light', !dark);
    button.setAttribute('aria-pressed', String(!dark));
    button.title = title;
    button.setAttribute('aria-label', title);
  };
  button.addEventListener('click', () => setTheme(isDark() ? 'light' : 'dark'));
  onThemeChange(show);
  show();
}
