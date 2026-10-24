// The Account menu at the right of every top bar (#account-btn, #account-pop, markup in pages.py): who is
// signed in, the appearance (dark, light, automatic), the account settings and sign out.
import {api, post} from './api';
import type {AuthState} from './auth-types';
import {node} from './dom';
import {onThemeChange, setTheme, storedTheme, type Theme} from './theme';

const THEMES: [Theme, string][] = [['dark', 'Scuro'], ['light', 'Chiaro'], ['auto', 'Automatico']];

function themeChoice(): HTMLDivElement {
  const group = node('div', 'segmented');
  group.setAttribute('role', 'radiogroup');
  group.setAttribute('aria-label', 'Aspetto');
  const buttons = THEMES.map(([theme, label]) => {
    const button = node('button', '', label);
    button.type = 'button';
    button.setAttribute('role', 'radio');
    button.addEventListener('click', () => setTheme(theme));
    return {theme, button};
  });
  const show = (): void => {
    const current = storedTheme();
    for (const {theme, button} of buttons) button.setAttribute('aria-checked', String(theme === current));
  };
  onThemeChange(show);
  show();
  group.append(...buttons.map(({button}) => button));
  return group;
}

/** Who is signed in, as the head of the menu; filled when the menu opens. */
function who(state: AuthState | null, error: string): HTMLDivElement {
  const head = node('div', 'pop-head');
  const avatar = node('span', 'pop-avatar', (state?.user ?? '?').slice(0, 1).toUpperCase());
  avatar.setAttribute('aria-hidden', 'true');
  const text = node('div', 'pop-who');
  if (state === null) {
    text.append(node('strong', '', 'Account'), node('span', '', error || 'Caricamento…'));
  } else if (state.local) {
    text.append(node('strong', '', 'Questo computer'), node('span', '', 'Sito su 127.0.0.1, senza accesso'));
  } else {
    const sessions = state.sessions ?? 1;
    text.append(node('strong', '', state.user ?? ''),
      node('span', '', sessions > 1 ? 'Connesso da ' + sessions + ' browser' : 'Connesso da questo browser'));
  }
  head.append(avatar, text);
  return head;
}

async function signOut(everywhere: boolean): Promise<void> {
  if (everywhere && !window.confirm('Uscire da tutti i browser, compreso questo?')) return;
  try {
    await post(everywhere ? '/api/auth/logout-all' : '/api/auth/logout');
  } catch {
    // Already signed out.
  }
  location.replace('/');
}

function menuButton(label: string, className: string, action: () => void): HTMLButtonElement {
  const button = node('button', 'pop-item ' + className, label);
  button.type = 'button';
  button.addEventListener('click', action);
  return button;
}

function fill(pop: HTMLElement, state: AuthState | null, error = ''): void {
  const appearance = node('div', 'pop-section');
  appearance.append(node('div', 'pop-label', 'Aspetto'), themeChoice());
  const items = node('div', 'pop-section pop-items');
  const settings = node('a', 'pop-item', 'Password e browser connessi');
  settings.href = '/account';
  items.appendChild(settings);
  if (state !== null && !state.local) {
    items.append(menuButton('Esci', '', () => void signOut(false)),
      menuButton('Esci da tutti i browser', 'danger', () => void signOut(true)));
  }
  pop.replaceChildren(who(state, error), appearance, items);
}

/** Opens and closes the menu: the icon, a click outside, Esc (the focus goes back to the icon). */
export function initAccountMenu(): void {
  const button = document.getElementById('account-btn');
  const pop = document.getElementById('account-pop');
  if (button === null || pop === null) return;
  let state: AuthState | null = null;

  const open = (): void => {
    fill(pop, state);
    pop.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    api<AuthState>('/api/auth/state').then((answer) => {
      state = answer;
      if (!pop.hidden) fill(pop, state);
    }).catch((problem: unknown) => {
      if (!pop.hidden && state === null) fill(pop, null, problem instanceof Error ? problem.message : String(problem));
    });
    pop.querySelector<HTMLElement>('[aria-checked="true"]')?.focus();
  };
  const close = (refocus: boolean): void => {
    if (pop.hidden) return;
    pop.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    if (refocus) button.focus();
  };

  button.addEventListener('click', () => (pop.hidden ? open() : close(false)));
  document.addEventListener('click', (event) => {
    const target = event.target as Node;
    if (!pop.contains(target) && !button.contains(target)) close(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !pop.hidden) {
      event.stopPropagation();
      close(true);
    }
  }, true);
}
