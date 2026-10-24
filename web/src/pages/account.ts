// Account: who is signed in, sign out, change the password, the appearance (dark, light, automatic).
import {api, messageOf, post} from '../common/api';
import type {AuthAnswer, AuthState} from '../common/auth-types';
import {byId} from '../common/dom';
import {initThemeSwitch, onThemeChange, setTheme, storedTheme, type Theme} from '../common/theme';

const choice = byId('theme-choice');

function showTheme(): void {
  const theme = storedTheme();
  for (const button of choice.querySelectorAll<HTMLElement>('[data-theme-choice]')) {
    button.setAttribute('aria-checked', String(button.dataset['themeChoice'] === theme));
  }
}

choice.addEventListener('click', (event) => {
  const button = (event.target as Element).closest<HTMLElement>('[data-theme-choice]');
  const theme = button?.dataset['themeChoice'];
  if (theme === 'dark' || theme === 'light' || theme === 'auto') setTheme(theme satisfies Theme);
});
onThemeChange(showTheme);
showTheme();

function render(state: AuthState): void {
  let text = 'Connesso come ' + (state.user ?? '') + '.';
  if (state.sessions !== undefined && state.sessions > 1) text += ' Connesso da ' + state.sessions + ' browser.';
  byId('admin-sub').textContent = text;
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

byId('btn-signout').addEventListener('click', () => void signOut(false));
byId('btn-signout-all').addEventListener('click', () => void signOut(true));

byId<HTMLFormElement>('password-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const error = byId('password-error');
  const fields = ['current', 'fresh', 'repeat'].map((id) => byId<HTMLInputElement>(id));
  const [current, fresh, repeat] = fields.map((field) => field.value);
  const button = byId<HTMLButtonElement>('btn-password');
  error.textContent = '';
  byId('password-ok').textContent = '';
  if (!current) { error.textContent = 'Scrivi la password attuale.'; return; }
  if (!fresh || fresh.length < 12) { error.textContent = 'La nuova password deve avere almeno 12 caratteri.'; return; }
  if (fresh !== repeat) { error.textContent = 'Le due nuove password sono diverse.'; return; }
  button.disabled = true;
  try {
    const answer = await post<AuthAnswer>('/api/auth/password', {current, password: fresh});
    for (const field of fields) field.value = '';
    render(answer.state);
    byId('password-ok').textContent = 'Password cambiata: gli altri browser sono usciti.';
  } catch (problem) {
    error.textContent = messageOf(problem);
  } finally {
    button.disabled = false;
  }
});

initThemeSwitch();
api<AuthState>('/api/auth/state').then(render).catch((problem: unknown) => {
  byId('admin-sub').textContent = messageOf(problem);
});
