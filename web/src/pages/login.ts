// The sign-in page, also the first visit that creates the administrator (setup code, then name and password).
import {api, messageOf, post} from '../common/api';
import type {AuthState} from '../common/auth-types';
import {byId} from '../common/dom';

type View = 'form-login' | 'form-code' | 'form-setup';
const VIEWS: View[] = ['form-login', 'form-code', 'form-setup'];
let setupCode = '';

function show(view: View): void {
  for (const id of VIEWS) byId(id).hidden = id !== view;
  const setup = view !== 'form-login';
  byId('auth-steps').hidden = !setup;
  byId('auth-back').hidden = view !== 'form-setup';
  byId('auth-head').classList.toggle('idle', !setup);
  const [code, password] = byId('auth-steps').children;
  if (code) code.className = view === 'form-code' ? 'on' : '';
  if (password) password.className = view === 'form-setup' ? 'on' : '';
  const title = byId('auth-title');
  const sub = byId('auth-sub');
  if (view === 'form-login') {
    title.textContent = 'Bentornato';
    sub.textContent = 'Accedi per revisionare i cartelli.';
    byId('login-user').focus();
  } else if (view === 'form-code') {
    title.textContent = 'Configuriamo il sito';
    sub.textContent = 'Prima il codice di configurazione di questo server.';
    byId('setup-code').focus();
  } else {
    title.textContent = 'Crea l’amministratore';
    sub.textContent = 'Il nome e la password con cui entrerai da ora in poi.';
    byId('setup-password').focus();
  }
}

/** Signed in: the page the browser asked for opens by itself. */
function done(): void {
  location.reload();
}

const value = (id: string): string => byId<HTMLInputElement>(id).value;

/** A form whose button is disabled while it waits, with its error line. */
function guard(form: View, button: string, error: string, action: () => Promise<void>): void {
  byId<HTMLFormElement>(form).addEventListener('submit', async (event) => {
    event.preventDefault();
    byId(error).textContent = '';
    byId<HTMLButtonElement>(button).disabled = true;
    try {
      await action();
    } catch (problem) {
      byId(error).textContent = messageOf(problem);
    } finally {
      byId<HTMLButtonElement>(button).disabled = false;
    }
  });
}

guard('form-login', 'login-submit', 'login-error', async () => {
  await post('/api/auth/login', {user: value('login-user'), password: value('login-password')}, true);
  done();
});
guard('form-code', 'code-submit', 'code-error', async () => {
  const code = value('setup-code').trim();
  if (!code) throw new Error('Scrivi il codice di configurazione.');
  await post('/api/auth/setup/check', {code}, true);
  setupCode = code;
  show('form-setup');
});
guard('form-setup', 'setup-submit', 'setup-error', async () => {
  const password = value('setup-password');
  if (password.length < 12) throw new Error('La password deve avere almeno 12 caratteri.');
  if (password !== value('setup-repeat')) throw new Error('Le due password sono diverse.');
  await post('/api/auth/setup', {code: setupCode, user: value('setup-user').trim(), password}, true);
  done();
});
byId('auth-back').addEventListener('click', () => show('form-code'));
byId('auth-transport').hidden = location.protocol === 'https:';

api<AuthState>('/api/auth/state', {}, true).then((state) => {
  if (state.signed_in) done();
  else show(state.configured ? 'form-login' : 'form-code');
}).catch((problem: unknown) => { byId('auth-sub').textContent = messageOf(problem); });
