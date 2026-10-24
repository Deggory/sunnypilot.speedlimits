// Calls to the server of the site: JSON, same origin, never cached.

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

interface Problem {
  message?: string;
}

/**
 * GET (or the given method) of a JSON API. A signed-out browser goes back to the sign-in page, except on
 * the sign-in page itself (signIn), where a 401 is a wrong name or password to show.
 */
export async function api<T>(path: string, options: RequestInit = {}, signIn = false): Promise<T> {
  const response = await fetch(path, {credentials: 'same-origin', cache: 'no-store', ...options});
  if (response.status === 401 && !signIn) {
    location.replace('/');
    throw new ApiError('Sessione finita.', 401);
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    data = {message: 'Risposta non valida dal server (HTTP ' + response.status + ').'};
  }
  if (!response.ok) {
    const problem = data as Problem;
    throw new ApiError(problem.message || 'HTTP ' + response.status, response.status);
  }
  return data as T;
}

/** POST of a JSON body (the server accepts only JSON, against cross-site forms). */
export function post<T>(path: string, body: unknown = {}, signIn = false): Promise<T> {
  return api<T>(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)},
    signIn);
}

/** The message of anything thrown. */
export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
