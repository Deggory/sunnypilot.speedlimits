// Calls to the server of the site: JSON, same origin, never cached.

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

interface Problem {
  message?: string;
}

/** The message of a call that got no answer (status 0). */
export const UNREACHABLE = 'Can\'t reach the site.';

/**
 * GET (or the given method) of a JSON API. A signed-out browser reloads the page, which the server answers with
 * the sign-in page at the same address (signing in brings the page back), except on the sign-in page itself
 * (signIn), where a 401 is a wrong name or password to show.
 */
export async function api<T>(path: string, options: RequestInit = {}, signIn = false): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {credentials: 'same-origin', cache: 'no-store', ...options});
  } catch {
    // No answer at all (the network, the site restarting): words, not the browser's "Failed to fetch".
    throw new ApiError(UNREACHABLE, 0);
  }
  if (response.status === 401 && !signIn) {
    location.reload();
    throw new ApiError('Signed out: sign in again.', 401);
  }
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    data = {message: 'The site gave an answer this page cannot read (error ' + response.status + ').'};
  }
  if (!response.ok) {
    const problem = data as Problem;
    throw new ApiError(problem.message || 'The site had a problem (error ' + response.status + ').', response.status);
  }
  return data as T;
}

/**
 * GET of the pollers of the top bar (the server, the comma): a signed-out browser gets an ApiError 401 and the page
 * stays as it is, so the review page keeps its unsaved confirmations on screen (review/server.ts).
 */
export function poll<T>(path: string): Promise<T> {
  return api<T>(path, {}, true);
}

/** True for the error of a signed-out browser. */
export function signedOut(error: unknown): boolean {
  return error instanceof ApiError && error.status === 401;
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
