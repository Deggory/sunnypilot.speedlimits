// GET /api/auth/state, and the answers of the sign-in calls.

export interface AuthState {
  configured: boolean;
  signed_in: boolean;
  /** The site on 127.0.0.1 without sign-in. */
  local: boolean;
  user?: string;
  sessions?: number;
}

export interface AuthAnswer {
  ok: boolean;
  state: AuthState;
}
