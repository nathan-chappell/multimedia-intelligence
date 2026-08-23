import { FormEvent, ReactNode, useEffect, useState } from "react";

import {
  clearBearerToken,
  getAuthError,
  getBearerToken,
  markAuthFailure,
  storeBearerToken,
} from "../../lib/config";

type AuthState = "checking" | "authenticated";

export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>("checking");

  useEffect(() => {
    const token = getBearerToken();
    if (!token) {
      window.location.replace("/login");
      return;
    }

    const controller = new AbortController();
    fetch("/api/auth/me", {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then((response) => {
        if (response.ok) {
          setState("authenticated");
          return;
        }
        if (response.status === 401) {
          markAuthFailure("Your session is invalid or has expired.");
        } else {
          markAuthFailure("We could not verify your session.");
        }
        window.location.replace("/auth-error");
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        markAuthFailure("The authentication service could not be reached.");
        window.location.replace("/auth-error");
      });

    return () => controller.abort();
  }, []);

  if (state === "checking") {
    return (
      <main className="auth-page" aria-live="polite">
        <section className="auth-card auth-card-centered">
          <span className="eyebrow">Multimedia Intelligence</span>
          <h1>Checking your session…</h1>
        </section>
      </main>
    );
  }

  return children;
}

export function LoginPage() {
  const [error, setError] = useState<string>();
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(undefined);
    setSubmitting(true);
    const form = new FormData(event.currentTarget);
    const body = new URLSearchParams({
      username: String(form.get("username") ?? ""),
      password: String(form.get("password") ?? ""),
    });

    try {
      const response = await fetch("/api/auth/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body,
      });
      if (!response.ok) {
        setError(
          response.status === 401
            ? "The username or password is incorrect."
            : "Sign-in is unavailable right now. Please try again.",
        );
        return;
      }
      const result = (await response.json()) as { access_token?: unknown };
      if (typeof result.access_token !== "string" || !result.access_token) {
        setError("The server returned an invalid sign-in response.");
        return;
      }
      storeBearerToken(result.access_token);
      window.location.replace("/");
    } catch {
      setError("The authentication service could not be reached.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <main className="auth-page">
      <section className="auth-card">
        <span className="eyebrow">Multimedia Intelligence</span>
        <h1>Welcome back.</h1>
        <p>Sign in to open your conversation workspace.</p>
        <form onSubmit={submit}>
          <label htmlFor="username">Username</label>
          <input id="username" name="username" autoComplete="username" required autoFocus />
          <label htmlFor="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
            required
          />
          {error && <p className="form-error" role="alert">{error}</p>}
          <button type="submit" disabled={submitting}>
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </section>
    </main>
  );
}

export function AuthErrorPage() {
  const message = getAuthError() ?? "Your session could not be verified.";

  const resetSession = () => {
    clearBearerToken();
    window.sessionStorage.removeItem("auth_error");
  };

  return (
    <main className="auth-page">
      <section className="auth-card auth-card-centered">
        <span className="eyebrow">Authentication error</span>
        <h1>We couldn’t open your workspace.</h1>
        <p>{message}</p>
        <a href="/login" onClick={resetSession}>Return to sign in</a>
      </section>
    </main>
  );
}
