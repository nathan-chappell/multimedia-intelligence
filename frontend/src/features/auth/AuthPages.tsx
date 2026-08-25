/* eslint-disable react-refresh/only-export-components */
import { SignIn, SignUp, useAuth } from "@clerk/react";
import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";

import { authenticatedFetch, setClerkTokenGetter } from "../../lib/config";

export interface SessionUser {
  id: string;
  username: string;
  email: string | null;
  full_name: string | null;
  role: "admin" | "user";
  is_admin: boolean;
  balance_microusd: number;
}

interface SessionContextValue {
  user: SessionUser;
  reload: () => Promise<void>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function useSessionUser(): SessionContextValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSessionUser must be used inside AuthGate");
  return value;
}

export function AuthGate({ children }: { children: ReactNode }) {
  if (import.meta.env.VITE_E2E_AUTH === "true") {
    return <E2EAuthGate>{children}</E2EAuthGate>;
  }
  return <ClerkAuthGate>{children}</ClerkAuthGate>;
}

function ClerkAuthGate({ children }: { children: ReactNode }) {
  const { getToken, isLoaded, isSignedIn } = useAuth();
  const [user, setUser] = useState<SessionUser>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    setClerkTokenGetter(async () => (await getToken()) ?? null);
    return () => setClerkTokenGetter(undefined);
  }, [getToken]);

  const reload = useCallback(async () => {
    const response = await authenticatedFetch("/api/auth/me");
    if (!response.ok) {
      const detail = await response.json().catch(() => null) as { detail?: string } | null;
      throw new Error(detail?.detail ?? "We could not verify your Clerk session.");
    }
    setUser((await response.json()) as SessionUser);
  }, []);

  useEffect(() => {
    if (!isLoaded) return;
    if (!isSignedIn) {
      window.location.replace("/login");
      return;
    }
    queueMicrotask(() => {
      void reload().catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Authentication failed");
      });
    });
  }, [isLoaded, isSignedIn, reload]);

  if (!isLoaded || (isSignedIn && !user && !error)) {
    return <AuthStatus title="Checking your Clerk session…" />;
  }
  if (error) {
    return (
      <main className="auth-page">
        <section className="auth-card auth-card-centered">
          <span className="eyebrow">Authentication error</span>
          <h1>We couldn’t open your workspace.</h1>
          <p>{error}</p>
          <button type="button" onClick={() => window.location.reload()}>Try again</button>
        </section>
      </main>
    );
  }
  if (!user) return null;

  return <SessionContext.Provider value={{ user, reload }}>{children}</SessionContext.Provider>;
}

function E2EAuthGate({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SessionUser>();
  const [error, setError] = useState<string>();
  const token = window.localStorage.getItem("e2e_clerk_token");

  const reload = useCallback(async () => {
    const response = await authenticatedFetch("/api/auth/me");
    if (!response.ok) throw new Error("Your Clerk session is invalid or has expired.");
    setUser((await response.json()) as SessionUser);
  }, []);

  useEffect(() => {
    if (!token) {
      window.location.replace("/login");
      return;
    }
    setClerkTokenGetter(async () => token);
    queueMicrotask(() => {
      void reload().catch((reason: unknown) => setError(String(reason)));
    });
    return () => setClerkTokenGetter(undefined);
  }, [reload, token]);

  if (error) return <AuthStatus title={error} />;
  if (!user) return <AuthStatus title="Checking your Clerk session…" />;
  return <SessionContext.Provider value={{ user, reload }}>{children}</SessionContext.Provider>;
}

export function LoginPage() {
  return (
    <main className="auth-page">
      <SignIn routing="path" path="/login" signUpUrl="/sign-up" forceRedirectUrl="/" />
    </main>
  );
}

export function SignUpPage() {
  return (
    <main className="auth-page">
      <SignUp routing="path" path="/sign-up" signInUrl="/login" forceRedirectUrl="/account" />
    </main>
  );
}

function AuthStatus({ title }: { title: string }) {
  return (
    <main className="auth-page" aria-live="polite">
      <section className="auth-card auth-card-centered">
        <span className="eyebrow">Multimedia Intelligence</span>
        <h1>{title}</h1>
      </section>
    </main>
  );
}
