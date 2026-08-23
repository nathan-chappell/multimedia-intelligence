import { ArtifactPanel } from "../features/artifacts/ArtifactPanel";
import { FileWorkspaceProvider } from "../features/artifacts/FileWorkspaceProvider";
import { ChatPanel } from "../features/chat/ChatPanel";
import { AuthErrorPage, AuthGate, LoginPage } from "../features/auth/AuthPages";
import { TransientStatusProvider } from "../features/status/TransientStatusProvider";

export function App() {
  if (window.location.pathname === "/login") {
    return <LoginPage />;
  }

  if (window.location.pathname === "/auth-error") {
    return <AuthErrorPage />;
  }

  return (
    <AuthGate>
      {window.location.pathname === "/" ? <Workspace /> : <NotFoundPage />}
    </AuthGate>
  );
}

function Workspace() {
  return (
    <TransientStatusProvider>
      <FileWorkspaceProvider>
        <main className="app-shell">
          <header className="masthead">
            <div>
              <span className="eyebrow">Conversation workspace</span>
              <h1>Multimedia Intelligence</h1>
            </div>
            <div className="scope-chip">
              <span aria-hidden="true" /> Files stay scoped to this conversation
            </div>
          </header>

          <section className="workspace" aria-label="Conversation workspace">
            <ChatPanel />
            <ArtifactPanel />
          </section>
        </main>
      </FileWorkspaceProvider>
    </TransientStatusProvider>
  );
}

function NotFoundPage() {
  return (
    <main className="not-found-page">
      <p className="eyebrow">404 · Not found</p>
      <h1>This page isn’t part of the workspace.</h1>
      <p>The address may be outdated, or the page may have moved.</p>
      <a href="/">Return to the conversation workspace</a>
    </main>
  );
}
