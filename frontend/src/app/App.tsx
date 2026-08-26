import { UserButton } from "@clerk/react";
import { ArtifactPanel } from "../features/artifacts/ArtifactPanel";
import { FileWorkspaceProvider } from "../features/artifacts/FileWorkspaceProvider";
import { ChatPanel } from "../features/chat/ChatPanel";
import { AuthGate, LoginPage, SignUpPage, useSessionUser } from "../features/auth/AuthPages";
import { AccountPage, AdminPage } from "../features/billing/BillingPages";
import { Link } from "../lib/Link";
import { usePathname } from "../lib/navigation";

export function App() {
  const pathname = usePathname();
  if (pathname === "/login") {
    return <LoginPage />;
  }

  if (pathname === "/sign-up") {
    return <SignUpPage />;
  }

  return (
    <AuthGate>
      <AuthenticatedRoute />
    </AuthGate>
  );
}

function AuthenticatedRoute() {
  const { user } = useSessionUser();
  const pathname = usePathname();
  if (pathname === "/") return <Workspace />;
  if (pathname === "/files") return <FilesPage />;
  if (pathname === "/account") return <AccountPage />;
  if (pathname === "/admin" && user.is_admin) return <AdminPage />;
  return <NotFoundPage />;
}

function Workspace() {
  const { user } = useSessionUser();
  return (
    <FileWorkspaceProvider>
      <main className="app-shell">
        <WorkspaceHeader user={user} eyebrow="Conversation workspace" />
        <section className="workspace" aria-label="Conversation workspace">
          <ChatPanel />
          <ArtifactPanel />
        </section>
      </main>
    </FileWorkspaceProvider>
  );
}

function FilesPage() {
  const { user } = useSessionUser();
  return (
    <FileWorkspaceProvider>
      <main className="app-shell files-page-shell">
        <WorkspaceHeader user={user} eyebrow="Collection library" />
        <ArtifactPanel fullPage />
      </main>
    </FileWorkspaceProvider>
  );
}

function WorkspaceHeader({ user, eyebrow }: { user: ReturnType<typeof useSessionUser>["user"]; eyebrow: string }) {
  return (
    <header className="masthead">
      <div><span className="eyebrow">{eyebrow}</span><h1>Multimedia Intelligence</h1></div>
      <div className="masthead-actions">
        <Link href="/" className="nav-link">Chat</Link>
        <Link href="/files" className="nav-link">Files</Link>
        <Link href="/account" className="balance-chip">{formatUsd(user.balance_microusd)} credit</Link>
        {user.is_admin && <Link href="/admin" className="nav-link">Admin</Link>}
        <UserButton />
      </div>
    </header>
  );
}

function formatUsd(microusd: number): string {
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" })
    .format(microusd / 1_000_000);
}

function NotFoundPage() {
  return (
    <main className="not-found-page">
      <p className="eyebrow">404 · Not found</p>
      <h1>This page isn’t part of the workspace.</h1>
      <p>The address may be outdated, or the page may have moved.</p>
      <Link href="/">Return to the conversation workspace</Link>
    </main>
  );
}
