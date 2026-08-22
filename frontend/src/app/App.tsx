import { ArtifactPanel } from "../features/artifacts/ArtifactPanel";
import { FileWorkspaceProvider } from "../features/artifacts/FileWorkspaceProvider";
import { ChatPanel } from "../features/chat/ChatPanel";

export function App() {
  return (
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
  );
}
