import { useRef, useState } from "react";

import { useCollectionLibrary, useConversationWorkspace } from "./useFileData";
import type { CollectionFileSummary } from "./fileData";

const acceptedExtensions = [
  ".md", ".txt", ".json", ".csv", ".pdf", ".png", ".jpeg", ".jpg", ".webp",
  ".gif", ".flac", ".mp3", ".mpga", ".m4a", ".ogg", ".wav", ".mp4", ".mpeg",
  ".webm",
].join(",");

export function ArtifactPanel({ fullPage = false }: { fullPage?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busyAssetId, setBusyAssetId] = useState<string | null>(null);
  const [reconciling, setReconciling] = useState(false);
  const library = useCollectionLibrary();
  const workspace = useConversationWorkspace();
  const focusedCollection = library.collections.find(
    (collection) => collection.id === library.focusedCollectionId,
  );

  async function updateInclusion(file: CollectionFileSummary) {
    setBusyAssetId(file.asset_id);
    setMessage(null);
    try {
      await library.setCollectionFileIncluded(file.asset_id, !file.included);
      setMessage(
        file.included
          ? "Removed from your workspace"
          : "Added to your workspace",
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not update the conversation");
    } finally {
      setBusyAssetId(null);
    }
  }

  async function refreshIndex() {
    setReconciling(true);
    setMessage("Checking OpenAI index status…");
    try {
      const result = await library.reconcileCollection();
      setMessage(
        result.provider_error
          ? `Provider check failed: ${result.provider_error}`
          : `${result.ready} ready · ${result.missing} missing · ${result.orphaned} orphaned`,
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not refresh index status");
    } finally {
      setReconciling(false);
    }
  }

  return (
    <aside className={`panel artifact-panel${fullPage ? " artifact-panel-page" : ""}`} aria-labelledby="artifact-title">
      <div className="panel-heading artifact-heading">
        <div><span className="eyebrow">Files</span><h2 id="artifact-title">Library & workspace</h2></div>
        <span className="counter">{workspace.files.length}</span>
      </div>

      <section className="file-section collection-section" aria-labelledby="collection-title">
        <div className="file-section-heading">
          <div>
            <span className="eyebrow" id="collection-title">Collection</span>
            <small>Semantic search index for workspace files</small>
          </div>
          <span className="counter">{library.collectionFiles.length}</span>
        </div>
      <div className="collection-control">
        <label htmlFor={fullPage ? "browse-collection-page" : "browse-collection"}>Browse collection</label>
        <div>
          <select
            id={fullPage ? "browse-collection-page" : "browse-collection"}
            value={library.focusedCollectionId ?? ""}
            disabled={library.collections.length === 0}
            onChange={(event) => {
              library.selectCollection(event.currentTarget.value);
            }}
          >
            {library.collections.map((collection) => (
              <option key={collection.id} value={collection.id}>
                {collectionLabel(collection)}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => {
              const name = window.prompt("Name the new collection");
              if (!name?.trim()) return;
              void library.createCollection(name).catch((error: unknown) =>
                setMessage(error instanceof Error ? error.message : "Could not create collection"),
              );
            }}
          >New</button>
        </div>
        {focusedCollection && (
          <div className="collection-access-summary">
            <span className="visibility-badge">{focusedCollection.slug}</span>
            <small>{focusedCollection.description || "Your private semantic search index."}</small>
          </div>
        )}
        <div className="collection-secondary-actions">
          <button
            type="button"
            disabled={reconciling || !focusedCollection}
            onClick={() => void refreshIndex()}
          >
            {reconciling ? "Refreshing…" : "Refresh index status"}
          </button>
        </div>
      </div>

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        aria-label="Workspace files"
        accept={acceptedExtensions}
        multiple
        onChange={(event) => {
          if (event.currentTarget.files?.length) {
            workspace.addFiles(event.currentTarget.files);
            setMessage(`${event.currentTarget.files.length} file(s) added to the workspace`);
          }
          event.currentTarget.value = "";
        }}
      />

      {message && <p className="collection-message" role="status">{message}</p>}
      {library.collectionFilesError && (
        <p className="collection-message collection-message-error" role="alert">{library.collectionFilesError}</p>
      )}

      <div className="artifact-content">
        {library.collectionFilesLoading ? (
          <p className="compact-empty">Loading collection files…</p>
        ) : library.collectionFiles.length === 0 ? (
          <div className="empty-state compact-empty">
            <h3>This collection is empty</h3>
            <p>Ask the agent to add a workspace file to this collection.</p>
          </div>
        ) : (
          <div className="collection-file-list" aria-label="Files in the focused collection">
            {library.collectionFiles.map((file) => (
              <article className="collection-file-row" key={file.asset_id}>
                <span className="file-route-icon" aria-hidden="true">{routeIcon(file.route)}</span>
                <div className="collection-file-copy">
                  <strong title={file.filename}>{file.filename}</strong>
                  <small>{file.route} · {formatBytes(file.size_bytes)} · {formatDate(file.created_at)}</small>
                  <span className={`index-badge index-badge-${file.provider_status}`}>{statusLabel(file)}</span>
                  {file.last_error && <small className="save-error" title={file.last_error}>Indexing needs attention</small>}
                </div>
                <button
                  className={`include-toggle${file.included ? " included" : ""}`}
                  type="button"
                  disabled={
                    busyAssetId === file.asset_id
                  }
                  onClick={() => void updateInclusion(file)}
                >
                  {busyAssetId === file.asset_id
                      ? "…"
                      : file.included
                        ? "In workspace"
                        : "Add to workspace"}
                </button>
              </article>
            ))}
          </div>
        )}
      </div>
      </section>

      <section className="file-section workspace-section" aria-labelledby="workspace-title">
        <div className="file-section-heading">
          <div>
            <span className="eyebrow" id="workspace-title">Workspace</span>
            <small>Your durable files, loaded by tools only when needed</small>
          </div>
          <div className="collection-secondary-actions">
            <span className="counter">{workspace.files.length}</span>
            <button type="button" onClick={() => inputRef.current?.click()}>Add files</button>
          </div>
        </div>
        <div className="artifact-content">
        {workspace.files.length === 0 ? (
          <div className="empty-state compact-empty">
            <h3>No workspace files</h3>
            <p>Add a local file or open one from a collection.</p>
          </div>
        ) : (
          <div className="workspace-file-list" aria-label="Files in the workspace">
            {workspace.files.map((entry) => (
              <article className="collection-file-row" key={entry.id}>
                <span className="file-route-icon" aria-hidden="true">{routeIcon(entry.route)}</span>
                <div className="collection-file-copy">
                  <strong title={entry.filename}>{entry.filename}</strong>
                  <small>{entry.route} · {formatBytes(entry.sizeBytes)}</small>
                  <span className="index-badge">
                    {entry.durability === "local" || entry.durability === "error"
                      ? "Browser staged"
                      : "Available to tools"}
                  </span>
                  {entry.saveError && <small className="save-error">{entry.saveError}</small>}
                </div>
                {!entry.durableAssetId && (
                  <button
                    className="include-toggle"
                    type="button"
                    disabled={entry.durability === "uploading"}
                    onClick={() => void workspace.saveFile(entry.id)}
                  >
                    {entry.durability === "uploading" ? "Uploading…" : "Upload"}
                  </button>
                )}
                <button
                  className="include-toggle"
                  type="button"
                  onClick={() => {
                    if (entry.durableAssetId) {
                      void workspace.setFileIncluded(entry.durableAssetId, false).catch(
                        (error: unknown) => setMessage(
                          error instanceof Error ? error.message : "Could not remove workspace file",
                        ),
                      );
                    } else {
                      workspace.removeFile(entry.id);
                    }
                  }}
                >Remove</button>
              </article>
            ))}
          </div>
        )}

        {workspace.artifacts.length > 0 && (
          <details className="derived-artifacts">
            <summary>Derived previews ({workspace.artifacts.length})</summary>
            {workspace.artifacts.map((artifact) => (
              <article className="artifact-card" key={artifact.id}>
                {artifact.previewUrl && <img src={artifact.previewUrl} alt={artifact.label} />}
                <div>
                  <strong>{artifact.label}</strong>
                  <small>
                    {formatBytes(artifact.blob.size)} · {artifact.durability === "saved" ? "saved" : "local preview"}
                  </small>
                </div>
              </article>
            ))}
          </details>
        )}
      </div>
      </section>
    </aside>
  );
}

function collectionLabel(collection: { name: string; slug: string }): string {
  return `${collection.name} · ${collection.slug}`;
}

function statusLabel(file: CollectionFileSummary): string {
  if (file.provider_status === "ready") return `${file.provider_file_count} indexed`;
  if (file.provider_status === "not_indexed") return "Not indexed";
  if (file.provider_status === "missing") return "Provider file missing";
  if (file.provider_status === "error") return "Index error";
  return file.ingestion_status.replaceAll("_", " ");
}

function routeIcon(route: string): string {
  const icons: Record<string, string> = { pdf: "PDF", csv: "CSV", json: "{ }", image: "IMG", audio: "AUD", video: "VID", text: "TXT" };
  return icons[route] ?? "FILE";
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
