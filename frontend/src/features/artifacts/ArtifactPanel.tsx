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
  const selectedCollection = library.collections.find((collection) => collection.selected);

  async function updateInclusion(file: CollectionFileSummary) {
    setBusyAssetId(file.asset_id);
    setMessage(null);
    try {
      await library.setCollectionFileIncluded(file.asset_id, !file.included);
      setMessage(
        file.included
          ? "Removed from the conversation workspace"
          : "Added to the conversation workspace",
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

  async function updateVisibility() {
    if (!selectedCollection?.can_manage) return;
    setMessage(null);
    try {
      await library.setCollectionPublic(selectedCollection.id, !selectedCollection.is_public);
      setMessage(
        selectedCollection.is_public
          ? "Collection is now private"
          : "Collection is now public and visible to signed-in users",
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not update visibility");
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
            <small>Durable library for uploads, ingestion, and search</small>
          </div>
          <span className="counter">{library.collectionFiles.length}</span>
        </div>
      <div className="collection-control">
        <label htmlFor={fullPage ? "active-collection-page" : "active-collection"}>Active collection</label>
        <div>
          <select
            id={fullPage ? "active-collection-page" : "active-collection"}
            value={library.selectedCollectionId ?? ""}
            disabled={library.collections.length === 0}
            onChange={(event) => {
              void library.selectCollection(event.currentTarget.value).catch((error: unknown) =>
                setMessage(error instanceof Error ? error.message : "Could not select collection"),
              );
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
        {selectedCollection && (
          <div className="collection-access-summary">
            <span className={`visibility-badge${selectedCollection.is_public ? " public" : ""}`}>
              {selectedCollection.is_public ? "Public" : "Private"}
            </span>
            <small>
              {selectedCollection.read_only
                ? "Shared read-only collection. Chat can search its indexed files."
                : selectedCollection.description || "Only you can add files to this collection."}
            </small>
            {selectedCollection.can_manage && (
              <button type="button" onClick={() => void updateVisibility()}>
                {selectedCollection.is_public ? "Make private" : "Make public"}
              </button>
            )}
          </div>
        )}
        <div className="collection-secondary-actions">
          <button
            type="button"
            disabled={selectedCollection?.read_only}
            title={selectedCollection?.read_only ? "Shared collections are read-only" : undefined}
            onClick={() => inputRef.current?.click()}
          >Upload files</button>
          <button
            type="button"
            disabled={reconciling || !selectedCollection?.can_manage}
            title={selectedCollection?.can_manage ? undefined : "Only the owner can refresh index status"}
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
        aria-label="Collection files"
        accept={acceptedExtensions}
        multiple
        onChange={(event) => {
          if (event.currentTarget.files?.length) {
            workspace.addFiles(event.currentTarget.files);
            setMessage(`${event.currentTarget.files.length} file(s) staged for upload`);
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
            <p>
              {selectedCollection?.read_only
                ? "The collection owner has not shared any files yet."
                : "Upload a file to make it available for indexing and conversations."}
            </p>
          </div>
        ) : (
          <div className="collection-file-list" aria-label="Files in the selected collection">
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
                    selectedCollection?.read_only ||
                    !workspace.activeThreadId ||
                    busyAssetId === file.asset_id
                  }
                  title={
                    selectedCollection?.read_only
                      ? "Shared collection files are available through chat search"
                      : workspace.activeThreadId
                        ? undefined
                        : "Start or select a conversation first"
                  }
                  onClick={() => void updateInclusion(file)}
                >
                  {selectedCollection?.read_only
                    ? "Shared"
                    : busyAssetId === file.asset_id
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
            <span className="eyebrow" id="workspace-title">Conversation workspace</span>
            <small>Files and previews available to browser tools in this conversation</small>
          </div>
          <span className="counter">{workspace.files.length}</span>
        </div>
        <div className="artifact-content">
        {workspace.files.length === 0 ? (
          <div className="empty-state compact-empty">
            <h3>No workspace files</h3>
            <p>Add a collection file or stage an upload for this conversation.</p>
          </div>
        ) : (
          <div className="workspace-file-list" aria-label="Files in the conversation workspace">
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
                    disabled={entry.durability === "uploading" || selectedCollection?.read_only}
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

function collectionLabel(collection: {
  name: string;
  is_public: boolean;
  owned: boolean;
}): string {
  if (collection.is_public) return `${collection.name} · Public${collection.owned ? "" : " · Shared"}`;
  return collection.owned ? collection.name : `${collection.name} · Private · Admin`;
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
