import { useRef } from "react";

import { useFileWorkspace } from "./useFileWorkspace";
import { useTransientStatus } from "../status/transientStatus";

const acceptedExtensions = [
  ".md",
  ".txt",
  ".json",
  ".csv",
  ".pdf",
  ".png",
  ".jpeg",
  ".jpg",
  ".webp",
  ".gif",
  ".flac",
  ".mp3",
  ".mpga",
  ".m4a",
  ".ogg",
  ".wav",
  ".mp4",
  ".mpeg",
  ".webm",
].join(",");

export function ArtifactPanel() {
  const inputRef = useRef<HTMLInputElement>(null);
  const { files, artifacts, addFiles, removeFile, saveFile } = useFileWorkspace();
  const { showStatus } = useTransientStatus();

  return (
    <aside className="panel artifact-panel" aria-labelledby="artifact-title">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Artifacts</span>
          <h2 id="artifact-title">Conversation files</h2>
        </div>
        <span className="counter">{files.length}</span>
      </div>

      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        aria-label="Conversation files"
        accept={acceptedExtensions}
        multiple
        onChange={(event) => {
          if (event.currentTarget.files?.length) {
            const count = event.currentTarget.files.length;
            addFiles(event.currentTarget.files);
            showStatus(`${count} ${count === 1 ? "file" : "files"} staged`, {
              tone: "success",
            });
          }
          event.currentTarget.value = "";
        }}
      />

      {files.length === 0 ? (
        <div className="empty-state">
          <div className="file-glyph" aria-hidden="true">
            <span />
          </div>
          <h3>No files staged</h3>
          <p>Select files to inspect locally, then save the ones this conversation should keep.</p>
          <button type="button" onClick={() => inputRef.current?.click()}>
            Select local files
          </button>
        </div>
      ) : (
        <div className="artifact-content">
          <button className="add-file-button" type="button" onClick={() => inputRef.current?.click()}>
            Add more files
          </button>
          <div className="file-list" aria-label="Locally staged conversation files">
            {files.map((entry) => (
              <article className="file-card" key={entry.id}>
                <div>
                  <strong>{entry.file.name}</strong>
                  <small>
                    {entry.route} · {formatBytes(entry.file.size)}
                  </small>
                  <code title={entry.id}>{entry.id.slice(0, 20)}…</code>
                  <small className={`durability durability-${entry.durability}`}>
                    {durabilityLabel(entry.durability)}
                  </small>
                  {entry.saveError && <small className="save-error">{entry.saveError}</small>}
                </div>
                <div className="file-actions">
                  {entry.durability !== "included" && (
                    <button
                      type="button"
                      disabled={entry.durability === "uploading"}
                      onClick={() => void saveFile(entry.id)}
                      aria-label={`Save ${entry.file.name}`}
                    >
                      {entry.durability === "uploading" ? "Saving…" : "Save"}
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => removeFile(entry.id)}
                    aria-label={`Remove ${entry.file.name}`}
                  >
                    Remove
                  </button>
                </div>
              </article>
            ))}
          </div>

          {artifacts.length > 0 && (
            <section className="derived-artifacts" aria-labelledby="derived-title">
              <span className="eyebrow" id="derived-title">
                Transient derivatives
              </span>
              {artifacts.map((artifact) => (
                <article className="artifact-card" key={artifact.id}>
                  {artifact.previewUrl && <img src={artifact.previewUrl} alt={artifact.label} />}
                  <div>
                    <strong>{artifact.label}</strong>
                    <small>{formatBytes(artifact.blob.size)} · local preview</small>
                  </div>
                </article>
              ))}
            </section>
          )}
        </div>
      )}
    </aside>
  );
}

function durabilityLabel(durability: string): string {
  const labels: Record<string, string> = {
    local: "Staged locally",
    uploading: "Saving",
    stored: "Stored; waiting for a conversation",
    included: "Saved to this conversation",
    error: "Save needs attention",
  };
  return labels[durability] ?? durability;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
