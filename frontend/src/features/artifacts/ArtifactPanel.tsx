import { useRef } from "react";

import { useFileWorkspace } from "./useFileWorkspace";

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
  const { files, artifacts, addFiles, removeFile } = useFileWorkspace();

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
          if (event.currentTarget.files) addFiles(event.currentTarget.files);
          event.currentTarget.value = "";
        }}
      />

      {files.length === 0 ? (
        <div className="empty-state">
          <div className="file-glyph" aria-hidden="true">
            <span />
          </div>
          <h3>No files staged</h3>
          <p>
            Select local files for bounded browser inspection. Bucket upload and durable
            thread inclusion remain a separate backend step.
          </p>
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
                </div>
                <button type="button" onClick={() => removeFile(entry.id)} aria-label={`Remove ${entry.file.name}`}>
                  Remove
                </button>
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
                    <small>{formatBytes(artifact.blob.size)} · browser only</small>
                  </div>
                </article>
              ))}
            </section>
          )}
        </div>
      )}

      <p className="implementation-note">
        Local staging is intentionally not durable. Any original or derivative sent to
        OpenAI must first be finalized in the configured S3-compatible bucket.
      </p>
    </aside>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
