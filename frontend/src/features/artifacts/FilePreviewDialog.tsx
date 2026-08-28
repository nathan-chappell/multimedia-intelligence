import { useCallback, useEffect, useRef, useState } from "react";

import type { HydratedLocalFile, IncludedLocalFile } from "./fileData";

const MAX_TEXT_PREVIEW_CHARACTERS = 100_000;

interface FilePreviewDialogProps {
  entry: IncludedLocalFile;
  resolveFile: (assetId: string) => Promise<HydratedLocalFile | undefined>;
  onClose: () => void;
}

export function FilePreviewDialog({
  entry,
  resolveFile,
  onClose,
}: FilePreviewDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const [preview, setPreview] = useState<PreviewState>({ status: "loading" });

  useEffect(() => {
    const dialog = dialogRef.current;
    returnFocusRef.current = document.activeElement as HTMLElement | null;
    if (dialog && !dialog.open) dialog.showModal();
    return () => returnFocusRef.current?.focus();
  }, []);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;

    void resolveFile(entry.durableAssetId ?? entry.id)
      .then(async (resolved) => {
        if (!resolved) throw new Error("This file is no longer available in the workspace.");
        if (resolved.route === "text" || resolved.route === "json" || resolved.route === "csv") {
          const contents = await resolved.file.text();
          if (!active) return;
          setPreview({
            status: "ready",
            file: resolved,
            text: contents.slice(0, MAX_TEXT_PREVIEW_CHARACTERS),
            truncated: contents.length > MAX_TEXT_PREVIEW_CHARACTERS,
          });
          return;
        }

        objectUrl = URL.createObjectURL(resolved.file);
        if (!active) {
          URL.revokeObjectURL(objectUrl);
          objectUrl = null;
          return;
        }
        setPreview({ status: "ready", file: resolved, objectUrl });
      })
      .catch((error: unknown) => {
        if (!active) return;
        setPreview({
          status: "error",
          message: error instanceof Error ? error.message : "Could not load the file preview.",
        });
      });

    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [entry, resolveFile]);

  const closePreview = useCallback(() => {
    dialogRef.current?.close();
    onClose();
  }, [onClose]);

  return (
    <dialog
      ref={dialogRef}
      className="file-preview-dialog"
      aria-labelledby="file-preview-title"
      onCancel={(event) => {
        event.preventDefault();
        closePreview();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) closePreview();
      }}
    >
      <div className="file-preview-shell">
        <header className="file-preview-header">
          <div>
            <span className="eyebrow">File preview</span>
            <h2 id="file-preview-title" title={entry.filename}>{entry.filename}</h2>
            <small>{entry.route} · {formatBytes(entry.sizeBytes)}</small>
          </div>
          <button type="button" aria-label="Close preview" onClick={closePreview} autoFocus>×</button>
        </header>
        <div className="file-preview-content">
          {preview.status === "loading" && <p className="file-preview-state" role="status">Loading preview…</p>}
          {preview.status === "error" && <p className="file-preview-state save-error" role="alert">{preview.message}</p>}
          {preview.status === "ready" && <PreviewContent preview={preview} />}
        </div>
      </div>
    </dialog>
  );
}

function PreviewContent({ preview }: { preview: ReadyPreview }) {
  const { file, objectUrl } = preview;
  if (file.route === "text" || file.route === "json" || file.route === "csv") {
    return (
      <div className="file-preview-text">
        <pre>{preview.text}</pre>
        {preview.truncated && (
          <p role="note">Preview truncated after {MAX_TEXT_PREVIEW_CHARACTERS.toLocaleString()} characters.</p>
        )}
      </div>
    );
  }
  if (!objectUrl) return null;
  if (file.route === "pdf") {
    return <iframe className="file-preview-pdf" src={objectUrl} title={`Preview of ${file.filename}`} />;
  }
  if (file.route === "image") {
    return <img className="file-preview-image" src={objectUrl} alt={`Preview of ${file.filename}`} />;
  }
  if (file.route === "audio") {
    return <audio className="file-preview-media" src={objectUrl} controls />;
  }
  if (file.route === "video") {
    return <video className="file-preview-video" src={objectUrl} controls />;
  }
  return (
    <div className="file-preview-unsupported">
      <p>This format cannot be previewed in the browser.</p>
      <dl>
        <div><dt>Type</dt><dd>{file.mediaType || "Unknown"}</dd></div>
        <div><dt>Size</dt><dd>{formatBytes(file.sizeBytes)}</dd></div>
      </dl>
      <a href={objectUrl} download={file.filename}>Download file</a>
    </div>
  );
}

type PreviewState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | ReadyPreview;

type ReadyPreview = {
  status: "ready";
  file: HydratedLocalFile;
  objectUrl?: string;
  text?: string;
  truncated?: boolean;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}
