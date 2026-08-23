import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { authenticatedFetch } from "../../lib/config";
import {
  classifyLocalFile,
  FileWorkspaceContext,
  type IncludedLocalFile,
  type TransientArtifact,
} from "./fileWorkspace";

export function FileWorkspaceProvider({ children }: { children: ReactNode }) {
  const [files, setFiles] = useState<IncludedLocalFile[]>([]);
  const [artifacts, setArtifacts] = useState<TransientArtifact[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const inclusionInFlight = useRef(new Set<string>());
  const artifactsRef = useRef(artifacts);

  useEffect(() => {
    artifactsRef.current = artifacts;
  }, [artifacts]);

  useEffect(
    () => () => {
      artifactsRef.current.forEach((artifact) => {
        if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
      });
    },
    [],
  );

  const addFiles = useCallback((incoming: FileList | readonly File[]) => {
    const additions = Array.from(incoming, (file): IncludedLocalFile => ({
      id: `local_${crypto.randomUUID()}`,
      file,
      route: classifyLocalFile(file),
      addedAt: Date.now(),
      durability: "local",
    }));
    setFiles((current) => [...current, ...additions]);
  }, []);

  const removeFile = useCallback((assetId: string) => {
    setFiles((current) => current.filter((entry) => entry.id !== assetId));
    setArtifacts((current) => {
      const retained: TransientArtifact[] = [];
      current.forEach((artifact) => {
        if (artifact.sourceAssetId === assetId) {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        } else {
          retained.push(artifact);
        }
      });
      return retained;
    });
  }, []);

  const getFile = useCallback(
    (assetId: string) => files.find((entry) => entry.id === assetId),
    [files],
  );

  const attachSavedFile = useCallback(async (localId: string, assetId: string, threadId: string) => {
    if (inclusionInFlight.current.has(localId)) return;
    inclusionInFlight.current.add(localId);
    try {
      const response = await authenticatedFetch(`/api/assets/${encodeURIComponent(assetId)}/includes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ thread_id: threadId }),
      });
      if (!response.ok) throw new Error(await apiError(response, "Could not include the file"));
      const result = (await response.json()) as SaveResponse;
      setFiles((current) =>
        current.map((entry) =>
          entry.id === localId
            ? { ...entry, durability: "included", includeId: result.include_id ?? undefined }
            : entry,
        ),
      );
    } catch (error) {
      setFiles((current) =>
        current.map((entry) =>
          entry.id === localId
            ? {
                ...entry,
                durability: "error",
                saveError: error instanceof Error ? error.message : "Could not include the file",
              }
            : entry,
        ),
      );
    } finally {
      inclusionInFlight.current.delete(localId);
    }
  }, []);

  const saveFile = useCallback(
    async (localId: string) => {
      const entry = files.find((candidate) => candidate.id === localId);
      if (!entry || entry.durability === "uploading") return;
      if (entry.durableAssetId) {
        if (activeThreadId) {
          await attachSavedFile(localId, entry.durableAssetId, activeThreadId);
        }
        return;
      }
      setFiles((current) =>
        current.map((candidate) =>
          candidate.id === localId
            ? { ...candidate, durability: "uploading", saveError: undefined }
            : candidate,
        ),
      );
      try {
        const query = new URLSearchParams({ filename: entry.file.name });
        if (activeThreadId) query.set("thread_id", activeThreadId);
        const response = await authenticatedFetch(`/api/assets?${query}`, {
          method: "POST",
          headers: { "Content-Type": entry.file.type || "application/octet-stream" },
          body: entry.file,
        });
        if (!response.ok) throw new Error(await apiError(response, "Could not save the file"));
        const result = (await response.json()) as SaveResponse;
        setFiles((current) =>
          current.map((candidate) =>
            candidate.id === localId
              ? {
                  ...candidate,
                  durability: result.include_id ? "included" : "stored",
                  durableAssetId: result.asset_id,
                  includeId: result.include_id ?? undefined,
                  saveError: undefined,
                }
              : candidate,
          ),
        );
      } catch (error) {
        setFiles((current) =>
          current.map((candidate) =>
            candidate.id === localId
              ? {
                  ...candidate,
                  durability: "error",
                  saveError: error instanceof Error ? error.message : "Could not save the file",
                }
              : candidate,
          ),
        );
      }
    },
    [activeThreadId, files, attachSavedFile],
  );

  useEffect(() => {
    if (!activeThreadId) return;
    files.forEach((entry) => {
      if (entry.durability === "stored" && entry.durableAssetId) {
        void attachSavedFile(entry.id, entry.durableAssetId, activeThreadId);
      }
    });
  }, [activeThreadId, files, attachSavedFile]);

  const registerArtifact = useCallback(
    (
      sourceAssetId: string,
      kind: TransientArtifact["kind"],
      label: string,
      blob: Blob,
    ): TransientArtifact => {
      const artifact: TransientArtifact = {
        id: `local_artifact_${crypto.randomUUID()}`,
        sourceAssetId,
        kind,
        label,
        blob,
        previewUrl: blob.type.startsWith("image/") ? URL.createObjectURL(blob) : null,
      };
      setArtifacts((current) => [...current, artifact]);
      return artifact;
    },
    [],
  );

  const value = useMemo(
    () => ({
      files,
      artifacts,
      addFiles,
      removeFile,
      saveFile,
      getFile,
      registerArtifact,
      activeThreadId,
      setActiveThreadId,
    }),
    [files, artifacts, addFiles, removeFile, saveFile, getFile, registerArtifact, activeThreadId],
  );

  return <FileWorkspaceContext.Provider value={value}>{children}</FileWorkspaceContext.Provider>;
}

interface SaveResponse {
  asset_id: string;
  include_id: string | null;
}

async function apiError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : fallback;
  } catch {
    return fallback;
  }
}
