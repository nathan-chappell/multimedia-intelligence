import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { authenticatedFetch } from "../../lib/config";
import { browserId } from "../../lib/browserId";
import {
  classifyLocalFile,
  FileWorkspaceContext,
  type FileCollection,
  type CollectionFileSummary,
  type IncludedLocalFile,
  type ReconciliationSummary,
  type TransientArtifact,
} from "./fileWorkspace";

export function FileWorkspaceProvider({ children }: { children: ReactNode }) {
  const [files, setFiles] = useState<IncludedLocalFile[]>([]);
  const [artifacts, setArtifacts] = useState<TransientArtifact[]>([]);
  const [collections, setCollections] = useState<FileCollection[]>([]);
  const [collectionFiles, setCollectionFiles] = useState<CollectionFileSummary[]>([]);
  const [collectionFilesLoading, setCollectionFilesLoading] = useState(false);
  const [collectionFilesError, setCollectionFilesError] = useState<string | null>(null);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(() => rememberedThreadId());
  const inclusionInFlight = useRef(new Set<string>());
  const activeThreadRef = useRef<string | null>(rememberedThreadId());
  const loadGeneration = useRef(0);
  const artifactsRef = useRef(artifacts);
  const selectedCollectionId = collections.find((collection) => collection.selected)?.id ?? null;

  const loadCollectionFiles = useCallback(async (collectionId: string, threadId: string | null) => {
    setCollectionFilesLoading(true);
    setCollectionFilesError(null);
    try {
      const query = new URLSearchParams({ limit: "100", offset: "0" });
      if (threadId) query.set("thread_id", threadId);
      const response = await authenticatedFetch(
        `/api/collections/${encodeURIComponent(collectionId)}/files?${query}`,
      );
      if (!response.ok) throw new Error(await apiError(response, "Could not load collection files"));
      const page = (await response.json()) as { items: CollectionFileSummary[] };
      setCollectionFiles(page.items);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not load collection files";
      setCollectionFiles([]);
      setCollectionFilesError(message);
      throw error;
    } finally {
      setCollectionFilesLoading(false);
    }
  }, []);

  const loadCollections = useCallback(async () => {
    const response = await authenticatedFetch("/api/collections");
    if (!response.ok) throw new Error(await apiError(response, "Could not load collections"));
    setCollections((await response.json()) as FileCollection[]);
  }, []);

  useEffect(() => {
    void loadCollections().catch(console.error);
  }, [loadCollections]);

  useEffect(() => {
    if (!selectedCollectionId) {
      setCollectionFiles([]);
      return;
    }
    void loadCollectionFiles(selectedCollectionId, activeThreadId).catch(console.error);
  }, [selectedCollectionId, activeThreadId, loadCollectionFiles]);

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
      id: `local_${browserId()}`,
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

  const loadSavedFiles = useCallback(async (
    threadId: string,
    generation: number,
    preserveLocal = false,
  ) => {
    try {
      const query = new URLSearchParams({ thread_id: threadId });
      const [response, derivedResponse] = await Promise.all([
        authenticatedFetch(`/api/assets?${query}`),
        authenticatedFetch(`/api/assets/derived?${query}`),
      ]);
      if (!response.ok) throw new Error(await apiError(response, "Could not load saved files"));
      if (!derivedResponse.ok) {
        throw new Error(await apiError(derivedResponse, "Could not load saved artifacts"));
      }
      const saved = (await response.json()) as SavedAssetResponse[];
      const savedArtifacts = (await derivedResponse.json()) as SavedArtifactResponse[];
      const restoredFiles = await Promise.all(
        saved.map(async (entry): Promise<IncludedLocalFile> => {
          const content = await authenticatedFetch(
            `/api/assets/${encodeURIComponent(entry.asset_id)}/content?${new URLSearchParams({
              thread_id: threadId,
            })}`,
          );
          if (!content.ok) {
            throw new Error(await apiError(content, `Could not restore ${entry.filename}`));
          }
          const blob = await content.blob();
          const file = new File([blob], entry.filename, {
            type: entry.media_type || blob.type || "application/octet-stream",
          });
          return {
            id: entry.asset_id,
            file,
            route: classifyLocalFile(file),
            addedAt: Date.now(),
            durability: "included",
            durableAssetId: entry.asset_id,
            includeId: entry.include_id ?? undefined,
            collectionId: entry.collection_id ?? undefined,
          };
        }),
      );
      const restoredArtifacts = await Promise.all(
        savedArtifacts.map(async (entry): Promise<TransientArtifact> => {
          const content = await authenticatedFetch(
            `/api/assets/derived/${encodeURIComponent(entry.artifact_id)}/content?${query}`,
          );
          if (!content.ok) {
            throw new Error(await apiError(content, `Could not restore ${entry.filename}`));
          }
          const blob = await content.blob();
          return {
            id: entry.artifact_id,
            sourceAssetId: entry.source_asset_id,
            kind: entry.kind === "chart" ? "chart" : "pdf_part",
            label: entry.filename,
            blob,
            previewUrl: blob.type.startsWith("image/") ? URL.createObjectURL(blob) : null,
            durability: "saved",
          };
        }),
      );
      if (loadGeneration.current !== generation || activeThreadRef.current !== threadId) {
        restoredArtifacts.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return;
      }
      setFiles((current) => {
        if (!preserveLocal) return restoredFiles;
        const restoredIds = new Set(restoredFiles.map((entry) => entry.id));
        const local = current.filter(
          (entry) => !entry.durableAssetId && !restoredIds.has(entry.id),
        );
        return [...local, ...restoredFiles];
      });
      setArtifacts((current) => {
        if (!preserveLocal) {
          current.forEach((artifact) => {
            if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
          });
          return restoredArtifacts;
        }
        const restoredIds = new Set(restoredArtifacts.map((artifact) => artifact.id));
        const local = current.filter(
          (artifact) => artifact.durability !== "saved" && !restoredIds.has(artifact.id),
        );
        current
          .filter((artifact) => artifact.durability === "saved" && !restoredIds.has(artifact.id))
          .forEach((artifact) => {
            if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
          });
        return [...local, ...restoredArtifacts];
      });
    } catch (error) {
      if (loadGeneration.current !== generation || activeThreadRef.current !== threadId) return;
      if (!preserveLocal) {
        setFiles([]);
        setArtifacts([]);
      }
      // Keep the conversation usable while making restoration failures visible.
      console.error(error);
    }
  }, []);

  const changeActiveThread = useCallback(
    (threadId: string | null) => {
      const previousThreadId = activeThreadRef.current;
      if (previousThreadId === threadId) return;
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      rememberThreadId(threadId);
      const generation = ++loadGeneration.current;

      setArtifacts((current) => {
        current.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return [];
      });

      if (threadId === null) {
        // Returning to ChatKit's new-thread screen starts a fresh file workspace.
        if (previousThreadId !== null) setFiles([]);
        return;
      }

      // When ChatKit creates a thread from the new-thread screen, preserve files
      // the user just staged so the existing inclusion effect can attach them.
      if (previousThreadId === null && files.length > 0) return;
      setFiles([]);
      void loadSavedFiles(threadId, generation);
    },
    [files.length, loadSavedFiles],
  );

  const restoreThread = useCallback(
    (threadId: string) => {
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      rememberThreadId(threadId);
      const generation = ++loadGeneration.current;
      setFiles([]);
      setArtifacts((current) => {
        current.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return [];
      });
      void loadSavedFiles(threadId, generation);
    },
    [loadSavedFiles],
  );

  const refreshThreadFiles = useCallback(async () => {
    const threadId = activeThreadRef.current;
    if (!threadId) return;
    const generation = ++loadGeneration.current;
    await loadSavedFiles(threadId, generation, true);
  }, [loadSavedFiles]);

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
                  collectionId: result.collection_id ?? undefined,
                  saveError: undefined,
                }
              : candidate,
          ),
        );
        if (result.collection_id) {
          await loadCollectionFiles(result.collection_id, activeThreadRef.current);
        }
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
    [activeThreadId, files, attachSavedFile, loadCollectionFiles],
  );

  useEffect(() => {
    if (!activeThreadId) return;
    files.forEach((entry) => {
      if (entry.durability === "stored" && entry.durableAssetId) {
        void attachSavedFile(entry.id, entry.durableAssetId, activeThreadId);
      }
    });
  }, [activeThreadId, files, attachSavedFile]);

  const selectCollection = useCallback(
    async (collectionId: string) => {
      const response = await authenticatedFetch("/api/collections/selection", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_id: collectionId }),
      });
      if (!response.ok) throw new Error(await apiError(response, "Could not select collection"));
      setCollections((current) =>
        current.map((collection) => ({
          ...collection,
          selected: collection.id === collectionId,
        })),
      );
      if (activeThreadRef.current) {
        const generation = ++loadGeneration.current;
        setFiles([]);
        await loadSavedFiles(activeThreadRef.current, generation);
      } else {
        setFiles([]);
      }
    },
    [loadSavedFiles],
  );

  const setCollectionPublic = useCallback(async (collectionId: string, isPublic: boolean) => {
    const response = await authenticatedFetch(
      `/api/collections/${encodeURIComponent(collectionId)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_public: isPublic }),
      },
    );
    if (!response.ok) {
      throw new Error(await apiError(response, "Could not update collection visibility"));
    }
    const updated = (await response.json()) as FileCollection;
    setCollections((current) =>
      current.map((collection) => (collection.id === updated.id ? updated : collection)),
    );
  }, []);

  const refreshCollectionFiles = useCallback(async () => {
    if (!selectedCollectionId) return;
    await loadCollectionFiles(selectedCollectionId, activeThreadRef.current);
  }, [selectedCollectionId, loadCollectionFiles]);

  const setCollectionFileIncluded = useCallback(
    async (assetId: string, included: boolean) => {
      if (!selectedCollectionId) throw new Error("Select a collection first");
      const selected = collections.find((collection) => collection.id === selectedCollectionId);
      if (selected?.read_only) throw new Error("Public collection files are read-only");
      const threadId = activeThreadRef.current;
      if (!threadId) throw new Error("Start or select a conversation first");
      const response = await authenticatedFetch(
        `/api/collections/${encodeURIComponent(selectedCollectionId)}/files/${encodeURIComponent(assetId)}/inclusion`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ thread_id: threadId, included }),
        },
      );
      if (!response.ok) throw new Error(await apiError(response, "Could not update conversation files"));
      const result = (await response.json()) as { include_id: string | null };
      setCollectionFiles((current) =>
        current.map((file) =>
          file.asset_id === assetId
            ? { ...file, included, include_id: included ? result.include_id : null }
            : file,
        ),
      );
      await refreshThreadFiles();
    },
    [collections, selectedCollectionId, refreshThreadFiles],
  );

  const reconcileCollection = useCallback(async (): Promise<ReconciliationSummary> => {
    if (!selectedCollectionId) throw new Error("Select a collection first");
    const response = await authenticatedFetch(
      `/api/collections/${encodeURIComponent(selectedCollectionId)}/reconcile`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error(await apiError(response, "Could not refresh index status"));
    const result = (await response.json()) as ReconciliationSummary;
    await loadCollectionFiles(selectedCollectionId, activeThreadRef.current);
    return result;
  }, [selectedCollectionId, loadCollectionFiles]);

  const createCollection = useCallback(
    async (name: string, description?: string) => {
      const response = await authenticatedFetch("/api/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description: description || null, select: true }),
      });
      if (!response.ok) throw new Error(await apiError(response, "Could not create collection"));
      await loadCollections();
      if (activeThreadRef.current) {
        const generation = ++loadGeneration.current;
        setFiles([]);
        await loadSavedFiles(activeThreadRef.current, generation);
      } else {
        setFiles([]);
      }
    },
    [loadCollections, loadSavedFiles],
  );

  const registerArtifact = useCallback(
    (
      sourceAssetId: string,
      kind: TransientArtifact["kind"],
      label: string,
      blob: Blob,
    ): TransientArtifact => {
      const artifact: TransientArtifact = {
        id: `local_artifact_${browserId()}`,
        sourceAssetId,
        kind,
        label,
        blob,
        previewUrl: blob.type.startsWith("image/") ? URL.createObjectURL(blob) : null,
        durability: "transient",
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
      collections,
      collectionFiles,
      collectionFilesLoading,
      collectionFilesError,
      selectedCollectionId,
      addFiles,
      removeFile,
      saveFile,
      getFile,
      registerArtifact,
      activeThreadId,
      setActiveThreadId: changeActiveThread,
      restoreThread,
      refreshThreadFiles,
      createCollection,
      selectCollection,
      setCollectionPublic,
      refreshCollectionFiles,
      setCollectionFileIncluded,
      reconcileCollection,
    }),
    [
      files,
      artifacts,
      collections,
      collectionFiles,
      collectionFilesLoading,
      collectionFilesError,
      selectedCollectionId,
      addFiles,
      removeFile,
      saveFile,
      getFile,
      registerArtifact,
      activeThreadId,
      changeActiveThread,
      restoreThread,
      refreshThreadFiles,
      createCollection,
      selectCollection,
      setCollectionPublic,
      refreshCollectionFiles,
      setCollectionFileIncluded,
      reconcileCollection,
    ],
  );

  return <FileWorkspaceContext.Provider value={value}>{children}</FileWorkspaceContext.Provider>;
}

interface SaveResponse {
  asset_id: string;
  include_id: string | null;
  collection_id: string | null;
}

interface SavedAssetResponse extends SaveResponse {
  filename: string;
  media_type: string;
  size_bytes: number;
}

interface SavedArtifactResponse {
  artifact_id: string;
  source_asset_id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  kind: string;
  collection_id: string | null;
}

async function apiError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : fallback;
  } catch {
    return fallback;
  }
}

function rememberedThreadId(): string | null {
  return window.sessionStorage.getItem("mi_active_thread_id") || null;
}

function rememberThreadId(threadId: string | null): void {
  if (threadId) window.sessionStorage.setItem("mi_active_thread_id", threadId);
  else window.sessionStorage.removeItem("mi_active_thread_id");
}
