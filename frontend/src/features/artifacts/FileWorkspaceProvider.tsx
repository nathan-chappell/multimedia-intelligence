import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { authenticatedFetch } from "../../lib/config";
import { browserId } from "../../lib/browserId";
import {
  classifyLocalFile,
  FileWorkspaceContext,
  type FileCollection,
  type CollectionFileSummary,
  type HydratedLocalFile,
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
  const filesRef = useRef<IncludedLocalFile[]>([]);
  const inclusionInFlight = useRef(new Set<string>());
  // Keep hydrated bytes for this page lifetime. The authenticated object store remains
  // authoritative across refreshes, avoiding a second user-scoped persistence layer.
  const contentLoads = useRef(new Map<string, Promise<File>>());
  const hydrationRef = useRef<WorkspaceHydration | null>(null);
  const activeThreadRef = useRef<string | null>(rememberedThreadId());
  const loadGeneration = useRef(0);
  const artifactsRef = useRef(artifacts);
  const selectedCollectionId = collections.find((collection) => collection.selected)?.id ?? null;

  const updateFiles = useCallback(
    (update: (current: IncludedLocalFile[]) => IncludedLocalFile[]) => {
      const next = update(filesRef.current);
      filesRef.current = next;
      setFiles(next);
    },
    [],
  );

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
      filename: file.name,
      mediaType: file.type || "application/octet-stream",
      sizeBytes: file.size,
      route: classifyLocalFile(file),
      addedAt: Date.now(),
      durability: "local",
    }));
    updateFiles((current) => [...current, ...additions]);
  }, [updateFiles]);

  const removeFile = useCallback((assetId: string) => {
    updateFiles((current) => current.filter((entry) => entry.id !== assetId));
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
  }, [updateFiles]);

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
      updateFiles((current) =>
        current.map((entry) =>
          entry.id === localId
            ? { ...entry, durability: "included", includeId: result.include_id ?? undefined }
            : entry,
        ),
      );
    } catch (error) {
      updateFiles((current) =>
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
  }, [updateFiles]);

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
      const saved = (await response.json()) as SavedAssetResponse[];
      const restoredFiles = saved.map(
        (entry): IncludedLocalFile => {
          const mediaType = entry.media_type || "application/octet-stream";
          return {
            id: entry.asset_id,
            filename: entry.filename,
            mediaType,
            sizeBytes: entry.size_bytes,
            route: classifyLocalFile(new File([], entry.filename, { type: mediaType })),
            addedAt: Date.now(),
            durability: "included",
            durableAssetId: entry.asset_id,
            includeId: entry.include_id ?? undefined,
            collectionId: entry.collection_id ?? undefined,
          };
        },
      );
      if (loadGeneration.current !== generation || activeThreadRef.current !== threadId) return;
      updateFiles((current) => {
        const hydratedByAssetId = new Map(
          current.flatMap((entry) =>
            entry.durableAssetId && entry.file
              ? [[entry.durableAssetId, entry.file] as const]
              : [],
          ),
        );
        const hydratedFiles = restoredFiles.map((entry) => {
          const file = entry.durableAssetId
            ? hydratedByAssetId.get(entry.durableAssetId)
            : undefined;
          return file ? { ...entry, file } : entry;
        });
        if (!preserveLocal) return hydratedFiles;
        const restoredIds = new Set(hydratedFiles.map((entry) => entry.id));
        const local = current.filter(
          (entry) => !entry.durableAssetId && !restoredIds.has(entry.id),
        );
        return [...local, ...hydratedFiles];
      });

      if (!derivedResponse.ok) {
        console.error(await apiError(derivedResponse, "Could not load saved artifacts"));
        return;
      }
      const savedArtifacts = (await derivedResponse.json()) as SavedArtifactResponse[];
      let restoredArtifacts: TransientArtifact[];
      try {
        restoredArtifacts = await Promise.all(
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
      } catch (error) {
        console.error(error);
        return;
      }
      if (loadGeneration.current !== generation || activeThreadRef.current !== threadId) {
        restoredArtifacts.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return;
      }
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
        updateFiles(() => []);
        setArtifacts([]);
      }
      throw error;
    }
  }, [updateFiles]);

  const startHydration = useCallback(
    (threadId: string, preserveLocal: boolean, force = false): Promise<void> => {
      const current = hydrationRef.current;
      if (!force && current?.threadId === threadId) return current.promise;
      const generation = ++loadGeneration.current;
      const hydration: WorkspaceHydration = {
        threadId,
        generation,
        error: null,
        promise: Promise.resolve(),
      };
      hydration.promise = loadSavedFiles(threadId, generation, preserveLocal).catch(
        (error: unknown) => {
          hydration.error = asError(error, "Could not hydrate conversation files");
          console.error(error);
        },
      );
      hydrationRef.current = hydration;
      return hydration.promise;
    },
    [loadSavedFiles],
  );

  const waitUntilReady = useCallback(async () => {
    while (true) {
      const threadId = activeThreadRef.current;
      if (!threadId) return;
      let hydration = hydrationRef.current;
      if (!hydration || hydration.threadId !== threadId) {
        void startHydration(threadId, true);
        hydration = hydrationRef.current;
      }
      if (!hydration) continue;
      await hydration.promise;
      if (activeThreadRef.current !== threadId) {
        throw new Error("The active conversation changed while restoring its files");
      }
      if (hydrationRef.current !== hydration) continue;
      if (hydration.error) throw hydration.error;
      return;
    }
  }, [startHydration]);

  const getFiles = useCallback((): readonly IncludedLocalFile[] => filesRef.current, []);

  const resolveFile = useCallback(
    async (assetId: string): Promise<HydratedLocalFile | undefined> => {
      await waitUntilReady();
      const entry = filesRef.current.find(
        (candidate) => candidate.id === assetId || candidate.durableAssetId === assetId,
      );
      if (!entry) return undefined;
      if (entry.file) return { ...entry, file: entry.file };
      const threadId = activeThreadRef.current;
      const durableAssetId = entry.durableAssetId;
      if (!threadId || !durableAssetId) return undefined;
      const cacheKey = `${threadId}:${durableAssetId}`;
      let content = contentLoads.current.get(cacheKey);
      if (!content) {
        content = loadAssetFile(durableAssetId, threadId, entry.filename, entry.mediaType);
        contentLoads.current.set(cacheKey, content);
        void content.then(
          () => contentLoads.current.delete(cacheKey),
          () => contentLoads.current.delete(cacheKey),
        );
      }
      const file = await content;
      if (activeThreadRef.current !== threadId) {
        throw new Error("The active conversation changed while restoring the file");
      }
      const hydrated: HydratedLocalFile = { ...entry, file };
      updateFiles((current) =>
        current.map((candidate) =>
          candidate.id === entry.id ? { ...candidate, file } : candidate,
        ),
      );
      return hydrated;
    },
    [updateFiles, waitUntilReady],
  );

  const changeActiveThread = useCallback(
    (threadId: string | null) => {
      const previousThreadId = activeThreadRef.current;
      if (previousThreadId === threadId) return;
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      rememberThreadId(threadId);
      hydrationRef.current = null;

      setArtifacts((current) => {
        current.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return [];
      });

      if (threadId === null) {
        // Returning to ChatKit's new-thread screen starts a fresh file workspace.
        if (previousThreadId !== null) updateFiles(() => []);
        return;
      }

      // When ChatKit creates a thread from the new-thread screen, preserve files
      // the user just staged so the existing inclusion effect can attach them.
      if (previousThreadId === null && filesRef.current.length > 0) return;
      updateFiles(() => []);
      void startHydration(threadId, false, true);
    },
    [startHydration, updateFiles],
  );

  const restoreThread = useCallback(
    (threadId: string) => {
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      rememberThreadId(threadId);
      if (
        hydrationRef.current?.threadId === threadId &&
        hydrationRef.current.error === null
      ) return;
      updateFiles(() => []);
      setArtifacts((current) => {
        current.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return [];
      });
      void startHydration(threadId, false, true);
    },
    [startHydration, updateFiles],
  );

  const refreshThreadFiles = useCallback(async () => {
    const threadId = activeThreadRef.current;
    if (!threadId) return;
    await startHydration(threadId, true, true);
  }, [startHydration]);

  const saveFile = useCallback(
    async (localId: string) => {
      const entry = filesRef.current.find((candidate) => candidate.id === localId);
      if (!entry || entry.durability === "uploading") return;
      if (entry.durableAssetId) {
        if (activeThreadId) {
          await attachSavedFile(localId, entry.durableAssetId, activeThreadId);
        }
        return;
      }
      updateFiles((current) =>
        current.map((candidate) =>
          candidate.id === localId
            ? { ...candidate, durability: "uploading", saveError: undefined }
            : candidate,
        ),
      );
      try {
        if (!entry.file) throw new Error("File content is unavailable for upload");
        const query = new URLSearchParams({ filename: entry.file.name });
        if (activeThreadId) query.set("thread_id", activeThreadId);
        const response = await authenticatedFetch(`/api/assets?${query}`, {
          method: "POST",
          headers: { "Content-Type": entry.file.type || "application/octet-stream" },
          body: entry.file,
        });
        if (!response.ok) throw new Error(await apiError(response, "Could not save the file"));
        const result = (await response.json()) as SaveResponse;
        updateFiles((current) =>
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
        updateFiles((current) =>
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
    [activeThreadId, attachSavedFile, loadCollectionFiles, updateFiles],
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
        updateFiles(() => []);
        await startHydration(activeThreadRef.current, false, true);
      } else {
        updateFiles(() => []);
      }
    },
    [startHydration, updateFiles],
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
        updateFiles(() => []);
        await startHydration(activeThreadRef.current, false, true);
      } else {
        updateFiles(() => []);
      }
    },
    [loadCollections, startHydration, updateFiles],
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
      getFiles,
      resolveFile,
      waitUntilReady,
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
      getFiles,
      resolveFile,
      waitUntilReady,
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

interface WorkspaceHydration {
  threadId: string;
  generation: number;
  promise: Promise<void>;
  error: Error | null;
}

async function loadAssetFile(
  assetId: string,
  threadId: string,
  filename: string,
  mediaType: string,
): Promise<File> {
  const content = await authenticatedFetch(
    `/api/assets/${encodeURIComponent(assetId)}/content?${new URLSearchParams({
      thread_id: threadId,
    })}`,
  );
  if (!content.ok) {
    throw new Error(await apiError(content, `Could not restore ${filename}`));
  }
  const blob = await content.blob();
  return new File([blob], filename, {
    type: mediaType || blob.type || "application/octet-stream",
  });
}

function asError(error: unknown, fallback: string): Error {
  return error instanceof Error ? error : new Error(fallback);
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
