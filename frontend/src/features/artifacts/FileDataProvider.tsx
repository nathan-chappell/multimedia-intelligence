import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { authenticatedFetch } from "../../lib/config";
import { browserId } from "../../lib/browserId";
import {
  classifyLocalFile,
  CollectionLibraryContext,
  ConversationWorkspaceContext,
  type FileCollection,
  type CollectionFileSummary,
  type HydratedLocalFile,
  type IncludedLocalFile,
  type ReconciliationSummary,
  type TransientArtifact,
} from "./fileData";

const COLLECTION_FILES_PAGE_SIZE = 10;

/** Coordinate collection search with the user's single durable workspace. */
export function FileDataProvider({
  children,
  collectionsEnabled = true,
}: {
  children: ReactNode;
  collectionsEnabled?: boolean;
}) {
  const [files, setFiles] = useState<IncludedLocalFile[]>([]);
  const [artifacts, setArtifacts] = useState<TransientArtifact[]>([]);
  const [collections, setCollections] = useState<FileCollection[]>([]);
  const [focusedCollectionId, setFocusedCollectionId] = useState<string | null>(null);
  const [collectionFiles, setCollectionFiles] = useState<CollectionFileSummary[]>([]);
  const [collectionFilesTotal, setCollectionFilesTotal] = useState(0);
  const [collectionFilesPage, setCollectionFilesPage] = useState(1);
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

  const updateFiles = useCallback(
    (update: (current: IncludedLocalFile[]) => IncludedLocalFile[]) => {
      const next = update(filesRef.current);
      filesRef.current = next;
      setFiles(next);
    },
    [],
  );

  const loadCollectionFiles = useCallback(async (collectionId: string, pageNumber: number) => {
    setCollectionFilesLoading(true);
    setCollectionFilesError(null);
    try {
      const query = new URLSearchParams({
        limit: String(COLLECTION_FILES_PAGE_SIZE),
        offset: String((pageNumber - 1) * COLLECTION_FILES_PAGE_SIZE),
      });
      const response = await authenticatedFetch(
        `/api/collections/${encodeURIComponent(collectionId)}/files?${query}`,
      );
      if (!response.ok) throw new Error(await apiError(response, "Could not load collection files"));
      const page = (await response.json()) as {
        items: CollectionFileSummary[];
        total: number;
      };
      const lastPage = Math.max(1, Math.ceil(page.total / COLLECTION_FILES_PAGE_SIZE));
      if (pageNumber > lastPage) {
        setCollectionFilesPage(lastPage);
        return;
      }
      setCollectionFiles(page.items);
      setCollectionFilesTotal(page.total);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not load collection files";
      setCollectionFiles([]);
      setCollectionFilesTotal(0);
      setCollectionFilesError(message);
      throw error;
    } finally {
      setCollectionFilesLoading(false);
    }
  }, []);

  const loadCollections = useCallback(async () => {
    const response = await authenticatedFetch("/api/collections");
    if (!response.ok) throw new Error(await apiError(response, "Could not load collections"));
    const loaded = (await response.json()) as FileCollection[];
    setCollections(loaded);
    setFocusedCollectionId((current) =>
      current && loaded.some((collection) => collection.id === current)
        ? current
        : (loaded[0]?.id ?? null),
    );
  }, []);

  useEffect(() => {
    if (!collectionsEnabled) {
      setCollections([]);
      setFocusedCollectionId(null);
      setCollectionFiles([]);
      setCollectionFilesTotal(0);
      return;
    }
    void loadCollections().catch(console.error);
  }, [collectionsEnabled, loadCollections]);

  useEffect(() => {
    if (!collectionsEnabled || !focusedCollectionId) {
      setCollectionFiles([]);
      setCollectionFilesTotal(0);
      return;
    }
    void loadCollectionFiles(focusedCollectionId, collectionFilesPage).catch(console.error);
  }, [collectionsEnabled, focusedCollectionId, collectionFilesPage, loadCollectionFiles]);

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
    updateFiles((current) => [...additions, ...current]);
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

  const attachSavedFile = useCallback(async (localId: string, assetId: string) => {
    if (inclusionInFlight.current.has(localId)) return;
    inclusionInFlight.current.add(localId);
    try {
      const response = await authenticatedFetch(`/api/assets/${encodeURIComponent(assetId)}/workspace`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
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

  const loadSavedFiles = useCallback(async (generation: number, preserveLocal = false) => {
    try {
      const response = await authenticatedFetch("/api/assets");
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
            sourceFileId: entry.source_asset_id ?? undefined,
          };
        },
      ).reverse();
      if (loadGeneration.current !== generation) return;
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

    } catch (error) {
      if (loadGeneration.current !== generation) return;
      if (!preserveLocal) updateFiles(() => []);
      throw error;
    }
  }, [updateFiles]);

  const startHydration = useCallback(
    (preserveLocal: boolean, force = false): Promise<void> => {
      const current = hydrationRef.current;
      if (!force && current) return current.promise;
      const generation = ++loadGeneration.current;
      const hydration: WorkspaceHydration = {
        generation,
        error: null,
        promise: Promise.resolve(),
      };
      hydration.promise = loadSavedFiles(generation, preserveLocal).catch(
        (error: unknown) => {
          hydration.error = asError(error, "Could not hydrate workspace files");
          console.error(error);
        },
      );
      hydrationRef.current = hydration;
      return hydration.promise;
    },
    [loadSavedFiles],
  );

  useEffect(() => {
    void startHydration(false);
  }, [startHydration]);

  const waitUntilReady = useCallback(async () => {
    while (true) {
      let hydration = hydrationRef.current;
      if (!hydration) {
        void startHydration(true);
        hydration = hydrationRef.current;
      }
      if (!hydration) continue;
      await hydration.promise;
      if (hydrationRef.current !== hydration) continue;
      if (hydration.error) throw hydration.error;
      return;
    }
  }, [startHydration]);

  const getFiles = useCallback((): readonly IncludedLocalFile[] => filesRef.current, []);

  const resolveFile = useCallback(
    async (assetId: string): Promise<HydratedLocalFile | undefined> => {
      await waitUntilReady();
      let entry = filesRef.current.find(
        (candidate) => candidate.id === assetId || candidate.durableAssetId === assetId,
      );
      if (!entry) {
        const response = await authenticatedFetch(
          `/api/assets/${encodeURIComponent(assetId)}/workspace`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
          },
        );
        if (!response.ok) return undefined;
        const loaded = (await response.json()) as SavedAssetResponse;
        const mediaType = loaded.media_type || "application/octet-stream";
        entry = {
          id: loaded.asset_id,
          filename: loaded.filename,
          mediaType,
          sizeBytes: loaded.size_bytes,
          route: classifyLocalFile(new File([], loaded.filename, { type: mediaType })),
          addedAt: Date.now(),
          durability: "included",
          durableAssetId: loaded.asset_id,
          includeId: loaded.include_id ?? undefined,
          collectionId: loaded.collection_id ?? undefined,
          sourceFileId: loaded.source_asset_id ?? undefined,
        };
        const loadedEntry = entry;
        updateFiles((current) =>
          current.some((candidate) => candidate.durableAssetId === loadedEntry.durableAssetId)
            ? current
            : [loadedEntry, ...current],
        );
      }
      if (entry.file) return { ...entry, file: entry.file };
      const durableAssetId = entry.durableAssetId;
      if (!durableAssetId) return undefined;
      const cacheKey = durableAssetId;
      let content = contentLoads.current.get(cacheKey);
      if (!content) {
        content = loadAssetFile(durableAssetId, entry.filename, entry.mediaType);
        contentLoads.current.set(cacheKey, content);
        void content.then(
          () => contentLoads.current.delete(cacheKey),
          () => contentLoads.current.delete(cacheKey),
        );
      }
      const file = await content;
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
      if (activeThreadRef.current === threadId) return;
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      rememberThreadId(threadId);
      setArtifacts((current) => {
        current.forEach((artifact) => {
          if (artifact.previewUrl) URL.revokeObjectURL(artifact.previewUrl);
        });
        return [];
      });
    },
    [],
  );

  const restoreThread = useCallback(
    (threadId: string) => {
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      rememberThreadId(threadId);
    },
    [],
  );

  const refreshThreadFiles = useCallback(async () => {
    await startHydration(true, true);
  }, [startHydration]);

  const saveFile = useCallback(
    async (localId: string) => {
      const entry = filesRef.current.find((candidate) => candidate.id === localId);
      if (!entry || entry.durability === "uploading") return;
      if (entry.durableAssetId) {
        await attachSavedFile(localId, entry.durableAssetId);
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
        if (collectionsEnabled && result.collection_id && result.collection_id === focusedCollectionId) {
          await loadCollectionFiles(result.collection_id, collectionFilesPage);
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
    [
      attachSavedFile,
      collectionFilesPage,
      collectionsEnabled,
      focusedCollectionId,
      loadCollectionFiles,
      updateFiles,
    ],
  );

  useEffect(() => {
    files.forEach((entry) => {
      if (entry.durability === "local") {
        void saveFile(entry.id);
      } else if (entry.durability === "stored" && entry.durableAssetId) {
        void attachSavedFile(entry.id, entry.durableAssetId);
      }
    });
  }, [files, attachSavedFile, saveFile]);

  const selectCollection = useCallback((collectionId: string) => {
    setCollectionFilesPage(1);
    setFocusedCollectionId(collectionId);
  }, []);

  const selectCollectionFilesPage = useCallback((page: number) => {
    setCollectionFilesPage(Math.max(1, page));
  }, []);

  const refreshCollectionFiles = useCallback(async () => {
    if (!focusedCollectionId) return;
    await loadCollectionFiles(focusedCollectionId, collectionFilesPage);
  }, [collectionFilesPage, focusedCollectionId, loadCollectionFiles]);

  const setFileIncluded = useCallback(
    async (assetId: string, included: boolean) => {
      const response = await authenticatedFetch(
        `/api/assets/${encodeURIComponent(assetId)}/inclusion`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ included }),
        },
      );
      if (!response.ok) {
        throw new Error(await apiError(response, "Could not update the workspace"));
      }
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
    [refreshThreadFiles],
  );

  const setCollectionFileIncluded = useCallback(
    async (assetId: string, included: boolean) => {
      if (!focusedCollectionId) throw new Error("Select a collection first");
      await setFileIncluded(assetId, included);
    },
    [focusedCollectionId, setFileIncluded],
  );

  const reconcileCollection = useCallback(async (): Promise<ReconciliationSummary> => {
    if (!focusedCollectionId) throw new Error("Select a collection first");
    const response = await authenticatedFetch(
      `/api/collections/${encodeURIComponent(focusedCollectionId)}/reconcile`,
      { method: "POST" },
    );
    if (!response.ok) throw new Error(await apiError(response, "Could not refresh index status"));
    const result = (await response.json()) as ReconciliationSummary;
    await loadCollectionFiles(focusedCollectionId, collectionFilesPage);
    return result;
  }, [collectionFilesPage, focusedCollectionId, loadCollectionFiles]);

  const createCollection = useCallback(
    async (name: string, description?: string) => {
      const response = await authenticatedFetch("/api/collections", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description: description || null }),
      });
      if (!response.ok) throw new Error(await apiError(response, "Could not create collection"));
      await loadCollections();
    },
    [loadCollections],
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

  const workspaceValue = useMemo(
    () => ({
      files,
      artifacts,
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
      setFileIncluded,
    }),
    [
      files,
      artifacts,
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
      setFileIncluded,
    ],
  );

  const libraryValue = useMemo(
    () => ({
      collections,
      collectionFiles,
      collectionFilesTotal,
      collectionFilesPage,
      collectionFilesPageSize: COLLECTION_FILES_PAGE_SIZE,
      collectionFilesLoading,
      collectionFilesError,
      focusedCollectionId,
      createCollection,
      selectCollection,
      selectCollectionFilesPage,
      refreshCollectionFiles,
      setCollectionFileIncluded,
      reconcileCollection,
    }),
    [
      collections,
      collectionFiles,
      collectionFilesTotal,
      collectionFilesPage,
      collectionFilesLoading,
      collectionFilesError,
      focusedCollectionId,
      createCollection,
      selectCollection,
      selectCollectionFilesPage,
      refreshCollectionFiles,
      setCollectionFileIncluded,
      reconcileCollection,
    ],
  );

  return (
    <CollectionLibraryContext.Provider value={libraryValue}>
      <ConversationWorkspaceContext.Provider value={workspaceValue}>
        {children}
      </ConversationWorkspaceContext.Provider>
    </CollectionLibraryContext.Provider>
  );
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
  source_asset_id: string | null;
}

interface WorkspaceHydration {
  generation: number;
  promise: Promise<void>;
  error: Error | null;
}

async function loadAssetFile(
  assetId: string,
  filename: string,
  mediaType: string,
): Promise<File> {
  const content = await authenticatedFetch(`/api/assets/${encodeURIComponent(assetId)}/content`);
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
