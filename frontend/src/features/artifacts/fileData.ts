import { createContext } from "react";

/** Browser and durable file routes shared by collection and workspace views. */
export type FileRoute =
  | "text"
  | "json"
  | "csv"
  | "pdf"
  | "image"
  | "audio"
  | "video"
  | "unsupported";

export interface IncludedLocalFile {
  id: string;
  file?: File;
  filename: string;
  mediaType: string;
  sizeBytes: number;
  route: FileRoute;
  addedAt: number;
  durability: "local" | "uploading" | "stored" | "included" | "error";
  durableAssetId?: string;
  includeId?: string;
  saveError?: string;
  collectionId?: string;
}

export interface HydratedLocalFile extends IncludedLocalFile {
  file: File;
}

export interface FileCollection {
  id: string;
  name: string;
  description: string | null;
  selected: boolean;
  is_public: boolean;
  owned: boolean;
  can_manage: boolean;
  read_only: boolean;
}

export interface CollectionFileSummary {
  asset_id: string;
  filename: string;
  media_type: string;
  route: FileRoute;
  size_bytes: number;
  created_at: string;
  collection_id: string;
  ingestion_status: string;
  provider_status: "not_indexed" | "pending" | "ready" | "missing" | "error";
  artifact_count: number;
  provider_file_count: number;
  included: boolean;
  include_id: string | null;
  last_error: string | null;
}

export interface ReconciliationSummary {
  ready: number;
  pending: number;
  missing: number;
  failed: number;
  orphaned: number;
  checked_at: string;
  provider_error: string | null;
}

export interface TransientArtifact {
  id: string;
  sourceAssetId: string;
  kind: "pdf_page_image" | "pdf_part" | "chart";
  label: string;
  blob: Blob;
  previewUrl: string | null;
  durability?: "transient" | "saved";
}

export interface ConversationWorkspaceValue {
  files: IncludedLocalFile[];
  artifacts: TransientArtifact[];
  addFiles: (files: FileList | readonly File[]) => void;
  removeFile: (assetId: string) => void;
  saveFile: (assetId: string) => Promise<void>;
  getFiles: () => readonly IncludedLocalFile[];
  resolveFile: (assetId: string) => Promise<HydratedLocalFile | undefined>;
  waitUntilReady: () => Promise<void>;
  activeThreadId: string | null;
  setActiveThreadId: (threadId: string | null) => void;
  restoreThread: (threadId: string) => void;
  refreshThreadFiles: () => Promise<void>;
  setFileIncluded: (assetId: string, included: boolean) => Promise<void>;
  registerArtifact: (
    sourceAssetId: string,
    kind: TransientArtifact["kind"],
    label: string,
    blob: Blob,
  ) => TransientArtifact;
}

export interface CollectionLibraryValue {
  collections: FileCollection[];
  collectionFiles: CollectionFileSummary[];
  collectionFilesLoading: boolean;
  collectionFilesError: string | null;
  selectedCollectionId: string | null;
  createCollection: (name: string, description?: string) => Promise<void>;
  selectCollection: (collectionId: string) => Promise<void>;
  setCollectionPublic: (collectionId: string, isPublic: boolean) => Promise<void>;
  refreshCollectionFiles: () => Promise<void>;
  setCollectionFileIncluded: (assetId: string, included: boolean) => Promise<void>;
  reconcileCollection: () => Promise<ReconciliationSummary>;
}

export const ConversationWorkspaceContext =
  createContext<ConversationWorkspaceValue | null>(null);
export const CollectionLibraryContext = createContext<CollectionLibraryValue | null>(null);

const textExtensions = new Set(["md", "txt"]);
const imageExtensions = new Set(["png", "jpeg", "jpg", "webp", "gif"]);
const audioExtensions = new Set(["flac", "mp3", "mpga", "m4a", "ogg", "wav"]);
const videoExtensions = new Set(["mp4", "mpeg", "webm"]);

export function classifyLocalFile(file: File): FileRoute {
  const extension = file.name.split(".").at(-1)?.toLocaleLowerCase() ?? "";
  if (textExtensions.has(extension)) return "text";
  if (extension === "json") return "json";
  if (extension === "csv") return "csv";
  if (extension === "pdf") return "pdf";
  if (imageExtensions.has(extension)) return "image";
  if (audioExtensions.has(extension)) return "audio";
  if (videoExtensions.has(extension)) return "video";
  return "unsupported";
}
