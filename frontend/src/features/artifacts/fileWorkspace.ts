import { createContext } from "react";

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
  file: File;
  route: FileRoute;
  addedAt: number;
  durability: "local" | "uploading" | "stored" | "included" | "error";
  durableAssetId?: string;
  includeId?: string;
  saveError?: string;
}

export interface TransientArtifact {
  id: string;
  sourceAssetId: string;
  kind: "pdf_page_image" | "pdf_part";
  label: string;
  blob: Blob;
  previewUrl: string | null;
}

export interface FileWorkspaceValue {
  files: IncludedLocalFile[];
  artifacts: TransientArtifact[];
  addFiles: (files: FileList | readonly File[]) => void;
  removeFile: (assetId: string) => void;
  saveFile: (assetId: string) => Promise<void>;
  getFile: (assetId: string) => IncludedLocalFile | undefined;
  activeThreadId: string | null;
  setActiveThreadId: (threadId: string | null) => void;
  registerArtifact: (
    sourceAssetId: string,
    kind: TransientArtifact["kind"],
    label: string,
    blob: Blob,
  ) => TransientArtifact;
}

export const FileWorkspaceContext = createContext<FileWorkspaceValue | null>(null);

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
