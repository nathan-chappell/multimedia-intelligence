import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {
  classifyLocalFile,
  FileWorkspaceContext,
  type IncludedLocalFile,
  type TransientArtifact,
} from "./fileWorkspace";

export function FileWorkspaceProvider({ children }: { children: ReactNode }) {
  const [files, setFiles] = useState<IncludedLocalFile[]>([]);
  const [artifacts, setArtifacts] = useState<TransientArtifact[]>([]);
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
    () => ({ files, artifacts, addFiles, removeFile, getFile, registerArtifact }),
    [files, artifacts, addFiles, removeFile, getFile, registerArtifact],
  );

  return <FileWorkspaceContext.Provider value={value}>{children}</FileWorkspaceContext.Provider>;
}
