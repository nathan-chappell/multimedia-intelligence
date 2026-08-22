import { useContext } from "react";

import { FileWorkspaceContext } from "./fileWorkspace";

export function useFileWorkspace() {
  const workspace = useContext(FileWorkspaceContext);
  if (!workspace) {
    throw new Error("useFileWorkspace must be used inside FileWorkspaceProvider");
  }
  return workspace;
}
