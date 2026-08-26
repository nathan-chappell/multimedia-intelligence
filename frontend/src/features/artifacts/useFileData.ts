import { useContext } from "react";

import {
  CollectionLibraryContext,
  ConversationWorkspaceContext,
} from "./fileData";

export function useConversationWorkspace() {
  const workspace = useContext(ConversationWorkspaceContext);
  if (!workspace) {
    throw new Error("useConversationWorkspace must be used inside FileDataProvider");
  }
  return workspace;
}

export function useCollectionLibrary() {
  const library = useContext(CollectionLibraryContext);
  if (!library) {
    throw new Error("useCollectionLibrary must be used inside FileDataProvider");
  }
  return library;
}
