import { useCallback, useRef, useState, type FormEvent } from "react";
import { ChatKit, useChatKit } from "@openai/chatkit-react";

import { executeFileClientTool } from "../artifacts/clientToolHandler";
import { useFileWorkspace } from "../artifacts/useFileWorkspace";
import { authenticatedFetch, config } from "../../lib/config";

export function ChatPanel() {
  const fileWorkspace = useFileWorkspace();
  const activeThreadRef = useRef<string | null>(null);
  const titleLoadGeneration = useRef(0);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [savedTitle, setSavedTitle] = useState("");
  const [draftTitle, setDraftTitle] = useState("");
  const [titleBusy, setTitleBusy] = useState(false);

  const loadTitle = useCallback(async (threadId: string, generation: number) => {
    try {
      const response = await authenticatedFetch(
        `/api/threads/${encodeURIComponent(threadId)}/title`,
      );
      if (!response.ok) return;
      const result = (await response.json()) as ThreadTitleResponse;
      if (titleLoadGeneration.current !== generation || activeThreadRef.current !== threadId) return;
      const title = result.title ?? "";
      setSavedTitle(title);
      setDraftTitle(title);
    } catch {
      // ChatKit remains usable when the optional title control cannot be loaded.
    }
  }, []);

  const chatkit = useChatKit({
    initialThread: window.sessionStorage.getItem("mi_active_thread_id") || null,
    api: {
      url: config.chatkitUrl,
      domainKey: config.domainKey,
      fetch: authenticatedFetch,
    },
    onClientTool: async (toolCall) => {
      return executeFileClientTool(fileWorkspace, toolCall);
    },
    onThreadChange: ({ threadId }) => {
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      setSavedTitle("");
      setDraftTitle("");
      const generation = ++titleLoadGeneration.current;
      if (threadId) void loadTitle(threadId, generation);
      fileWorkspace.setActiveThreadId(threadId);
    },
    onThreadLoadStart: ({ threadId }) => {
      fileWorkspace.restoreThread(threadId);
    },
    onResponseEnd: () => {
      void fileWorkspace.refreshThreadFiles();
    },
    history: {
      enabled: true,
      showDelete: true,
      showRename: true,
    },
    header: { title: { enabled: false } },
    threadItemActions: { feedback: true },
    composer: {
      placeholder: "Ask about this conversation's files…",
      models: [
        {
          id: "gpt-5.6-luna",
          label: "GPT-5.6 Luna",
          description: "Fast, cost-sensitive workloads",
          default: true,
        },
        {
          id: "gpt-5.6",
          label: "GPT-5.6",
          description: "Frontier model with medium reasoning",
        },
        {
          id: "gpt-5.6-terra",
          label: "GPT-5.6 Terra",
          description: "Balanced capability and cost",
        },
      ],
      dictation: { enabled: true },
      // ChatKit attachments stay disabled. All files use our conversation-scoped
      // asset and ingestion pipeline instead.
      attachments: { enabled: false },
    },
  });

  async function saveTitle(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const threadId = activeThreadRef.current;
    const title = draftTitle.trim();
    if (!threadId || !title || title === savedTitle || titleBusy) return;
    setTitleBusy(true);
    try {
      const response = await authenticatedFetch(
        `/api/threads/${encodeURIComponent(threadId)}/title`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title }),
        },
      );
      if (!response.ok) throw new Error(await responseError(response, "Could not save title"));
      const result = (await response.json()) as ThreadTitleResponse;
      setSavedTitle(result.title ?? "");
      setDraftTitle(result.title ?? "");
      await chatkit.fetchUpdates();
      await notifyChatKit("Conversation title saved");
    } catch (error) {
      await notifyChatKit(error instanceof Error ? error.message : "Could not save title", "danger");
    } finally {
      setTitleBusy(false);
    }
  }

  async function suggestTitle() {
    const threadId = activeThreadRef.current;
    if (!threadId || titleBusy) return;
    setTitleBusy(true);
    try {
      const response = await authenticatedFetch(
        `/api/threads/${encodeURIComponent(threadId)}/title/suggest`,
        { method: "POST" },
      );
      if (!response.ok) throw new Error(await responseError(response, "Could not suggest title"));
      const result = (await response.json()) as ThreadTitleResponse;
      const title = result.title ?? "";
      setSavedTitle(title);
      setDraftTitle(title);
      await chatkit.fetchUpdates();
      await notifyChatKit("Conversation title updated");
    } catch (error) {
      await notifyChatKit(
        error instanceof Error ? error.message : "Could not suggest title",
        "danger",
      );
    } finally {
      setTitleBusy(false);
    }
  }

  async function notifyChatKit(message: string, level: "info" | "warning" | "danger" = "info") {
    try {
      await chatkit.sendCustomAction({ type: "app.notice", payload: { message, level } });
    } catch {
      // The underlying action is optional feedback; the primary operation already completed.
    }
  }

  return (
    <article className="panel chat-panel">
      <div className="panel-heading">
        <div className="thread-title-block">
          <span className="eyebrow">Conversation</span>
          {activeThreadId ? (
            <form className="thread-title-form" onSubmit={(event) => void saveTitle(event)}>
              <input
                aria-label="Conversation title"
                maxLength={80}
                value={draftTitle}
                placeholder="Untitled conversation"
                disabled={titleBusy}
                onChange={(event) => setDraftTitle(event.currentTarget.value)}
              />
              {draftTitle.trim() && draftTitle.trim() !== savedTitle && (
                <button type="submit" aria-label="Save conversation title" title="Save title">
                  ✓
                </button>
              )}
            </form>
          ) : (
            <h2>New conversation</h2>
          )}
        </div>
        <button
          className="suggest-title-button"
          type="button"
          disabled={!activeThreadId || titleBusy}
          aria-label="Suggest conversation title"
          title="Suggest a title with GPT-5.6 Luna"
          onClick={() => void suggestTitle()}
        >
          ✦
        </button>
      </div>
      <ChatKit control={chatkit.control} className="chatkit" />
    </article>
  );
}

interface ThreadTitleResponse {
  thread_id: string;
  title: string | null;
}

async function responseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : fallback;
  } catch {
    return fallback;
  }
}
