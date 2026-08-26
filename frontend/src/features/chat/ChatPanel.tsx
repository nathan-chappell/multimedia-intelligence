import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { ChatKit, useChatKit } from "@openai/chatkit-react";

import { executeFileClientTool } from "../artifacts/clientToolHandler";
import { useConversationWorkspace } from "../artifacts/useFileData";
import { authenticatedFetch, config } from "../../lib/config";

export function ChatPanel() {
  const conversationWorkspace = useConversationWorkspace();
  const activeThreadRef = useRef<string | null>(null);
  const titleLoadGeneration = useRef(0);
  const toastTimer = useRef<number | undefined>(undefined);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [savedTitle, setSavedTitle] = useState("");
  const [draftTitle, setDraftTitle] = useState("");
  const [titleBusy, setTitleBusy] = useState(false);
  const [toast, setToast] = useState<AppToast>();

  const dismissToast = useCallback(() => {
    if (toastTimer.current !== undefined) window.clearTimeout(toastTimer.current);
    toastTimer.current = undefined;
    setToast(undefined);
  }, []);

  const showToast = useCallback((nextToast: AppToast) => {
    if (toastTimer.current !== undefined) window.clearTimeout(toastTimer.current);
    setToast(nextToast);
    toastTimer.current = window.setTimeout(() => {
      setToast(undefined);
      toastTimer.current = undefined;
    }, nextToast.level === "danger" ? 10_000 : 6_000);
  }, []);

  useEffect(() => () => {
    if (toastTimer.current !== undefined) window.clearTimeout(toastTimer.current);
  }, []);

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
      domainKey: config.domainKey ?? "",
      fetch: authenticatedFetch,
    },
    onClientTool: async (toolCall) => {
      return executeFileClientTool(conversationWorkspace, toolCall);
    },
    onEffect: ({ name, data }) => {
      if (name !== "app.toast") return;
      const message = typeof data?.message === "string" ? data.message : "";
      if (!message) return;
      const level = isToastLevel(data?.level) ? data.level : "info";
      showToast({
        level,
        message,
        title: typeof data?.title === "string" ? data.title : undefined,
      });
    },
    onThreadChange: ({ threadId }) => {
      activeThreadRef.current = threadId;
      setActiveThreadId(threadId);
      setSavedTitle("");
      setDraftTitle("");
      const generation = ++titleLoadGeneration.current;
      if (threadId) void loadTitle(threadId, generation);
      conversationWorkspace.setActiveThreadId(threadId);
    },
    onThreadLoadStart: ({ threadId }) => {
      conversationWorkspace.restoreThread(threadId);
    },
    onResponseEnd: () => {
      void conversationWorkspace.refreshThreadFiles();
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
      {toast && (
        <div
          className={`app-toast app-toast-${toast.level}`}
          role={toast.level === "danger" ? "alert" : "status"}
          aria-live={toast.level === "danger" ? "assertive" : "polite"}
        >
          <div>
            {toast.title && <strong>{toast.title}</strong>}
            <span>{toast.message}</span>
          </div>
          <button type="button" onClick={dismissToast} aria-label="Dismiss notification">
            ×
          </button>
        </div>
      )}
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

type ToastLevel = "info" | "warning" | "danger";

interface AppToast {
  level: ToastLevel;
  message: string;
  title?: string;
}

function isToastLevel(value: unknown): value is ToastLevel {
  return value === "info" || value === "warning" || value === "danger";
}

async function responseError(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : fallback;
  } catch {
    return fallback;
  }
}
