import { ChatKit, useChatKit } from "@openai/chatkit-react";

import { executeFileClientTool } from "../artifacts/clientToolHandler";
import { useFileWorkspace } from "../artifacts/useFileWorkspace";
import { useTransientStatus } from "../status/transientStatus";
import { authenticatedFetch, config } from "../../lib/config";

export function ChatPanel() {
  const fileWorkspace = useFileWorkspace();
  const { showStatus } = useTransientStatus();
  const chatkit = useChatKit({
    api: {
      url: config.chatkitUrl,
      domainKey: config.domainKey,
      fetch: authenticatedFetch,
    },
    onClientTool: async (toolCall) => {
      const label = clientToolLabel(toolCall.name);
      showStatus(`${label}…`, { durationMs: 0 });
      const result = await executeFileClientTool(fileWorkspace, toolCall);
      if (result.ok === false) {
        showStatus(`${label} failed`, { tone: "error", durationMs: 5000 });
      } else {
        showStatus(`${label} complete`, { tone: "success" });
      }
      return result;
    },
    onThreadChange: ({ threadId }) => {
      fileWorkspace.setActiveThreadId(threadId);
    },
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

  return (
    <article className="panel chat-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Assistant</span>
          <h2>Ask, compare, extract</h2>
        </div>
        <span className="status">Ready</span>
      </div>
      <ChatKit control={chatkit.control} className="chatkit" />
    </article>
  );
}

function clientToolLabel(name: string): string {
  const labels: Record<string, string> = {
    list_files: "Checking conversation files",
    read_text_chars: "Reading text",
    json_chars: "Reading JSON",
    json_path: "Querying JSON",
    csv_head: "Sampling CSV rows",
    csv_stats: "Calculating CSV statistics",
    pdf_random_sample: "Sampling PDF pages",
    pdf_render_page: "Rendering PDF page",
    pdf_extract_range: "Extracting PDF pages",
  };
  return labels[name] ?? "Running browser tool";
}
