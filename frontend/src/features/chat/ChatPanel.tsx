import { ChatKit, useChatKit } from "@openai/chatkit-react";

import { executeFileClientTool } from "../artifacts/clientToolHandler";
import { useFileWorkspace } from "../artifacts/useFileWorkspace";
import { authenticatedFetch, config } from "../../lib/config";

export function ChatPanel() {
  const fileWorkspace = useFileWorkspace();
  const chatkit = useChatKit({
    api: {
      url: config.chatkitUrl,
      domainKey: config.domainKey,
      fetch: authenticatedFetch,
    },
    onClientTool: (toolCall) => executeFileClientTool(fileWorkspace, toolCall),
    composer: {
      placeholder: "Ask about this conversation's files…",
      models: [
        {
          id: "gpt-5.6",
          label: "GPT-5.6",
          description: "Frontier model with medium reasoning",
          default: true,
        },
        {
          id: "gpt-5.6-terra",
          label: "GPT-5.6 Terra",
          description: "Balanced capability and cost",
        },
        {
          id: "gpt-5.6-luna",
          label: "GPT-5.6 Luna",
          description: "Fast, cost-sensitive workloads",
        },
      ],
      // Keep disabled until the backend maps local uploads to safe, conversation-owned
      // OpenAI file inputs. Enabling early would create convincing but broken plumbing.
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
