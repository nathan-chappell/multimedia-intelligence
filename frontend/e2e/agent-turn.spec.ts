import { expect, test, type Page, type Route } from "@playwright/test";

const threadId = "thr_e2e_agent";
const userMessageId = "msg_e2e_user";
const toolItemId = "fc_e2e_list_files";
const toolCallId = "call_e2e_list_files";
const assistantMessageId = "msg_e2e_assistant";

test("shows a host toast when ChatKit emits an application effect", async ({ page }) => {
  await authenticate(page);
  await page.route("**/api/collections", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/assets?**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/assets/derived?**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/threads/**", (route) =>
    route.fulfill({ status: 404, contentType: "application/json", body: "{}" }),
  );
  await page.route("**/chatkit", async (route) => {
    const request = route.request().postDataJSON() as ChatKitRequest;
    if (request.type === "threads.list") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ data: [], has_more: false }),
      });
      return;
    }
    if (request.type === "threads.create") {
      await fulfillEvents(route, [
        ...initialToolCallEvents().slice(0, 2),
        {
          type: "client_effect",
          name: "app.toast",
          data: {
            level: "danger",
            title: "Credit required",
            message: "Credit balance is exhausted.",
          },
        },
      ]);
      return;
    }
    throw new Error(`Unexpected ChatKit request: ${request.type}`);
  });

  await page.goto("/");
  const chat = page.frameLocator("iframe");
  const composer = chat.getByRole("textbox", {
    name: "Ask about this conversation's files…",
  });
  await composer.fill("Try another request.");
  await composer.press("Enter");

  const toast = page.getByRole("alert");
  await expect(toast).toContainText("Credit required");
  await expect(toast).toContainText("Credit balance is exhausted.");
});

test("completes a browser tool call and renders the agent response", async ({ page }) => {
  await authenticate(page);
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "collection_general",
          name: "General",
          description: null,
          selected: true,
          is_public: false,
          owned: true,
          can_manage: true,
          read_only: false,
        },
      ]),
    }),
  );
  await page.route("**/api/assets?**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/assets/derived?**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/threads/**", (route) =>
    route.fulfill({ status: 404, contentType: "application/json", body: "{}" }),
  );

  let continuationSeen = false;
  await page.route("**/chatkit", async (route) => {
    const request = route.request().postDataJSON() as ChatKitRequest;
    if (request.type === "threads.list") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ data: [], has_more: false }),
      });
      return;
    }
    if (request.type === "threads.create") {
      expect(request.params.input?.content?.[0]?.text).toBe("Inspect my staged files.");
      await fulfillEvents(route, initialToolCallEvents());
      return;
    }
    if (request.type === "threads.add_client_tool_output") {
      continuationSeen = true;
      expect(request.params.thread_id).toBe(threadId);
      expect(request.params.result).toMatchObject({
        ok: true,
        page: 1,
        total: 1,
        hasMore: false,
      });
      expect(request.params.result?.files?.[0]).toMatchObject({
        name: "notes.md",
        route: "text",
      });
      await fulfillEvents(route, [
        toolResultWidgetEvent(),
        ...assistantEvents("I found notes.md and can inspect it."),
      ]);
      return;
    }
    throw new Error(`Unexpected ChatKit request: ${request.type}`);
  });

  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles({
    name: "notes.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("# Notes\nThe ingestion pipeline is connected.\n"),
  });

  const chat = page.frameLocator("iframe");
  const composer = chat.getByRole("textbox", {
    name: "Ask about this conversation's files…",
  });
  await composer.fill("Inspect my staged files.");
  await composer.press("Enter");

  await expect(chat.getByText("I found notes.md and can inspect it.")).toBeVisible({
    timeout: 15_000,
  });
  const resultSummary = chat.getByText("Found 1 conversation file");
  await expect(resultSummary).toBeVisible();
  await expect(chat.getByText("Browser tool result · List files")).toBeHidden();
  await resultSummary.locator("..").getByRole("button").click();
  await expect(chat.getByText("Browser tool result · List files")).toBeVisible();
  await expect(chat.getByText('"route": "text"')).toBeVisible();
  expect(continuationSeen).toBe(true);
});

test("hydrates a saved conversation file before a client tool reads it", async ({ page }) => {
  await authenticate(page);
  await page.addInitScript((savedThreadId) => {
    window.sessionStorage.setItem("mi_active_thread_id", savedThreadId);
  }, threadId);
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "collection_general",
          name: "General",
          description: null,
          selected: true,
          is_public: false,
          owned: true,
          can_manage: true,
          read_only: false,
        },
      ]),
    }),
  );
  await page.route("**/api/collections/collection_general/files**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0, limit: 100, offset: 0 }),
    }),
  );
  let contentLoads = 0;
  await page.route("**/api/assets/asset_saved/content?**", async (route) => {
    contentLoads += 1;
    await route.fulfill({
      status: 200,
      contentType: "text/markdown",
      body: "# Demo notes\nThe project codename is Juniper.\n",
    });
  });
  await page.route("**/api/assets/derived?**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/assets?**", async (route) => {
    // Make the client tool race the refresh hydration. It must wait for this metadata.
    await new Promise((resolve) => setTimeout(resolve, 350));
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          asset_id: "asset_saved",
          include_id: "include_saved",
          filename: "demo-notes.md",
          media_type: "text/markdown",
          size_bytes: 48,
          collection_id: "collection_general",
        },
      ]),
    });
  });
  await page.route("**/api/threads/**", (route) =>
    route.fulfill({ status: 404, contentType: "application/json", body: "{}" }),
  );

  let continuationSeen = false;
  await page.route("**/chatkit", async (route) => {
    const request = route.request().postDataJSON() as ChatKitRequest;
    if (request.type === "threads.list") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ data: [], has_more: false }),
      });
      return;
    }
    if (request.type === "threads.get_by_id") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: threadId,
          created_at: "2026-08-24T10:00:00Z",
          status: { type: "active" },
          items: { data: [], has_more: false },
        }),
      });
      return;
    }
    if (request.type === "threads.add_user_message") {
      await fulfillEvents(route, [
        {
          type: "thread.item.done",
          item: {
            id: "msg_hydrated_user",
            thread_id: threadId,
            created_at: "2026-08-24T10:00:02Z",
            type: "user_message",
            content: [{ type: "input_text", text: "What is the project codename?" }],
            attachments: [],
            quoted_text: "",
            inference_options: { model: "gpt-5.6-luna" },
          },
        },
        { type: "stream_options", stream_options: { allow_cancel: true } },
        {
          type: "thread.item.done",
          item: {
            id: "fc_hydrated_read",
            thread_id: threadId,
            created_at: "2026-08-24T10:00:02Z",
            type: "client_tool_call",
            status: "pending",
            call_id: "call_hydrated_read",
            name: "read_text_chars",
            arguments: { assetId: "asset_saved", start: 0, count: 200 },
          },
        },
      ]);
      return;
    }
    if (request.type === "threads.add_client_tool_output") {
      continuationSeen = true;
      expect(request.params.result).toMatchObject({
        ok: true,
        assetId: "asset_saved",
        start: 0,
      });
      expect(request.params.result?.text).toContain("Juniper");
      await fulfillEvents(route, assistantEvents("The project codename is Juniper."));
      return;
    }
    throw new Error(`Unexpected ChatKit request: ${request.type}`);
  });

  await page.goto("/");
  const chat = page.frameLocator("iframe");
  const composer = chat.getByRole("textbox", {
    name: "Ask about this conversation's files…",
  });
  await composer.fill("What is the project codename?");
  await composer.press("Enter");

  await expect(chat.getByText("The project codename is Juniper.")).toBeVisible({
    timeout: 15_000,
  });
  expect(continuationSeen).toBe(true);
  expect(contentLoads).toBe(1);
});

test("renders an inline chart and restores its saved collection artifact", async ({ page }) => {
  await authenticate(page);
  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
    "base64",
  );
  const dataUrl = `data:image/png;base64,${png.toString("base64")}`;
  let derivedLoads = 0;
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "collection_trends",
          name: "Language Trends",
          description: null,
          selected: true,
          is_public: false,
          owned: true,
          can_manage: true,
          read_only: false,
        },
      ]),
    }),
  );
  await page.route("**/api/assets?**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          asset_id: "asset_trends",
          include_id: "include_trends",
          filename: "programming-language-trends.csv",
          media_type: "text/csv",
          size_bytes: 26,
          collection_id: "collection_trends",
        },
      ]),
    }),
  );
  await page.route("**/api/assets/derived?**", (route) => {
    derivedLoads += 1;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(
        derivedLoads === 1
          ? []
          : [
              {
                artifact_id: "artifact_chart",
                source_asset_id: "asset_trends",
                filename: "programming-trends.png",
                media_type: "image/png",
                size_bytes: png.length,
                kind: "chart",
                collection_id: "collection_trends",
              },
            ],
      ),
    });
  });
  await page.route("**/api/assets/derived/artifact_chart/content?**", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: png }),
  );
  await page.route("**/api/assets/asset_trends/content?**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/csv",
      body: "year,language,value\n2025,TypeScript,49\n",
    }),
  );
  await page.route("**/api/threads/**", (route) =>
    route.fulfill({ status: 404, contentType: "application/json", body: "{}" }),
  );
  await page.route("**/chatkit", async (route) => {
    const request = route.request().postDataJSON() as ChatKitRequest;
    if (request.type === "threads.list") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ data: [], has_more: false }),
      });
      return;
    }
    if (request.type === "threads.create") {
      const events = initialToolCallEvents().slice(0, 3);
      events.push({
        type: "thread.item.done",
        item: {
          id: "chart_message",
          thread_id: threadId,
          created_at: "2026-08-24T10:00:01Z",
          type: "generated_image",
          image: { id: "artifact_chart", url: dataUrl },
        },
      });
      events.push(...assistantEvents("I created the requested chart with two series."));
      await fulfillEvents(route, events);
      return;
    }
    throw new Error(`Unexpected ChatKit request: ${request.type}`);
  });

  await page.goto("/");
  const chat = page.frameLocator("iframe");
  const composer = chat.getByRole("textbox", {
    name: "Ask about this conversation's files…",
  });
  await composer.fill("Chart the language trends.");
  await composer.press("Enter");

  await expect(chat.getByText("I created the requested chart with two series.")).toBeVisible({
    timeout: 15_000,
  });
  await expect(chat.locator('img[src^="data:image/png;base64,"]').first()).toBeVisible();
  await page.getByText("Derived previews (1)").click();
  await expect(page.getByText("programming-trends.png")).toBeVisible();
  await expect(page.getByText(/saved/)).toBeVisible();
  expect(derivedLoads).toBeGreaterThanOrEqual(2);
});

async function authenticate(page: Page): Promise<void> {
  await page.addInitScript(() =>
    window.localStorage.setItem("e2e_clerk_token", "test-token"),
  );
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "test-user",
        username: "tester",
        email: "tester@example.com",
        full_name: "Test User",
        role: "user",
        is_admin: false,
        balance_microusd: 5_000_000,
      }),
    }),
  );
}

async function fulfillEvents(route: Route, events: object[]): Promise<void> {
  await route.fulfill({
    status: 200,
    contentType: "text/event-stream",
    body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
  });
}

function initialToolCallEvents(): object[] {
  const createdAt = "2026-08-24T10:00:00Z";
  return [
    {
      type: "thread.created",
      thread: {
        id: threadId,
        created_at: createdAt,
        status: { type: "active" },
        metadata: {},
        items: { data: [], has_more: false },
      },
    },
    {
      type: "thread.item.done",
      item: {
        id: userMessageId,
        thread_id: threadId,
        created_at: createdAt,
        type: "user_message",
        content: [{ type: "input_text", text: "Inspect my staged files." }],
        attachments: [],
        quoted_text: "",
        inference_options: { model: "gpt-5.6-luna" },
      },
    },
    { type: "stream_options", stream_options: { allow_cancel: true } },
    {
      type: "thread.item.done",
      item: {
        id: toolItemId,
        thread_id: threadId,
        created_at: createdAt,
        type: "client_tool_call",
        status: "pending",
        call_id: toolCallId,
        name: "list_files",
        arguments: { page: 1, durableFiles: [] },
      },
    },
  ];
}

function assistantEvents(text: string): object[] {
  const createdAt = "2026-08-24T10:00:01Z";
  return [
    { type: "stream_options", stream_options: { allow_cancel: true } },
    {
      type: "thread.item.added",
      item: {
        id: assistantMessageId,
        thread_id: threadId,
        created_at: createdAt,
        type: "assistant_message",
        content: [],
      },
    },
    {
      type: "thread.item.updated",
      item_id: assistantMessageId,
      update: {
        type: "assistant_message.content_part.added",
        content_index: 0,
        content: { type: "output_text", text: "", annotations: [] },
      },
    },
    {
      type: "thread.item.updated",
      item_id: assistantMessageId,
      update: {
        type: "assistant_message.content_part.text_delta",
        content_index: 0,
        delta: text,
      },
    },
    {
      type: "thread.item.done",
      item: {
        id: assistantMessageId,
        thread_id: threadId,
        created_at: createdAt,
        type: "assistant_message",
        content: [{ type: "output_text", text, annotations: [] }],
      },
    },
  ];
}

function toolResultWidgetEvent(): object {
  const preview = JSON.stringify(
    {
      ok: true,
      page: 1,
      pageSize: 10,
      total: 1,
      hasMore: false,
      files: [
        {
          name: "notes.md",
          mediaType: "text/markdown",
          sizeBytes: 47,
          route: "text",
          durability: "local",
        },
      ],
    },
    null,
    2,
  );
  return {
    type: "thread.item.done",
    item: {
      id: "tool_result_e2e_list_files",
      thread_id: threadId,
      created_at: "2026-08-24T10:00:01Z",
      type: "widget",
      widget: {
        type: "Card",
        size: "sm",
        collapsed: true,
        status: { text: "Found 1 conversation file" },
        children: [
          {
            type: "Caption",
            value: "Browser tool result · List files",
            color: "secondary",
            size: "sm",
          },
          { type: "Markdown", value: `\`\`\`json\n${preview}\n\`\`\`` },
        ],
      },
      copy_text: preview,
    },
  };
}

interface ChatKitRequest {
  type: string;
  params: {
    thread_id?: string;
    input?: { content?: Array<{ text?: string }> };
    result?: {
      ok?: boolean;
      assetId?: string;
      start?: number;
      text?: string;
      page?: number;
      total?: number;
      hasMore?: boolean;
      files?: Array<{ name?: string; route?: string }>;
    };
  };
}
