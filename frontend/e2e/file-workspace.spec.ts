import { expect, test } from "@playwright/test";

test("adds a local file to the durable user workspace", async ({ page }) => {
  let collectionRequests = 0;
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
  await page.route("**/api/collections", (route) => {
    collectionRequests += 1;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "collection_general",
          slug: "general",
          name: "General",
          description: null,
        },
      ]),
    });
  });
  await page.route("**/api/collections/collection_general/files**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0, limit: 100, offset: 0 }),
    }),
  );
  await page.route(/\/api\/assets(?:\?.*)?$/, (route) => {
    if (route.request().method() === "POST") {
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({
          asset_id: "asset_exchange_rates",
          include_id: "workspace_exchange_rates",
          filename: "exchange-rates.csv",
          media_type: "text/csv",
          size_bytes: 39,
          collection_id: null,
        }),
      });
    }
    return route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
  });
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "Multimedia Intelligence" })).toBeVisible();
  await page.locator('input[type="file"]').setInputFiles({
    name: "exchange-rates.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("date,currency,rate\n2026-08-21,EUR,0.86\n"),
  });

  await expect(page.getByText("exchange-rates.csv")).toBeVisible();
  await expect(page.getByRole("status")).toContainText("1 file(s) added to the workspace");
  await expect(page.getByText("Collection", { exact: true })).toHaveCount(0);
  await expect(
    page.getByLabel("Workspace files").getByText("Workspace", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Available to tools")).toBeVisible();
  await page.getByLabel("Files in the workspace").getByRole("button", { name: "Preview" }).click();
  await expect(page.getByRole("dialog").locator("pre")).toContainText("2026-08-21,EUR,0.86");
  await page.getByRole("button", { name: "Close preview" }).click();
  expect(collectionRequests).toBe(0);
});

test("pages compact workspace files and hydrates a durable text preview", async ({ page }) => {
  let clearRequests = 0;
  await page.addInitScript(() => window.localStorage.setItem("e2e_clerk_token", "test-token"));
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
  let assets = Array.from({ length: 6 }, (_, index) => ({
    asset_id: `asset_${index + 1}`,
    include_id: `include_${index + 1}`,
    filename: `workspace-${index + 1}.txt`,
    media_type: "text/plain",
    size_bytes: 24,
    collection_id: null,
    source_asset_id: null,
  }));
  await page.route("**/api/assets/workspace", (route) => {
    clearRequests += 1;
    const removedCount = assets.length;
    assets = [];
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ removed_count: removedCount }),
    });
  });
  await page.route(/\/api\/assets(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(assets),
    }),
  );
  await page.route("**/api/assets/asset_1/inclusion", (route) => {
    assets = assets.filter((asset) => asset.asset_id !== "asset_1");
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ include_id: null }),
    });
  });
  await page.route("**/api/assets/asset_6/content", (route) =>
    route.fulfill({ status: 200, contentType: "text/plain", body: "contents for file 6" }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/");
  const workspaceList = page.getByLabel("Files in the workspace");
  await expect(workspaceList.locator("article")).toHaveCount(5);
  await expect(page.getByText("workspace-6.txt")).toBeVisible();
  await expect(page.getByText("workspace-1.txt")).toHaveCount(0);

  await workspaceList.locator("article").filter({ hasText: "workspace-6.txt" })
    .getByRole("button", { name: "Preview" }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByText("contents for file 6")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);

  await page.getByRole("navigation", { name: "Workspace files pagination" })
    .getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("workspace-1.txt")).toBeVisible();
  await expect(workspaceList.locator("article")).toHaveCount(1);
  await workspaceList.getByRole("button", { name: "Remove" }).click();
  await expect(page.getByText("workspace-1.txt")).toHaveCount(0);
  await expect(page.getByText("workspace-6.txt")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Workspace files pagination" })).toHaveCount(0);

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "Clear workspace" }).click();
  await expect(page.getByRole("heading", { name: "No workspace files" })).toBeVisible();
  await expect(page.getByRole("status")).toContainText("Stored files and collection indexes were not deleted");
  expect(clearRequests).toBe(1);
});

test("keeps user workspace files when the focused collection changes", async ({ page }) => {
  const collectionQueries: Array<{ collectionId: string; limit: string | null; offset: string | null }> = [];
  await page.addInitScript(() => {
    window.localStorage.setItem("e2e_clerk_token", "test-token");
    window.sessionStorage.setItem("mi_active_thread_id", "thread_workspace");
  });
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
  const collections = [
    {
      id: "collection_one",
      slug: "collection-one",
      name: "Collection one",
      description: null,
    },
    {
      id: "collection_two",
      slug: "collection-two",
      name: "Collection two",
      description: null,
    },
  ];
  await page.route("**/api/collections", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(collections) }),
  );
  await page.route("**/api/collections/*/files**", (route) => {
    const url = new URL(route.request().url());
    const collectionId = url.pathname.split("/").at(-2) ?? "unknown";
    const offset = url.searchParams.get("offset");
    collectionQueries.push({ collectionId, limit: url.searchParams.get("limit"), offset });
    const suffix = offset === "10" ? "second-page" : "first-page";
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [{
          asset_id: `${collectionId}-${suffix}`,
          filename: `${collectionId}-${suffix}.pdf`,
          media_type: "application/pdf",
          route: "pdf",
          size_bytes: 1024,
          created_at: "2026-08-25T10:00:00Z",
          collection_id: collectionId,
          ingestion_status: "ready",
          provider_status: "ready",
          artifact_count: 0,
          provider_file_count: 1,
          included: false,
          include_id: null,
          last_error: null,
        }],
        total: 11,
        limit: 10,
        offset: Number(offset),
      }),
    });
  });
  await page.route("**/api/assets/derived?**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route(/\/api\/assets(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          asset_id: "asset_workspace",
          include_id: "include_workspace",
          filename: "workspace-notes.md",
          media_type: "text/markdown",
          size_bytes: 128,
          collection_id: "collection_one",
        },
      ]),
    }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/files");
  await expect(page.getByText("workspace-notes.md")).toBeVisible();
  await expect(page.getByText("collection_one-first-page.pdf")).toBeVisible();
  await page.getByRole("navigation", { name: "Collection files pagination" })
    .getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("collection_one-second-page.pdf")).toBeVisible();
  await page.getByLabel("Browse collection").selectOption("collection_two");
  await expect(page.getByLabel("Browse collection")).toHaveValue("collection_two");
  await expect(page.getByText("collection_two-first-page.pdf")).toBeVisible();
  await expect(page.getByText("workspace-notes.md")).toBeVisible();
  expect(collectionQueries).toContainEqual({ collectionId: "collection_one", limit: "10", offset: "0" });
  expect(collectionQueries).toContainEqual({ collectionId: "collection_one", limit: "10", offset: "10" });
  expect(collectionQueries).toContainEqual({ collectionId: "collection_two", limit: "10", offset: "0" });
});

test("previews durable workspace files with type-aware browser controls", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("e2e_clerk_token", "test-token");
    const counts = { created: 0, revoked: 0 };
    const trackedUrls = new Set<string>();
    const createObjectUrl = URL.createObjectURL.bind(URL);
    const revokeObjectUrl = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (value) => {
      const url = createObjectUrl(value);
      if (value instanceof File && value.name.startsWith("sample.")) {
        counts.created += 1;
        trackedUrls.add(url);
      }
      return url;
    };
    URL.revokeObjectURL = (value) => {
      if (trackedUrls.delete(value)) counts.revoked += 1;
      revokeObjectUrl(value);
    };
    (window as typeof window & { previewUrlCounts: typeof counts }).previewUrlCounts = counts;
  });
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
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([{ id: "general", slug: "general", name: "General", description: null }]),
    }),
  );
  await page.route("**/api/collections/general/files**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0, limit: 10, offset: 0 }),
    }),
  );
  const assets = [
    ["text", "sample.txt", "text/plain"],
    ["pdf", "sample.pdf", "application/pdf"],
    ["image", "sample.png", "image/png"],
    ["audio", "sample.mp3", "audio/mpeg"],
    ["video", "sample.mp4", "video/mp4"],
    ["other", "sample.bin", "application/octet-stream"],
  ].map(([id, filename, mediaType]) => ({
    asset_id: id,
    include_id: `include-${id}`,
    filename,
    media_type: mediaType,
    size_bytes: 32,
    collection_id: null,
    source_asset_id: null,
  }));
  await page.route(/\/api\/assets(?:\?.*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(assets) }),
  );
  await page.route("**/api/assets/*/content", (route) => {
    const id = new URL(route.request().url()).pathname.split("/").at(-2) ?? "other";
    const mediaType = assets.find((asset) => asset.asset_id === id)?.media_type ?? "application/octet-stream";
    return route.fulfill({
      status: 200,
      contentType: mediaType,
      body: id === "text" ? "A durable text preview" : "preview bytes",
    });
  });
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/files");
  const previewUrlBaseline = await page.evaluate(
    () => (window as typeof window & { previewUrlCounts: { created: number; revoked: number } })
      .previewUrlCounts,
  );

  async function openPreview(filename: string) {
    await page.getByLabel("Files in the workspace").locator("article").filter({ hasText: filename })
      .getByRole("button", { name: "Preview" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
  }

  await openPreview("sample.txt");
  await expect(page.getByText("A durable text preview")).toBeVisible();
  await page.getByRole("button", { name: "Close preview" }).click();

  await openPreview("sample.pdf");
  await expect(page.getByRole("dialog").locator("iframe")).toBeVisible();
  await page.keyboard.press("Escape");

  await openPreview("sample.png");
  await expect(page.getByRole("dialog").locator("img")).toBeVisible();
  await page.keyboard.press("Escape");

  await openPreview("sample.mp3");
  await expect(page.getByRole("dialog").locator("audio[controls]")).toBeVisible();
  await page.keyboard.press("Escape");

  await openPreview("sample.mp4");
  await expect(page.getByRole("dialog").locator("video[controls]")).toBeVisible();
  await page.keyboard.press("Escape");

  await openPreview("sample.bin");
  await expect(page.getByRole("link", { name: "Download file" })).toBeVisible();
  await page.getByRole("dialog").click({ position: { x: 2, y: 2 } });
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect.poll(async () => {
    const counts = await page.evaluate(
      () => (window as typeof window & { previewUrlCounts: { created: number; revoked: number } })
        .previewUrlCounts,
    );
    return counts.created - counts.revoked;
  }).toBe(previewUrlBaseline.created - previewUrlBaseline.revoked);
  const finalUrlCounts = await page.evaluate(
    () => (window as typeof window & { previewUrlCounts: { created: number; revoked: number } })
      .previewUrlCounts,
  );
  expect(finalUrlCounts.created - previewUrlBaseline.created).toBeGreaterThanOrEqual(5);
});
