import { expect, test } from "@playwright/test";

test("adds a local file to the durable user workspace", async ({ page }) => {
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
  await expect(page.getByText("Collection", { exact: true })).toBeVisible();
  await expect(
    page.getByLabel("Library & workspace").getByText("Workspace", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("Available to tools")).toBeVisible();
});

test("keeps user workspace files when the selected collection changes", async ({ page }) => {
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
      name: "Collection one",
      description: null,
      selected: true,
      is_public: false,
      owned: true,
      can_manage: true,
      read_only: false,
    },
    {
      id: "collection_two",
      name: "Collection two",
      description: null,
      selected: false,
      is_public: false,
      owned: true,
      can_manage: true,
      read_only: false,
    },
  ];
  await page.route("**/api/collections/selection", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...collections[1], selected: true }),
    }),
  );
  await page.route("**/api/collections", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(collections) }),
  );
  await page.route("**/api/collections/*/files**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0, limit: 100, offset: 0 }),
    }),
  );
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
  await page.getByLabel("Active collection").selectOption("collection_two");
  await expect(page.getByLabel("Active collection")).toHaveValue("collection_two");
  await expect(page.getByText("workspace-notes.md")).toBeVisible();
});
