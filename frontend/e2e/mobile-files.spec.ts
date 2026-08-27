import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 } });

test("mobile file library is separate, dense, and preserves conversation inclusion", async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("e2e_clerk_token", "test-token");
    window.sessionStorage.setItem("mi_active_thread_id", "thread_mobile");
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
  await page.route("**/api/collections/collection_general/files**", (route) => {
    if (route.request().method() === "PUT") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          asset_id: "asset_notes",
          thread_id: "thread_mobile",
          included: true,
          include_id: "include_mobile",
        }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        items: [
          {
            asset_id: "asset_notes",
            filename: "interview-demo-notes.md",
            media_type: "text/markdown",
            route: "text",
            size_bytes: 2048,
            created_at: "2026-08-25T10:00:00Z",
            collection_id: "collection_general",
            ingestion_status: "ready",
            provider_status: "ready",
            artifact_count: 2,
            provider_file_count: 2,
            included: false,
            include_id: null,
            last_error: null,
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      }),
    });
  });
  await page.route("**/api/assets**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/");
  await expect(page.locator(".workspace > .artifact-panel")).toBeHidden();
  await page.getByRole("link", { name: "Files" }).evaluate((link) => {
    window.sessionStorage.setItem("mi_active_thread_id", "thread_mobile");
    (link as HTMLAnchorElement).click();
  });
  await expect(page).toHaveURL(/\/files$/);
  await expect(page.getByText("interview-demo-notes.md")).toBeVisible();
  await expect(page.getByText("2 indexed")).toBeVisible();
  await page.getByRole("button", { name: "Add to workspace", exact: true }).click();
  await expect(page.getByRole("button", { name: "In workspace" })).toBeVisible();
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBe(false);
});
