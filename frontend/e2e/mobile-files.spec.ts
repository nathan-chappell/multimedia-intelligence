import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 } });

test("mobile file library is separate, dense, and preserves conversation inclusion", async ({ page }) => {
  const collectionQueries: string[] = [];
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
          slug: "general",
          name: "General",
          description: null,
        },
      ]),
    }),
  );
  await page.route("**/api/collections/collection_general/files**", (route) => {
    collectionQueries.push(route.request().url());
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
        limit: 10,
        offset: 0,
      }),
    });
  });
  await page.route("**/api/assets/asset_mobile/content", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: "image bytes" }),
  );
  await page.route("**/api/assets/asset_notes/inclusion", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ include_id: "include_mobile" }),
    }),
  );
  await page.route(/\/api\/assets(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([{
        asset_id: "asset_mobile",
        include_id: "include_asset_mobile",
        filename: "mobile-preview.png",
        media_type: "image/png",
        size_bytes: 128,
        collection_id: null,
        source_asset_id: null,
      }]),
    }),
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
  expect(new URL(collectionQueries[0]).searchParams.get("limit")).toBe("10");

  const addFilesButton = page.getByRole("button", { name: "Add files" });
  expect((await addFilesButton.boundingBox())?.height).toBeGreaterThanOrEqual(44);
  const workspaceList = page.getByLabel("Files in the workspace");
  expect(await workspaceList.evaluate((element) => getComputedStyle(element).maxHeight)).not.toBe("none");
  await workspaceList.locator("article").filter({ hasText: "mobile-preview.png" })
    .getByRole("button", { name: "Preview" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  const dialogBounds = await dialog.boundingBox();
  expect(dialogBounds?.width).toBeLessThanOrEqual(390);
  expect(dialogBounds?.height).toBeLessThanOrEqual(844);
  expect((await page.getByRole("button", { name: "Close preview" }).boundingBox())?.height)
    .toBeGreaterThanOrEqual(44);
  await page.keyboard.press("Escape");
  const hasHorizontalOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
  expect(hasHorizontalOverflow).toBe(false);
});
