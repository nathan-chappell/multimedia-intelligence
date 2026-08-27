import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 } });

const sessionUser = {
  id: "test-user",
  username: "tester",
  email: "tester@example.com",
  full_name: "Test User",
  role: "user",
  is_admin: false,
  balance_microusd: 5_000_000,
};

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("e2e_clerk_token", "test-token");
  });
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(sessionUser),
    }),
  );
  await page.route("**/api/collections/*/files**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0, limit: 100, offset: 0 }),
    }),
  );
  await page.route(/\/api\/assets(?:\?.*)?$/, (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
});

test("shows a public collection as a searchable read-only workspace", async ({ page }) => {
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "collection_shared",
          name: "Interview Demo",
          description: "Seeded evidence",
          selected: true,
          is_public: true,
          owned: false,
          can_manage: false,
          read_only: true,
        },
      ]),
    }),
  );

  await page.goto("/files");

  await expect(page.getByLabel("Active collection")).toHaveValue("collection_shared");
  await expect(page.getByText("Public", { exact: true })).toBeVisible();
  await expect(page.getByText(/Chat can search its indexed files/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Add files" })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Refresh index status" })).toBeDisabled();
});

test("lets an owner publish a private collection", async ({ page }) => {
  const privateCollection = {
    id: "collection_owned",
    name: "My Research",
    description: null,
    selected: true,
    is_public: false,
    owned: true,
    can_manage: true,
    read_only: false,
  };
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([privateCollection]),
    }),
  );
  await page.route("**/api/collections/collection_owned", async (route) => {
    expect(await route.request().postDataJSON()).toEqual({ is_public: true });
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ ...privateCollection, is_public: true }),
    });
  });

  await page.goto("/files");
  await page.getByRole("button", { name: "Make public" }).click();

  await expect(page.getByText("Public", { exact: true })).toBeVisible();
  await expect(page.getByRole("status")).toContainText("visible to signed-in users");
});
