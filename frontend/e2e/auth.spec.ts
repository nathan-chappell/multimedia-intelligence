import { expect, test } from "@playwright/test";

const sessionUser = {
  id: "user_test",
  username: "Test User",
  email: "test@example.com",
  full_name: "Test User",
  role: "user",
  is_admin: false,
  balance_microusd: 5_000_000,
};

test("sends an unauthenticated visitor to Clerk sign in", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/login$/);
});

test("hydrates the app with a Clerk bearer token", async ({ page }) => {
  await page.addInitScript(() =>
    window.localStorage.setItem("e2e_clerk_token", "clerk-test-token"),
  );
  await page.route("**/api/auth/me", async (route) => {
    expect(route.request().headers().authorization).toBe("Bearer clerk-test-token");
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(sessionUser) });
  });
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([{
        id: "collection_general",
        name: "General",
        description: null,
        selected: true,
        is_public: false,
        owned: true,
        can_manage: true,
        read_only: false,
      }]),
    }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Multimedia Intelligence" })).toBeVisible();
  await expect(page.getByRole("link", { name: /\$5\.00 credit/ })).toBeVisible();
});

test("navigates between authenticated pages without reloading or rechecking auth", async ({ page }) => {
  const adminUser = { ...sessionUser, role: "admin", is_admin: true } as const;
  let authChecks = 0;
  let documentRequests = 0;

  page.on("request", (request) => {
    if (request.resourceType() === "document" && request.frame() === page.mainFrame()) {
      documentRequests += 1;
    }
  });
  await page.addInitScript(() =>
    window.localStorage.setItem("e2e_clerk_token", "clerk-test-token"),
  );
  await page.route("**/api/auth/me", async (route) => {
    authChecks += 1;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(adminUser) });
  });
  await page.route("**/api/collections", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([{
        id: "collection_general",
        name: "General",
        description: null,
        selected: true,
        is_public: false,
        owned: true,
        can_manage: true,
        read_only: false,
      }]),
    }),
  );
  await page.route("**/api/collections/collection_general/files**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [], total: 0, limit: 100, offset: 0 }),
    }),
  );
  await page.route("**/api/assets**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/api/billing/ledger?**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ balance_microusd: 5_000_000, items: [], total: 0, limit: 10, offset: 0 }),
    }),
  );
  await page.route("**/api/admin/users**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ items: [adminUser], total: 1 }),
    }),
  );
  await page.route("**/api/admin/billing/ledger**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ balance_microusd: null, items: [], total: 0, limit: 10, offset: 0 }),
    }),
  );
  await page.route("**/api/admin/billing/coupons", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "[]" }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Multimedia Intelligence" })).toBeVisible();
  const initialAuthChecks = authChecks;
  await page.getByRole("link", { name: "Files" }).click();
  await expect(page).toHaveURL(/\/files$/);
  await page.getByRole("link", { name: "Chat" }).click();
  await expect(page).toHaveURL(/\/$/);
  await page.getByRole("link", { name: /\$5\.00 credit/ }).click();
  await expect(page.getByRole("heading", { name: "Your account" })).toBeVisible();
  await page.getByRole("link", { name: "← Workspace" }).click();
  await page.getByRole("link", { name: "Admin" }).click();
  await expect(page.getByRole("heading", { name: "Access and billing admin" })).toBeVisible();
  await page.goBack();
  await expect(page.getByRole("heading", { name: "Multimedia Intelligence" })).toBeVisible();

  expect(authChecks).toBe(initialAuthChecks);
  expect(documentRequests).toBe(1);
});

test("shows an inline error for a rejected Clerk session", async ({ page }) => {
  await page.addInitScript(() =>
    window.localStorage.setItem("e2e_clerk_token", "expired-token"),
  );
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({ status: 401, contentType: "application/json", body: JSON.stringify({ detail: "Invalid Clerk session" }) }),
  );

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Your Clerk session is invalid or has expired." })).toBeVisible();
});
