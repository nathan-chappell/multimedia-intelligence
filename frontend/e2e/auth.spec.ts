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
      body: JSON.stringify([{ id: "collection_general", name: "General", description: null, selected: true }]),
    }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Multimedia Intelligence" })).toBeVisible();
  await expect(page.getByRole("link", { name: /\$5\.00 credit/ })).toBeVisible();
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
