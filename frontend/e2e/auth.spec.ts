import { expect, test } from "@playwright/test";

test("sends an unauthenticated visitor to sign in", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "Welcome back." })).toBeVisible();
});

test("signs in and stores the returned access token", async ({ page }) => {
  await page.route("**/api/auth/token", async (route) => {
    expect(route.request().postData()).toContain("username=admin");
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_token: "signed-test-token", token_type: "bearer" }),
    });
  });
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "user_admin", username: "admin", is_admin: true }),
    }),
  );
  await page.route("**/chatkit", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.goto("/login");
  await page.getByLabel("Username").fill("admin");
  await page.getByLabel("Password").fill("local-development-admin-password");
  await page.getByRole("button", { name: "Sign in" }).click();

  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { name: "Multimedia Intelligence" })).toBeVisible();
  expect(await page.evaluate(() => localStorage.getItem("api_bearer_token"))).toBe(
    "signed-test-token",
  );
});

test("shows an error page for an invalid or expired token", async ({ page }) => {
  await page.addInitScript(() =>
    window.localStorage.setItem("api_bearer_token", "expired-token"),
  );
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({
      status: 401,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Invalid or expired bearer token" }),
    }),
  );

  await page.goto("/");

  await expect(page).toHaveURL(/\/auth-error$/);
  await expect(page.getByText("Your session is invalid or has expired.")).toBeVisible();
  await expect(page.getByRole("link", { name: "Return to sign in" })).toBeVisible();
});
