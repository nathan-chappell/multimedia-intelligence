import { expect, test } from "@playwright/test";

test("stages a conversation file and offers save", async ({ page }) => {
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
        { id: "collection_general", name: "General", description: null, selected: true },
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
  await expect(page.getByRole("status")).toContainText("1 file(s) staged");
  await expect(page.getByText("Ready to upload")).toBeVisible();
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeVisible();
  await expect(page.locator(".counter")).toHaveText("0");
});
