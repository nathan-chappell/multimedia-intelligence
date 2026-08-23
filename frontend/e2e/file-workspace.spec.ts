import { expect, test } from "@playwright/test";

test("stages a conversation file and offers save", async ({ page }) => {
  await page.addInitScript(() =>
    window.localStorage.setItem("api_bearer_token", "test-token"),
  );
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "test-user", username: "tester", is_admin: false }),
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
  await expect(page.getByRole("status")).toContainText("1 file staged");
  await expect(page.getByText("Staged locally")).toBeVisible();
  await expect(page.getByRole("button", { name: "Save exchange-rates.csv" })).toBeVisible();
  await expect(page.locator(".counter")).toHaveText("1");
});
