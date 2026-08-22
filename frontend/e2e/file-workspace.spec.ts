import { expect, test } from "@playwright/test";

test("stages a conversation file without claiming it is durable", async ({ page }) => {
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
  await expect(page.getByText(/Local staging is intentionally not durable/)).toBeVisible();
  await expect(page.locator(".counter")).toHaveText("1");
});
