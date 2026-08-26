import { expect, test } from "@playwright/test";

test.use({ viewport: { width: 390, height: 844 } });

test("keeps account controls and ledger attribution usable on mobile", async ({ page }) => {
  await page.addInitScript(() =>
    window.localStorage.setItem("e2e_clerk_token", "test-token"),
  );
  await page.route("**/api/auth/me", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id: "user_mobile",
        username: "mobile-user",
        email: "mobile@example.com",
        full_name: "Mobile User",
        role: "user",
        is_admin: false,
        balance_microusd: 2_500_000,
      }),
    }),
  );
  await page.route("**/api/billing/ledger", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        balance_microusd: 2_499_678,
        total: 1,
        items: [{
          id: "event_1",
          user_id: "user_mobile",
          amount_microusd: -322,
          event_type: "agent_model_usage",
          description: "Agent model usage",
          actor_user_id: null,
          thread_id: "thread_mobile",
          provider_request_id: "req_mobile_123456789",
          provider_response_id: "resp_mobile_123456789",
          trace_id: "trace_mobile",
          agent_span_id: "span_mobile",
          metadata: { model: "gpt-5.6-luna", input_tokens: 42, output_tokens: 11 },
          created_at: "2026-08-26T01:00:00Z",
        }],
      }),
    }),
  );

  await page.goto("/account");

  await expect(page.getByRole("heading", { name: "Redeem coupon" })).toBeVisible();
  await expect(page.getByLabel("Coupon code")).toBeVisible();
  await expect(page.getByRole("button", { name: "Apply code" })).toBeVisible();
  await expect(page.getByRole("columnheader")).toHaveCount(0);

  await page.locator(".attribution-preview summary").click();
  const preview = page.getByText("Event attribution").locator("..");
  await expect(preview).toContainText("resp_mobile_123456789");
  await expect(preview).toContainText("gpt-5.6-luna");
  await expect(page.getByRole("link", { name: /OpenAI response retrieval reference/ })).toBeVisible();

  const bodyWidth = await page.locator("body").evaluate((element) => element.scrollWidth);
  expect(bodyWidth).toBeLessThanOrEqual(390);
});
