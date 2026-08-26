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
  await page.route("**/api/billing/ledger/event_1/attribution", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        event: {
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
        },
        provider_response: {
          id: "resp_mobile_123456789",
          status: "completed",
          model: "gpt-5.6-luna",
          usage: { input_tokens: 42, output_tokens: 11, total_tokens: 53 },
          output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "Done" }] }],
        },
      }),
    }),
  );
  await page.route("**/api/billing/ledger?**", (route) => {
    const offset = Number(new URL(route.request().url()).searchParams.get("offset") ?? 0);
    const firstEvent = {
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
    };
    const items = offset === 0
      ? [firstEvent, ...Array.from({ length: 9 }, (_, index) => ({
          ...firstEvent,
          id: `event_${index + 2}`,
          description: `Earlier event ${index + 2}`,
          provider_request_id: null,
          provider_response_id: null,
        }))]
      : [{ ...firstEvent, id: "event_11", description: "Oldest event" }];
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        balance_microusd: 2_499_678,
        total: 11,
        limit: 10,
        offset,
        items,
      }),
    });
  });

  await page.goto("/account");

  await expect(page.getByRole("heading", { name: "Redeem coupon" })).toBeVisible();
  await expect(page.getByLabel("Coupon code")).toBeVisible();
  await expect(page.getByRole("button", { name: "Apply code" })).toBeVisible();
  await expect(page.getByRole("columnheader")).toHaveCount(0);

  const firstLedgerEvent = page.getByRole("listitem").filter({ hasText: "Agent model usage" });
  await firstLedgerEvent.locator(".attribution-preview summary").click();
  const preview = firstLedgerEvent.locator(".attribution-card");
  await expect(preview).toContainText("resp_mobile_123456789");
  await expect(preview).toContainText("gpt-5.6-luna");
  await expect(preview).toContainText("Retrieved OpenAI response");
  await expect(preview).toContainText('"status": "completed"');
  await expect(page.getByRole("link", { name: /Responses retrieve API reference/ })).toBeVisible();

  await page.getByText("Available credit").click();
  await expect(preview).toBeHidden();

  await expect(page.getByText("Page 1 of 2")).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("Oldest event")).toBeVisible();
  await expect(page.getByText("Page 2 of 2")).toBeVisible();

  const bodyWidth = await page.locator("body").evaluate((element) => element.scrollWidth);
  expect(bodyWidth).toBeLessThanOrEqual(390);
});
