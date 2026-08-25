import { expect, test, type Page } from "@playwright/test";

const liveBaseUrl = process.env.LIVE_E2E_BASE_URL;
const demoEnabled = process.env.RUN_DEMO_E2E === "1" && Boolean(liveBaseUrl);

test.use({ ignoreHTTPSErrors: true });

test("live seeded collections complete all three demo scenarios", async ({ page }) => {
  test.skip(
    !demoEnabled,
    "Set RUN_DEMO_E2E=1 and LIVE_E2E_BASE_URL after multimedia-demo seed",
  );
  test.setTimeout(10 * 60_000);
  const transportErrors: string[] = [];
  page.on("requestfailed", (request) => {
    if (request.url().endsWith("/chatkit")) {
      transportErrors.push(`ChatKit request failed: ${request.failure()?.errorText ?? "unknown"}`);
    }
  });
  page.on("response", (response) => {
    if (response.url().endsWith("/chatkit") && response.status() >= 400) {
      transportErrors.push(`ChatKit returned ${response.status()}`);
    }
  });

  await login(page);
  const chat = page.frameLocator("iframe");
  const composer = chat.getByRole("textbox", {
    name: "Ask about this conversation's files…",
  });

  await chooseCollection(page, "Language Trends");
  await composer.fill(
    "Chart global usage for TypeScript, Python, Rust, and Java from 2021–2025, then chart US " +
      "median compensation. Report sample sizes and methodology caveats; do not imply causality.",
  );
  await composer.press("Enter");
  await expect(chat.locator('img[src^="data:image/png;base64,"]').first()).toBeVisible({
    timeout: 180_000,
  });
  await expect(chat.getByText(/respondent|sample size/i).last()).toBeVisible({ timeout: 180_000 });

  await chooseCollection(page, "Type Systems");
  await composer.fill(
    "Explain TypeScript structural typing, generic constraints, conditional types, and mapped " +
      "types using this collection. Separate direct sources from synthesis and cite TAPL pages.",
  );
  await composer.press("Enter");
  await expect(chat.getByText(/structural typing/i).last()).toBeVisible({ timeout: 180_000 });
  await expect(chat.getByText(/TAPL|Types and Programming Languages/i).last()).toBeVisible();

  await chooseCollection(page, "ML Foundations");
  await composer.fill(
    "Compare Transformer, BERT, GPT-3, and LoRA by architecture, objective, adaptation, and " +
      "efficiency. Cite each paper and state synthesis limitations.",
  );
  await composer.press("Enter");
  await expect(chat.getByText(/LoRA/i).last()).toBeVisible({ timeout: 180_000 });
  await expect(chat.getByText(/BERT/i).last()).toBeVisible();

  expect(transportErrors).toEqual([]);
  await expect(chat.getByText(/missing function call/i)).toHaveCount(0);
});

async function login(page: Page): Promise<void> {
  await page.goto(`${liveBaseUrl}/login`);
  await page.getByLabel("Username").fill(process.env.ADMIN_USERNAME ?? "admin");
  await page.getByLabel("Password").fill(process.env.ADMIN_PASSWORD ?? "admin");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(`${liveBaseUrl}/`);
}

async function chooseCollection(page: Page, name: string): Promise<void> {
  await page.getByLabel("Active collection").selectOption({ label: name });
  await expect(page.getByRole("status")).toContainText("Collection selected");
}
