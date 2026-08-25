import { expect, test } from "@playwright/test";

const liveBaseUrl = process.env.LIVE_E2E_BASE_URL;
const liveEnabled = process.env.RUN_OPENAI_E2E === "1" && Boolean(liveBaseUrl);

test.use({ ignoreHTTPSErrors: true });

test("live agent reads a staged file through the browser tool loop", async ({ page }) => {
  test.skip(!liveEnabled, "Set RUN_OPENAI_E2E=1 and LIVE_E2E_BASE_URL to run the live agent E2E");
  test.setTimeout(120_000);

  const transportErrors: string[] = [];
  let rejectTransportFailure: (error: Error) => void = () => undefined;
  const transportFailure = new Promise<never>((_resolve, reject) => {
    rejectTransportFailure = reject;
  });
  void transportFailure.catch(() => undefined);
  const reportTransportFailure = (message: string) => {
    transportErrors.push(message);
    rejectTransportFailure(new Error(message));
  };
  let activeChatKitRequests = 0;
  page.on("request", (request) => {
    if (request.url().endsWith("/chatkit")) activeChatKitRequests += 1;
  });
  const finishChatKitRequest = (request: { url(): string }) => {
    if (request.url().endsWith("/chatkit")) activeChatKitRequests -= 1;
  };
  page.on("requestfinished", finishChatKitRequest);
  page.on("requestfailed", (request) => {
    finishChatKitRequest(request);
    if (request.url().endsWith("/chatkit")) {
      reportTransportFailure(`ChatKit request failed: ${request.failure()?.errorText ?? "unknown"}`);
    }
  });
  page.on("response", (response) => {
    if (!response.url().endsWith("/chatkit")) return;
    if (response.status() >= 400) reportTransportFailure(`ChatKit returned ${response.status()}`);
    void response
      .finished()
      .then(async (failure) => {
        if (failure) {
          reportTransportFailure(`ChatKit stream failed: ${failure}`);
          return;
        }
        const body = await response.text();
        if (/data:\s*\{[^\n]*"type":"error"/.test(body)) {
          reportTransportFailure("ChatKit stream emitted an error event");
        }
      })
      .catch((error: unknown) => {
        reportTransportFailure(
          `Could not inspect ChatKit stream: ${error instanceof Error ? error.message : String(error)}`,
        );
      });
  });

  await page.goto(`${liveBaseUrl}/login`);
  await page.getByLabel("Username").fill(process.env.ADMIN_USERNAME ?? "admin");
  await page.getByLabel("Password").fill(process.env.ADMIN_PASSWORD ?? "admin");
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL(`${liveBaseUrl}/`);

  await page.locator('input[type="file"]').setInputFiles({
    name: "agent-connection-check.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("# Connection check\nThe project codename is SILVER HARBOR.\n"),
  });

  const chat = page.frameLocator("iframe");
  const composer = chat.getByRole("textbox", {
    name: "Ask about this conversation's files…",
  });
  await composer.fill(
    "Read the staged file and tell me its project codename. Use the file tools; do not guess.",
  );
  await composer.press("Enter");

  await Promise.race([
    expect(chat.getByText(/SILVER HARBOR/i)).toBeVisible({ timeout: 90_000 }),
    transportFailure,
  ]);
  await expect.poll(() => activeChatKitRequests, { timeout: 90_000 }).toBe(0);
  expect(transportErrors).toEqual([]);
  await expect(chat.getByText(/missing function call/i)).toHaveCount(0);
});
