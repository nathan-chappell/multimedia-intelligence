import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { createServer } from "vite";

const frontendRoot = dirname(dirname(fileURLToPath(import.meta.url)));
process.env.VITE_E2E_AUTH = "true";
process.env.VITE_CLERK_PUBLISHABLE_KEY ??= "pk_test_dGVzdC5jbGVyay5hY2NvdW50cy5kZXYk";
const playwrightCli = join(
  frontendRoot,
  "node_modules",
  "@playwright",
  "test",
  "cli.js",
);

const server = await createServer({
  root: frontendRoot,
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});

let exitCode = 1;

try {
  await server.listen();

  exitCode = await new Promise((resolve, reject) => {
    const child = spawn(
      process.execPath,
      [playwrightCli, "test", ...process.argv.slice(2)],
      {
        cwd: frontendRoot,
        env: {
          ...process.env,
          PLAYWRIGHT_MANAGED_SERVER: "1",
        },
        stdio: "inherit",
      },
    );

    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (signal) {
        reject(new Error(`Playwright exited after receiving ${signal}`));
        return;
      }
      resolve(code ?? 1);
    });
  });
} finally {
  await server.close();
}

process.exitCode = exitCode;
