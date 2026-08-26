import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

const envDir = fileURLToPath(new URL("..", import.meta.url));
const requiredFrontendEnv = ["VITE_CLERK_PUBLISHABLE_KEY"] as const;

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, envDir, "VITE_");
  const missing = requiredFrontendEnv.filter((name) => !env[name]?.trim());

  if (missing.length > 0) {
    throw new Error(
      `Missing required frontend environment variable${missing.length === 1 ? "" : "s"}: ${missing.join(", ")}. Add ${missing.length === 1 ? "it" : "them"} to the repository-root .env file before starting or building the app.`,
    );
  }

  return {
    envDir,
    plugins: [react()],
    server: {
      allowedHosts: [".trycloudflare.com"],
      proxy: {
        "/api": { target: "https://localhost:8000", secure: false },
        "/chatkit": { target: "https://localhost:8000", secure: false },
      },
    },
  };
});
