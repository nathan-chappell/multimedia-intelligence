import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: "https://localhost:8000", secure: false },
      "/chatkit": { target: "https://localhost:8000", secure: false },
    },
  },
});
