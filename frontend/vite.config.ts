/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    // Docker bind mounts do not emit inotify events reliably on Windows hosts.
    watch: { usePolling: true },
  },
});
