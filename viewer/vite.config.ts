import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  publicDir: process.env.VITE_PUBLIC_DIR || "public",
  // Relative asset URLs work both at localhost and under a project Pages path.
  base: "./",
  server: { host: "127.0.0.1", port: 8000 },
  preview: { host: "127.0.0.1", port: 8000 },
  test: { environment: "jsdom", setupFiles: "./src/testSetup.ts" },
});
