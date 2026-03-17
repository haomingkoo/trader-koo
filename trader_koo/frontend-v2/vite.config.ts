import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/v2/",
  server: {
    port: 3000,
    proxy: {
      "/api": "http://localhost:8000",
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: {
    outDir: "../../dist-v2",
    emptyOutDir: true,
    sourcemap: true,
  },
});
