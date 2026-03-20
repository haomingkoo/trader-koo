import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
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
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("plotly")) return "plotly";
          if (id.includes("react-dom") || id.includes("/react/")) return "react";
          if (id.includes("react-router")) return "router";
          if (id.includes("@tanstack/react-query")) return "query";
        },
      },
    },
  },
});
