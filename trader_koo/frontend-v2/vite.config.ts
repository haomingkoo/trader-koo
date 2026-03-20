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
        manualChunks: {
          plotly: ["plotly.js"],
          react: ["react", "react-dom"],
          router: ["react-router-dom"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
});
