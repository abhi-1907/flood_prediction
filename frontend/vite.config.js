import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],

  server: {
    port: 5173,
    open: true,           // Auto-opens browser on `npm run dev`
    cors: true,

    // Proxy only used if you switch floodApi.js baseURL back to "/api"
    // Currently floodApi.js hits http://localhost:8000 directly, so proxy is
    // kept as a fallback but not actively used.
    proxy: {
      "/api": {
        target:       "http://localhost:8000",
        changeOrigin: true,
        rewrite:      (path) => path.replace(/^\/api/, ""),
      },
    },
  },

  build: {
    outDir:       "dist",
    sourcemap:    false,
    chunkSizeWarningLimit: 800,  // Recharts + Leaflet are naturally large
    rollupOptions: {
      output: {
        // Split vendor chunks for better caching
        manualChunks: {
          react:    ["react", "react-dom", "react-router-dom"],
          charts:   ["recharts"],
          map:      ["leaflet", "react-leaflet"],
          axios:    ["axios"],
        },
      },
    },
  },

  // Let Vite handle Leaflet image assets
  assetsInclude: ["**/*.png", "**/*.jpg", "**/*.svg"],
});
