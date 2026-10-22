import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// VITE_BASE_PATH lets GitHub Pages serve the app from /<repo>/.
// Vercel, Netlify and local development use "/" by default.
const rawBase = process.env.VITE_BASE_PATH || "/";
const base = rawBase.endsWith("/") ? rawBase : `${rawBase}/`;

export default defineConfig({
  plugins: [react()],
  base,
});
