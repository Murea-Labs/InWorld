import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Relative asset and API paths, so the same build works at / or under any
// sub-path behind a reverse proxy.
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:8080',
      '/clips': 'http://127.0.0.1:8080',
      '/refs': 'http://127.0.0.1:8080',
    },
  },
})
