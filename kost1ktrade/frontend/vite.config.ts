import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/signal-bot': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/grid-bot': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/balance': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    }
  }
})
