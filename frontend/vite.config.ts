import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/analytics-summary': 'http://127.0.0.1:8000',
      '/customers': 'http://127.0.0.1:8000',
      '/predict': 'http://127.0.0.1:8000',
      '/predict-batch': 'http://127.0.0.1:8000',
      '/segment': 'http://127.0.0.1:8000',
      '/analyze-customer': 'http://127.0.0.1:8000',
    },
  },
})
