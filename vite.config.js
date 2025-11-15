import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/instruct-mix2mix/', // keep this for repo-based GitHub Pages
})

