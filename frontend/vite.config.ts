import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  build: {
    // No source maps in production — keeps bundle lean and hides internals
    sourcemap: false,

    // Warn if any chunk exceeds 600 KB (helps catch accidental heavy imports)
    chunkSizeWarningLimit: 600,

    rollupOptions: {
      output: {
        // Split heavy vendor libraries into their own cached chunks.
        // Browsers re-use these across deploys since hashes only change
        // when the library version changes — not when app code changes.
        manualChunks: {
          // React core (~140 KB gzipped)
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          // Framer Motion — heavy animation library (~50 KB gzipped)
          'vendor-framer': ['framer-motion'],
          // Markdown renderer
          'vendor-markdown': ['react-markdown'],
        },
      },
    },
  },

  // Ensure environment variables are validated at build time
  define: {
    __APP_VERSION__: JSON.stringify(process.env.npm_package_version),
  },
})
