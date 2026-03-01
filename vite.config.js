import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile' // <-- named import

export default defineConfig({
  plugins: [
    viteSingleFile()  // use as a function
  ],
  build: {
    target: 'esnext',
    minify: 'esbuild',
    assetsInlineLimit: 1000000, // inline assets <1MB
  }
})
