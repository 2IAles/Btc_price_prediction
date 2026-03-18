import { build } from 'vite';
import react from '@vitejs/plugin-react';

await build({
  plugins: [react()],
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});

console.log('Build complete!');
