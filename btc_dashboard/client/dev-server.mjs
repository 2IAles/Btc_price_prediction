import { createServer } from 'vite';
import react from '@vitejs/plugin-react';

const server = await createServer({
  plugins: [react()],
  server: {
    port: 3002,
    proxy: {
      '/api': {
        target: 'http://localhost:3003',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:3003',
        ws: true,
      },
    },
  },
});

await server.listen();
server.printUrls();
