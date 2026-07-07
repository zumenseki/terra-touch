import { defineConfig } from 'vite';

export default defineConfig({
  base: './', // 相対パス = GitHub Pages のサブパスでもそのまま動く
  server: { port: 5173, strictPort: true, host: true },
  build: {
    rollupOptions: {
      input: {
        main: './index.html', // サンドボックス
        geo: './geo.html',     // 実写・富士
      },
    },
  },
});
