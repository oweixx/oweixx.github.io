import { defineConfig } from 'astro/config';
import { remarkPlugins, rehypePlugins } from './src/lib/markdown.mjs';
import sirv from 'sirv';
import { unified as markdownProcessor } from '@astrojs/markdown-remark';

export default defineConfig({
  site: 'https://oweixx.github.io',
  trailingSlash: 'always',
  publicDir: './public',
  devToolbar: { enabled: false },
  markdown: {
    syntaxHighlight: false,
    processor: markdownProcessor({
      remarkPlugins, rehypePlugins,
      smartypants: false,
      remarkRehype: { allowDangerousHtml: false },
    }),
  },
  integrations: [{
    name: 'local-writer',
    hooks: {
      'astro:config:setup': ({ command, injectRoute }) => {
        if (command === 'dev') {
          injectRoute({ pattern: '/write', entrypoint: './src/dev/Writer.astro' });
        }
      },
    },
  }],
  // Keep the existing /assets/... addresses and original assets in place.
  vite: {
    plugins: [{
      name: 'existing-assets',
      configureServer(server) {
        server.middlewares.use('/assets', sirv('assets', { dev: true }));
      },
    }],
  },
});
