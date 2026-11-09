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
          injectRoute({ pattern: '/theme-preview', entrypoint: './src/dev/ThemePreview.astro' });
          injectRoute({ pattern: '/badge-preview', entrypoint: './src/dev/BadgePreview.astro' });
          injectRoute({ pattern: '/write', entrypoint: './src/dev/Writer.astro' });
          injectRoute({ pattern: '/write/images', entrypoint: './src/dev/images.ts' });
          injectRoute({ pattern: '/write/posts', entrypoint: './src/dev/posts.ts' });
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
        server.middlewares.use('/profile-variants', sirv('output/profile-variants', { dev: true }));
      },
    }],
  },
});
