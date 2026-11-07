import { cp, mkdir, writeFile } from 'node:fs/promises';

// Keep the existing /assets/... URLs in the static build.
await mkdir('dist/assets', { recursive: true });
await cp('assets', 'dist/assets', { recursive: true });
await writeFile('dist/.nojekyll', '');
