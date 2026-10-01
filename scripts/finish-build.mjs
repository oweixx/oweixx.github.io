import { cp, mkdir, writeFile } from 'node:fs/promises';

// publicDir is empty; copy only actual site assets, never legacy or drafts.
await mkdir('dist/assets', { recursive: true });
await cp('assets', 'dist/assets', { recursive: true });
await writeFile('dist/.nojekyll', '');
