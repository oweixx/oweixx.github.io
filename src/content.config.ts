import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

const blog = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/blog' }),
  schema: z.object({
    title: z.string().min(1),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    description: z.string().default(''),
    category: z.enum(['Papers', 'Research', 'Notes', 'Life']).default('Notes'),
    tags: z.array(z.string()).default([]),
    slug: z.string().regex(/^[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*$/).optional(),
    draft: z.boolean().default(true),
  }).refine((post) => !post.updated || post.updated >= post.date, {
    message: 'updated must be on or after date',
  }),
});

export const collections = { blog };
