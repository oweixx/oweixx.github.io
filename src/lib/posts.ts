import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'blog'>;
export const formatDate = (date: Date) => date.toISOString().slice(0, 10).replaceAll('-', '.');
export const postSlug = (post: Post) => post.data.slug ?? post.id.replace(/^\d{4}-\d{2}-\d{2}-/, '');
export const postUrl = (post: Post) => `/blog/${post.data.date.getUTCFullYear()}/${postSlug(post)}/`;

export async function getPosts() {
  const posts = (await getCollection('blog', ({ data }) => import.meta.env.DEV || !data.draft))
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf() || a.id.localeCompare(b.id));
  const paths = new Set<string>();
  for (const post of posts) {
    const path = postUrl(post);
    if (paths.has(path)) throw new Error(`Duplicate blog URL: ${path}`);
    paths.add(path);
  }
  return posts;
}
