import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import { remarkGalleries, rehypeImages } from './galleries.mjs';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';

export const remarkPlugins = [remarkGfm, remarkMath, remarkDirective, remarkGalleries];
const safeMarkdownSchema = {
  ...defaultSchema,
  tagNames: [...defaultSchema.tagNames, 'figure', 'figcaption'],
  attributes: {
    ...defaultSchema.attributes,
    code: [['className', /^language-./, 'math-inline', 'math-display']],
    div: [...(defaultSchema.attributes.div ?? []), ['className', 'image-gallery', 'gallery-track']],
    figure: [['className', 'gallery-slide']],
  },
};
export const rehypePlugins = [[rehypeSanitize, safeMarkdownSchema], rehypeImages, rehypeKatex, [rehypeHighlight, { detect: false }], rehypeSlug];

// Preview and publication share the full pipeline, including image galleries.
export async function previewMarkdown(markdown) {
  return String(await unified()
    .use(remarkParse)
    .use(remarkPlugins)
    .use(remarkRehype)
    .use(rehypePlugins)
    .use(rehypeStringify)
    .process(markdown));
}
