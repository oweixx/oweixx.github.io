import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';

export const remarkPlugins = [remarkGfm, remarkMath];
const safeMarkdownSchema = {
  ...defaultSchema,
  attributes: {
    ...defaultSchema.attributes,
    code: [['className', /^language-./, 'math-inline', 'math-display']],
  },
};
export const rehypePlugins = [[rehypeSanitize, safeMarkdownSchema], rehypeKatex, [rehypeHighlight, { detect: false }], rehypeSlug];

// Both the live writer and published posts use these math/code plugins.
export async function previewMarkdown(markdown) {
  return String(await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeSanitize, safeMarkdownSchema)
    .use(rehypeKatex)
    .use(rehypeHighlight, { detect: false })
    .use(rehypeSlug)
    .use(rehypeStringify)
    .process(markdown));
}
