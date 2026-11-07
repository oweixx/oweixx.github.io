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
const safeHtmlPlugins = [[rehypeSanitize, safeMarkdownSchema], rehypeImages];
const formattingPlugins = [rehypeKatex, [rehypeHighlight, { detect: false }], rehypeSlug];
export const rehypePlugins = [...safeHtmlPlugins, ...formattingPlugins];

function rememberPreviewSource() {
  return (tree, file) => {
    // KaTeX replaces math nodes, but keeps their place among the root blocks.
    file.data.previewPositions = tree.children.map(node => node.position);
  };
}

// Add trusted source offsets after sanitizing, only in the local writer.
function rehypePreviewSource() {
  return (tree, file) => {
    tree.children.forEach((node, index) => {
      const position = node.position ?? file.data.previewPositions[index];
      if (node.type !== 'element' || !position) return;
      node.properties['data-source-start'] = position.start.offset;
      node.properties['data-source-end'] = position.end.offset;
    });
  };
}

// Preview and publication share the full pipeline, including image galleries.
export async function previewMarkdown(markdown, { sourceMap = false } = {}) {
  const processor = unified()
    .use(remarkParse)
    .use(remarkPlugins)
    .use(remarkRehype)
    .use(safeHtmlPlugins);
  if (sourceMap) processor.use(rememberPreviewSource);
  processor.use(formattingPlugins);
  if (sourceMap) processor.use(rehypePreviewSource);
  return String(await processor
    .use(rehypeStringify)
    .process(markdown));
}
