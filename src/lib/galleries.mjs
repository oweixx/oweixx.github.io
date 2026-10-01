const element = (tagName, properties, children) => ({ type: 'element', tagName, properties, children });
const text = (value) => ({ type: 'text', value });

// Generate structured HTML before sanitizing, using only Markdown images.
export function remarkGalleries() {
  return (tree, file) => {
    const definitions = new Map();
    function collect(node) {
      if (node.type === 'definition') definitions.set(node.identifier.toLowerCase(), node);
      node.children?.forEach(collect);
    }
    collect(tree);
    function transform(node) {
      if (node.type !== 'containerDirective' || node.name !== 'gallery') {
        node.children?.forEach(transform);
        return;
      }
      const images = [];
      for (const paragraph of node.children) {
        if (paragraph.type !== 'paragraph') file.fail('갤러리 안에는 Markdown 이미지와 빈 줄만 넣어주세요.', paragraph);
        for (const child of paragraph.children) {
          if (child.type === 'text' && !child.value.trim() || child.type === 'break') continue;
          if (child.type === 'image') images.push(child);
          else if (child.type === 'imageReference' && definitions.has(child.identifier.toLowerCase())) {
            images.push({ ...definitions.get(child.identifier.toLowerCase()), alt: child.alt });
          } else file.fail('갤러리 안에는 Markdown 이미지만 넣어주세요: ![설명](이미지 주소)', child);
        }
      }
      if (!images.length) file.fail('갤러리에 이미지를 한 장 이상 넣어주세요.', node);
      node.data = {
        hName: 'div', hProperties: { className: ['image-gallery'] },
        hChildren: [element('div', { className: ['gallery-track'] }, images.map((image) => {
          const caption = image.title || image.alt || '';
          return element('figure', { className: ['gallery-slide'] }, [
            element('img', { src: image.url, alt: image.alt || '', ...(image.title ? { title: image.title } : {}) }, []),
            ...(caption ? [element('figcaption', {}, [text(caption)])] : []),
          ]);
        }))],
      };
      node.children = [];
    }
    transform(tree);
  };
}

// Ordinary image titles become captions; alt text remains available to readers.
export function rehypeImages() {
  return (tree) => {
    function transform(node) {
      if (node.type === 'element' && node.tagName === 'img') {
        node.properties.loading = 'lazy';
        node.properties.decoding = 'async';
      }
      if (node.type === 'element' && node.tagName === 'p') {
        const children = node.children.filter((child) => child.type !== 'text' || child.value.trim());
        if (children.length === 1 && children[0].tagName === 'img' && children[0].properties.title) {
          node.tagName = 'figure';
          node.properties = { className: ['post-image'] };
          node.children = [children[0], element('figcaption', {}, [text(String(children[0].properties.title))])];
        }
      }
      node.children?.forEach(transform);
    }
    transform(tree);
  };
}
