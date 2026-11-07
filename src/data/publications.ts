export const publications = [
  {
    id: 'edgar',
    title: 'EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure',
    image: '/assets/edgar.gif', imageAlt: 'Animated comparison of the original and EDGAR-refined Gaussian avatar',
    authors: [
      { name: 'Kwan Yun', url: 'https://kwanyun.github.io/' },
      { name: 'MinHyeok Bang', url: 'https://oweixx.github.io/', me: true },
      { name: 'Colette Suhjung Koo', url: 'https://sj0414.github.io/colette-koo/' },
      { name: 'Junyong Noh', url: 'https://vml.kaist.ac.kr/main/people/person/1' },
    ],
    venue: 'ACM Transactions on Graphics', year: 2026, note: 'Accepted', impactFactor: '13.0',
    paper: 'https://arxiv.org/abs/2611.07909', website: 'https://kwanyun.github.io/edgar/#top', code: 'https://github.com/kwanyun/edgar-code/',
    summary: 'We use identity-guided diffusion to repair Gaussian avatar renderings, then transfer these improvements back into a single 3D avatar for fast rendering.',
    // Source: https://github.com/kwanyun/edgar-code#citation (2026-11-08)
    bibtex: `@misc{yun2026edgar,
  title={EDGAR: Efficient Diffusion-Driven Gaussian Avatar Refinement via Synthetic Failure},
  author={Kwan Yun and Minhyeok Bang and Colette Suhjung Koo and Junyong Noh},
  year={2026},
  eprint={2611.07909},
  archivePrefix={arXiv},
  primaryClass={cs.GR},
  url={https://arxiv.org/abs/2611.07909},
}`,
  },
  {
    id: 'geometry-aware-style-transfer', title: 'Geometry-Aware Style Transfer in 3D Gaussian Splatting',
    image: '/assets/geometry.jpeg', imageAlt: 'Geometry-aware style transfer results in 3D Gaussian Splatting',
    authors: [
      { name: 'Min Hyeok Bang', url: 'https://oweixx.github.io/', me: true, equal: true },
      { name: 'Jun Hyeong Kim', url: 'https://kjun627.github.io/', equal: true },
      { name: 'Seung Wook Kim', url: 'https://pknu-vlab.github.io/' },
      { name: 'Se Ho Lee', url: 'https://vilab.jbnu.ac.kr/' },
    ],
    venue: 'European Conference on Computer Vision (ECCV)', year: 2026, note: '', impactFactor: '',
    paper: 'https://arxiv.org/abs/2606.24144', website: 'https://kjun627.github.io/gast-web/', code: 'https://github.com/oweixx/gast',
    summary: 'We transfer both appearance and geometry from a style image to a 3D Gaussian scene by alternating color and geometry updates with RGB, depth, and edge guidance.',
    // Source: https://github.com/oweixx/gast#citation (2026-11-08)
    bibtex: `@inproceedings{bang2026gast,
  title = {Geometry-aware style transfer in 3D gaussian splatting},
  author = {Bang, Min Hyeok and Kim, Jun Hyeong and Kim, Seung-Wook and Lee, Se-Ho},
  booktitle = {Proc. European Conference on Computer Vision (ECCV)},
  address = {Malmö, Sweden},
  year = {2026}
}`,
  },
];
