export const posts = [
  {
    slug: 'kl-gradient-recovery-trick',
    title: 'The KL Gradient–Recovery Trick',
    date: '2026-11-05',
    tags: ['machine learning', 'research', 'my publications'],
    excerpt:
      'When two distributions come from the same linear output head, the gradient of their forward KL can be recovered from the difference of two cross-entropy gradients without materializing any logits or computing the KL.',
    cover: '/blog/assets/2026-10-27-soft-mtp/soft_mtp_cover_no_titles.png',
    socialImage: '/blog/assets/2026-10-27-soft-mtp/soft_mtp_cover_no_titles.png',
    layout: 'paper',
    // Google Scholar indexing metadata. See src/data/scholar.js.
    scholar: {
      title: 'Multi-Token Prediction Done Right',
      authors: ['Jayden Teoh'],
      institution: 'Massachusetts Institute of Technology',
      arxivId: '2610.38331',
      pdf: 'https://arxiv.org/pdf/2610.38331',
      keywords: [
        'multi-token prediction',
        'self-distillation',
        'language model pretraining',
        'speculative decoding',
        'straight-through estimator',
      ],
    },
    load: () => import('../posts/2026-10-27-soft-mtp.mdx'),
  },
  {
    slug: 'nextlat',
    title: 'Next-Latent Prediction Transformers',
    date: '2026-05-25',
    tags: ['machine learning', 'research', 'my publications'],
    excerpt:
      'NextLat extends standard next-token prediction training with self-supervised predictions in the latent space.',
    cover: '/blog/assets/2026-05-25-nextlat/manhattan_population_webby-final-frame.png',
    socialImage: '/blog/assets/2026-05-25-nextlat/manhattan_population_webby-final-frame.png',
    layout: 'paper',
    scholar: {
      title: 'Next-Latent Prediction Transformers Learn Compact World Models',
      authors: [
        'Jayden Teoh',
        'Manan Tomar',
        'Kwangjun Ahn',
        'Edward S. Hu',
        'Tim Pearce',
        'Pratyusha Sharma',
        'Akshay Krishnamurthy',
        'Riashat Islam',
        'Alex Lamb',
        'John Langford',
      ],
      conference: 'Advances in Neural Information Processing Systems (NeurIPS)',
      arxivId: '2511.05963',
      pdf: 'https://arxiv.org/pdf/2511.05963',
      keywords: [
        'next-latent prediction',
        'world models',
        'representation learning',
        'language model pretraining',
        'speculative decoding',
      ],
    },
    load: () => import('../posts/2026-05-25-nextlat.mdx'),
  },
  {
    slug: 'diffusion-models',
    title: 'Mathematics behind Diffusion Models',
    date: '2025-08-16',
    tags: ['machine learning', 'research'],
    excerpt: 'Personal notes during my journey of understanding diffusion models.',
    cover: '/blog/assets/2025-08-16-diffusion-models/Untitled%203.png',
    load: () => import('../posts/2025-08-16-diffusion-models.mdx'),
  },
]

export function postYear(post) {
  return post.date.slice(0, 4)
}

export function postUrl(post) {
  return `/blog/${postYear(post)}/${post.slug}`
}

export function getPost(year, slug) {
  return posts.find((p) => p.slug === slug && postYear(p) === year)
}

export function getAllTags() {
  const set = new Set()
  for (const p of posts) for (const t of p.tags) set.add(t)
  return ['all', ...Array.from(set)]
}
