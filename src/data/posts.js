export const posts = [
  {
    slug: 'nextlat',
    title: 'Next-Latent Prediction Transformers',
    date: '2026-05-25',
    tags: ['machine learning', 'research', 'my publications'],
    excerpt:
      'NextLat extends standard next-token prediction training with self-supervised predictions in the latent space.',
    cover: '/blog/assets/2026-05-25-nextlat/manhattan_population_webby.gif',
    socialImage: '/blog/assets/2026-05-25-nextlat/manhattan_population_webby-final-frame.png',
    layout: 'paper',
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
