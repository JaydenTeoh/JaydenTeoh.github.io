export const posts = [
  {
    slug: '2025-08-16-diffusion-models',
    title: 'Mathematics behind Diffusion Models',
    date: '2025-08-16',
    tags: ['machine learning', 'research'],
    excerpt: 'Personal notes during my journey of understanding diffusion models.',
    cover: '/blog/assets/2025-08-16-diffusion-models/Untitled%203.png',
    load: () => import('../posts/2025-08-16-diffusion-models.mdx'),
  },
]

export function getPost(slug) {
  return posts.find((p) => p.slug === slug)
}

export function getAllTags() {
  const set = new Set()
  for (const p of posts) for (const t of p.tags) set.add(t)
  return ['all', ...Array.from(set)]
}
