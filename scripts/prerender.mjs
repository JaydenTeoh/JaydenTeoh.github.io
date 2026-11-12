// Static pre-rendering for the GitHub Pages SPA.
//
// A client-rendered SPA on GitHub Pages has two SEO-fatal problems:
//   1. Deep links (e.g. /blog/2026/nextlat) have no real file, so Pages serves
//      the 404.html fallback with an HTTP 404 status -> Google won't index them.
//   2. Meta tags are injected by JS after load, so non-JS scrapers (X, LinkedIn,
//      Slack) only ever see the generic index.html title/description.
//
// This script runs after `vite build`. For every known route it writes a real
// <route>/index.html (200 status) with the correct <title>, description,
// canonical, Open Graph, Twitter, Highwire `citation_*` (Google Scholar), and
// JSON-LD baked into the initial HTML. Post routes additionally get their MDX
// body rendered into #root (see prerender-body.mjs) so that crawlers which
// never execute JS still receive the article text, and the `/` and `/blog`
// routes get their navigation links (see prerender-nav.mjs) so those crawlers
// have a path to the posts at all. It also regenerates sitemap.xml so it can
// never drift from the route list.

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { posts, postYear } from '../src/data/posts.js'
import { scholarTags } from '../src/data/scholar.js'
import { renderPostBody } from './prerender-body.mjs'
import { renderNavBody } from './prerender-nav.mjs'
import {
  buildTitle,
  canonicalPath,
  defaultOgImage,
  siteDescription,
  siteName,
  siteUrl,
} from '../src/data/site.js'

const distDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'dist')

const toAbsolute = (path) => new URL(path, siteUrl).toString()
const escapeAttr = (value) =>
  String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
// Inline JSON-LD: only `<` needs escaping to avoid prematurely closing <script>.
const escapeJsonLd = (obj) => JSON.stringify(obj).replace(/</g, '\\u003c')

// Routes to pre-render. Descriptions/images mirror what each page passes to the
// <SEO> component so the pre-rendered HTML and the hydrated DOM agree.
const routes = [
  { path: '/', title: 'Jayden Teoh', description: siteDescription, image: defaultOgImage },
  {
    path: '/blog',
    title: 'Blog',
    description: "Jayden Teoh's blog about machine learning, research, and writing.",
    image: defaultOgImage,
  },
  ...posts.map((post) => ({
    path: `/blog/${postYear(post)}/${post.slug}`,
    title: post.title,
    description: post.excerpt || `Read ${post.title} by Jayden Teoh.`,
    image: post.socialImage || post.cover || defaultOgImage,
    type: 'article',
    date: post.date,
    tags: post.tags,
    isPost: true,
    post,
  })),
]

// Paths a post used to live at. GitHub Pages serves static files only, so there
// is no way to return a real 301 — without a file here Pages falls back to
// 404.html and the old link is simply dead. A zero-delay meta refresh is what
// Google documents as equivalent to a permanent redirect, so this returns 200,
// sends the crawler on to the new URL, and stops a saved link breaking.
// Deliberately absent from sitemap.xml: you ask Google to follow a redirect,
// never to index one.
const redirects = [
  // renamed 2026-11-06, after one day live
  { from: '/blog/2026/soft-mtp/', to: '/blog/2026/kl-gradient-recovery-trick/' },
]

function metaTags(route) {
  const fullTitle = buildTitle(route.title)
  const canonicalUrl = toAbsolute(canonicalPath(route.path))
  const imageUrl = route.image ? toAbsolute(route.image) : null
  const ogType = route.type || 'website'

  const tags = [
    ['property="og:title"', `<meta property="og:title" content="${escapeAttr(fullTitle)}" />`],
    [
      'property="og:description"',
      `<meta property="og:description" content="${escapeAttr(route.description)}" />`,
    ],
    ['property="og:type"', `<meta property="og:type" content="${ogType}" />`],
    ['property="og:url"', `<meta property="og:url" content="${escapeAttr(canonicalUrl)}" />`],
    [
      'name="twitter:card"',
      `<meta name="twitter:card" content="${imageUrl ? 'summary_large_image' : 'summary'}" />`,
    ],
    ['name="twitter:title"', `<meta name="twitter:title" content="${escapeAttr(fullTitle)}" />`],
    [
      'name="twitter:description"',
      `<meta name="twitter:description" content="${escapeAttr(route.description)}" />`,
    ],
    ['rel="canonical"', `<link rel="canonical" href="${escapeAttr(canonicalUrl)}" />`],
  ]
  if (imageUrl) {
    tags.push([
      'property="og:image"',
      `<meta property="og:image" content="${escapeAttr(imageUrl)}" />`,
    ])
    tags.push([
      'name="twitter:image"',
      `<meta name="twitter:image" content="${escapeAttr(imageUrl)}" />`,
    ])
  }

  // Highwire Press tags, for Google Scholar. Scholar does not run JS, so these
  // only work baked in here. Posts without a `scholar` block emit none.
  const scholar = route.post?.scholar
  for (const { name, content } of scholarTags(route.post, { canonicalUrl, toAbsolute })) {
    tags.push([`name="${name}"`, `<meta name="${name}" content="${escapeAttr(content)}" />`])
  }

  // A post about one of my own papers is a ScholarlyArticle, not a BlogPosting:
  // it gets the real author list, the paper title, and a link to the PDF.
  const identifiers = []
  if (scholar?.arxivId) identifiers.push(`arXiv:${scholar.arxivId}`)
  if (scholar?.doi) identifiers.push(`https://doi.org/${scholar.doi}`)
  const pdfUrl = scholar?.pdf
    ? /^https?:\/\//.test(scholar.pdf)
      ? scholar.pdf
      : toAbsolute(scholar.pdf)
    : null

  const jsonLd = route.isPost
    ? {
        '@context': 'https://schema.org',
        '@type': scholar ? 'ScholarlyArticle' : 'BlogPosting',
        headline: scholar?.title || route.title,
        description: route.description,
        image: imageUrl,
        datePublished: route.date,
        dateModified: route.date,
        author: scholar?.authors?.length
          ? scholar.authors.map((name) => ({ '@type': 'Person', name }))
          : { '@type': 'Person', name: siteName, url: siteUrl },
        publisher: { '@type': 'Person', name: siteName },
        mainEntityOfPage: canonicalUrl,
        ...(scholar?.keywords?.length
          ? { keywords: scholar.keywords.join(', ') }
          : route.tags?.length
            ? { keywords: route.tags.join(', ') }
            : {}),
        ...(identifiers.length ? { identifier: identifiers } : {}),
        ...(scholar?.arxivId ? { sameAs: `https://arxiv.org/abs/${scholar.arxivId}` } : {}),
        ...(pdfUrl
          ? {
              encoding: {
                '@type': 'MediaObject',
                encodingFormat: 'application/pdf',
                contentUrl: pdfUrl,
              },
            }
          : {}),
      }
    : null

  return { fullTitle, head: tags.map(([, html]) => html), jsonLd }
}

// Inject route-specific metadata into the built index.html template, plus the
// statically rendered article body when there is one.
function render(template, route, body = null) {
  const { fullTitle, head, jsonLd } = metaTags(route)
  const extra = [...head]
  if (jsonLd) {
    extra.push(`<script type="application/ld+json">${escapeJsonLd(jsonLd)}</script>`)
  }

  // A function replacement, because article text containing `$&` or `$1` would
  // otherwise be interpreted as a replacement pattern and silently mangled.
  const withBody = body
    ? template.replace('<div id="root"></div>', () => `<div id="root">${body}</div>`)
    : template

  return withBody
    .replace(/<title>[\s\S]*?<\/title>/, `<title>${escapeAttr(fullTitle)}</title>`)
    .replace(
      /<meta\s+name="description"[\s\S]*?\/>/,
      `<meta name="description" content="${escapeAttr(route.description)}" />`,
    )
    .replace(
      /<meta\s+name="robots"[^>]*\/>/,
      `<meta name="robots" content="${route.noindex ? 'noindex,nofollow' : 'index,follow'}" />`,
    )
    .replace('</head>', `    ${extra.join('\n    ')}\n  </head>`)
}

async function writeRoute(template, route) {
  // Posts get their full article text; `/` and `/blog` get their links, which
  // is the part of a navigation page a crawler needs. 404.html passes through
  // with an empty #root, since there is nothing there to link to.
  const body = route.isPost ? await renderPostBody(route.post) : renderNavBody(route.path)
  const html = render(template, route, body)
  const outPath =
    route.path === '/'
      ? join(distDir, 'index.html')
      : join(distDir, route.path, 'index.html')
  await mkdir(dirname(outPath), { recursive: true })
  await writeFile(outPath, html)
  return outPath
}

function redirectHtml(target) {
  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Moved</title>
    <link rel="canonical" href="${escapeAttr(target)}" />
    <meta http-equiv="refresh" content="0; url=${escapeAttr(target)}" />
  </head>
  <body>
    <p>This page has moved to <a href="${escapeAttr(target)}">${escapeAttr(target)}</a>.</p>
  </body>
</html>
`
}

function buildSitemap() {
  const entries = routes
    .map((route) => {
      const loc = toAbsolute(canonicalPath(route.path))
      const lastmod = route.date ? `\n    <lastmod>${route.date}</lastmod>` : ''
      return `  <url>\n    <loc>${loc}</loc>${lastmod}\n  </url>`
    })
    .join('\n')
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${entries}\n</urlset>\n`
}

async function main() {
  const template = await readFile(join(distDir, 'index.html'), 'utf8')

  for (const route of routes) {
    const outPath = await writeRoute(template, route)
    console.log(`prerendered ${route.path} -> ${outPath.replace(distDir, 'dist')}`)
  }

  // 404.html: real 404 status for genuinely missing paths, kept noindex, and
  // still boots the SPA so a user mistyping a URL lands in the app.
  const notFoundHtml = render(template, {
    path: '/404',
    title: 'Page not found',
    description: 'This page could not be found.',
    image: defaultOgImage,
    noindex: true,
  })
  await writeFile(join(distDir, '404.html'), notFoundHtml)
  console.log('wrote dist/404.html (noindex SPA fallback)')

  for (const { from, to } of redirects) {
    const target = toAbsolute(canonicalPath(to))
    const outPath = join(distDir, from, 'index.html')
    await mkdir(dirname(outPath), { recursive: true })
    await writeFile(outPath, redirectHtml(target))
    console.log(`redirect ${from} -> ${to}`)
  }

  await writeFile(join(distDir, 'sitemap.xml'), buildSitemap())
  console.log(`wrote dist/sitemap.xml (${routes.length} urls)`)
}

main().catch((err) => {
  console.error('prerender failed:', err)
  process.exit(1)
})
