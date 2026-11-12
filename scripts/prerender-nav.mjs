// Static markup for the two navigation routes, `/` and `/blog`.
//
// Post routes get their real MDX body (prerender-body.mjs), but `/` and `/blog`
// are React pages, and Node has no JSX transform, so their components cannot be
// imported here. They used to ship an empty <div id="root">, which meant the
// static HTML carried *zero* anchors: nothing linked to any post except the one
// line in sitemap.xml. Googlebot does run JS, but rendering is a separate,
// deprioritised queue, so on a low-authority domain a post reachable only by
// sitemap is the classic "Discovered - currently not indexed" shape.
//
// So we emit just the links: the header nav on both routes, plus the post list
// on /blog. That gives a crawler a real path / -> /blog -> every post without
// executing anything.
//
// This duplicates class names from Header.jsx, Blog.jsx and PostCard.jsx. The
// duplication is deliberate and the markup is kept deliberately small: it is a
// crawl path, not a second renderer. Only what styles identically is included,
// so the frame before React mounts looks like the real page. Two things are
// skipped on purpose - the tag-filter buttons (interactive, and dead without
// JS) and the per-tag colour pills (they would mean copying PostCard's colour
// hash as well, and they are not links, so they buy no crawlability).
//
// Nothing here needs to match what React renders: createRoot() clears the
// container before mounting, so there is no hydration contract to break.

import { posts, postUrl } from '../src/data/posts.js'
import { canonicalPath } from '../src/data/site.js'

const escape = (value) =>
  String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')

// Mirrors Header.jsx, including the home-only `top-bar-home` modifier and the
// brand link it hides on `/`.
function header(path) {
  const isHome = path === '/'
  const navLink = (href, label) =>
    `<a class="nav-link${path === href ? ' nav-link-active' : ''}" href="${escape(
      canonicalPath(href),
    )}">${label}</a>`

  return (
    `<header class="top-bar${isHome ? ' top-bar-home' : ''}">` +
    `<div class="name-block">` +
    (isHome ? '' : `<a class="brand-link" href="/"><p class="brand">Jayden Teoh</p></a>`) +
    `</div>` +
    `<nav class="nav-links">${navLink('/', 'home')}${navLink('/blog', 'blog')}</nav>` +
    `</header>`
  )
}

// Mirrors PostCard.jsx. Hrefs are the canonical trailing-slash form so a
// crawler never has to follow the Pages 301 from the no-slash URL.
function postCard(post) {
  const href = escape(canonicalPath(postUrl(post)))
  return (
    `<article class="post-card">` +
    (post.cover
      ? `<a class="post-cover" href="${href}" aria-hidden="true">` +
        `<img src="${escape(post.cover)}" alt="" loading="lazy" /></a>`
      : '') +
    `<div class="post-info">` +
    `<h3 class="post-title"><a href="${href}">${escape(post.title)}</a></h3>` +
    `<p class="post-date">${escape(post.date)}</p>` +
    (post.excerpt ? `<p class="post-excerpt">${escape(post.excerpt)}</p>` : '') +
    `</div>` +
    `</article>`
  )
}

// Mirrors Blog.jsx's page frame, minus the tag filter.
function blogMain() {
  return (
    `<main class="writings-page">` +
    `<div class="writings-header">` +
    `<h1 class="page-title">blog</h1>` +
    `<p class="page-subtitle">thoughts on machine learning, research, and other topics.</p>` +
    `</div>` +
    `<div class="post-list">${posts.map(postCard).join('')}</div>` +
    `</main>`
  )
}

// Returns the #root contents for a nav route, or null for anything else.
export function renderNavBody(path) {
  if (path === '/') return `<div class="app-shell">${header('/')}</div>`
  if (path === '/blog') return `<div class="app-shell">${header('/blog')}${blogMain()}</div>`
  return null
}
