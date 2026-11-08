// Static rendering of each post's MDX body, for readers that never run JS.
//
// The pre-rendered <head> (see prerender.mjs) is enough for Google, which
// executes the SPA's JS, and for the social scrapers that only read meta tags.
// It is NOT enough for anything that does a plain HTTP fetch and reads what
// comes back: Common Crawl's CCBot, GPTBot and ClaudeBot do no rendering, and
// the Wayback Machine archives exactly what it is served. Without this they all
// get an empty <div id="root"> and the article text is absent from the record.
//
// main.jsx mounts with createRoot(), NOT hydrateRoot(), so React clears the
// container on first render instead of trying to match it. This markup
// therefore has no hydration contract to honour — it is a payload for non-JS
// readers that the browser discards a few hundred ms later. It still carries
// the real class names so that hand-off isn't a flash of unstyled text.

import { readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { createElement as h } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { evaluate } from '@mdx-js/mdx'
import * as runtime from 'react/jsx-runtime'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

const postsDir = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'posts')

// Plain-HTML stand-ins for the MDX components, carrying the real class names so
// the markup inherits the stylesheet already parsed in <head>. Interaction-only
// affordances (citation tooltips, the copy button, the benchmark slider) are
// dropped; any TEXT they wrap is kept, because text is the whole point here.
const components = {
  PaperHero: ({ title, kicker, authors, venue }) =>
    h('header', { className: 'paper-hero' }, [
      kicker ? h('div', { key: 'k', className: 'paper-kicker' }, kicker) : null,
      h('h1', { key: 't', className: 'paper-title' }, title),
      authors?.length
        ? h('p', { key: 'a', className: 'paper-authors-line' }, authors.map((a) => a.name).join(', '))
        : null,
      venue ? h('p', { key: 'v', className: 'paper-venue' }, venue) : null,
    ]),

  TOC: ({ items }) =>
    items?.length
      ? h(
          'nav',
          { className: 'paper-toc paper-toc--sidebar', 'aria-label': 'Table of contents' },
          items.map((it) => h('a', { key: it.slug, href: `#${it.slug}` }, it.text)),
        )
      : null,

  Section: ({ id, children }) =>
    h('section', { id, className: 'paper-section paper-prose' }, children),

  TLDR: ({ children }) => h('section', { className: 'paper-tldr' }, children),

  Abstract: ({ children }) =>
    h('section', { className: 'paper-abstract' }, h('div', { className: 'paper-container' }, children)),

  Takeaway: ({ label = 'Key Insight', children }) =>
    h('div', { className: 'paper-takeaway', 'data-label': label }, children),

  Grid: ({ children }) => h('div', { className: 'paper-grid' }, children),

  Figure: ({ src, alt = '', caption }) =>
    h('figure', { className: 'paper-figure' }, [
      h('img', { key: 'i', src, alt: alt || caption || '' }),
      caption ? h('figcaption', { key: 'c' }, caption) : null,
    ]),

  VideoFigure: ({ src, caption }) =>
    h('figure', { className: 'paper-figure' }, [
      h('video', { key: 'v', src, controls: true }),
      caption ? h('figcaption', { key: 'c' }, caption) : null,
    ]),

  BibTeX: ({ note, children }) =>
    h('section', { className: 'paper-bibtex-section' }, [
      h('h2', { key: 'h' }, 'BibTeX'),
      note ? h('p', { key: 'n', className: 'paper-bibtex-note' }, note) : null,
      h(
        'pre',
        { key: 'p', className: 'paper-bibtex' },
        h('code', null, typeof children === 'string' ? children.trim() : children),
      ),
    ]),

  // The footnote body is real prose, so it is kept even though the live
  // component shows it only inside a tooltip.
  Footnote: ({ children }) => h('span', { className: 'paper-footnote-body' }, children),

  // Numbering comes from a DOM pass that cannot run here, and a wrong number is
  // worse than none. The citation TARGETS are already in the JSON-LD and the
  // post's own reference list.
  Cite: () => null,

  // Purely an interactive gallery; its slides are images already linked
  // elsewhere on the page.
  BenchmarkSlider: () => null,
}

// posts.js records each filename only inside its lazy `load` import. Reading it
// back from there keeps one source of truth instead of a second path to drift.
function mdxFileFor(post) {
  const match = /posts\/([^'"]+\.mdx)/.exec(String(post.load))
  if (!match) {
    throw new Error(`prerender-body: no .mdx path in the posts.js entry for "${post.slug}"`)
  }
  return match[1]
}

export async function renderPostBody(post) {
  const path = join(postsDir, mdxFileFor(post))
  const raw = await readFile(path, 'utf8')

  // Node cannot import a stylesheet the way Vite can. The CSS is already in the
  // built bundle and this markup is discarded on mount, so dropping the import
  // here changes nothing a reader ever sees.
  const source = raw.replace(/^[ \t]*import\s+['"][^'"]+\.css['"];?[ \t]*$/gm, '')

  const { default: Content } = await evaluate(source, {
    ...runtime,
    baseUrl: pathToFileURL(path),
    remarkPlugins: [remarkGfm, remarkMath],
    // MathML rather than KaTeX's default span layout. KaTeX's HTML duplicates
    // every character across a visual and an accessible tree, which bloats the
    // page ~6x (the diffusion post: 1.15MB -> 190KB) and extracts as garbled
    // text. MathML carries the same formula compactly and the browser discards
    // all of it on mount anyway.
    rehypePlugins: [[rehypeKatex, { output: 'mathml' }]],
  })

  return renderToStaticMarkup(h(Content, { components }))
}
