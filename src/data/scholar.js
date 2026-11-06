// Google Scholar indexing metadata (Highwire Press `citation_*` tags).
//
// Scholar's crawler does not execute JavaScript and ignores JSON-LD, so these
// tags only count if they are in the HTML the server returns. scripts/prerender.mjs
// bakes them into each post's index.html; SEO.jsx re-syncs them on client-side
// navigation so the live DOM never carries another post's citation.
//
// Scholar also requires the PDF it is pointed at to be freely readable with no
// login, and the landing page to hold at least the abstract.
// Tag reference: https://scholar.google.com/intl/en/scholar/inclusion.html#indexing

// Scholar accepts YYYY, YYYY/MM or YYYY/MM/DD. Post dates are ISO (YYYY-MM-DD).
export function scholarDate(isoDate) {
  return String(isoDate ?? '').replace(/-/g, '/')
}

// Flat {name, content} pairs. Rendered as <meta> by both the prerenderer and
// the client, so this stays free of DOM and HTML-string concerns.
//
// `toAbsolute` resolves a site-relative path against the site origin;
// `canonicalUrl` is the already-canonicalised absolute URL of the post.
export function scholarTags(post, { canonicalUrl, toAbsolute }) {
  const scholar = post?.scholar
  if (!scholar) return []

  const tags = []
  const add = (name, content) => {
    if (content) tags.push({ name, content })
  }
  const absolute = (url) => (/^https?:\/\//.test(url) ? url : toAbsolute(url))

  add('citation_title', scholar.title || post.title)
  // Repeated once per author, in author order. Scholar accepts "First Last".
  for (const author of scholar.authors || []) add('citation_author', author)
  add('citation_publication_date', scholar.date || scholarDate(post.date))
  // Exactly one venue tag: a conference paper and a technical report are
  // different records to Scholar, so posts set whichever applies.
  add('citation_conference_title', scholar.conference)
  add('citation_journal_title', scholar.journal)
  add('citation_technical_report_institution', scholar.institution)
  add('citation_arxiv_id', scholar.arxivId)
  add('citation_doi', scholar.doi)
  add('citation_abstract_html_url', canonicalUrl)
  add('citation_pdf_url', scholar.pdf && absolute(scholar.pdf))
  add('citation_language', 'en')
  if (scholar.keywords?.length) add('citation_keywords', scholar.keywords.join('; '))

  return tags
}
