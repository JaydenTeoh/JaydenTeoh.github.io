export const siteName = 'Jayden Teoh'
export const siteUrl = 'https://jaydenteoh.github.io'
export const siteDescription =
  'Jayden Teoh is a student researcher writing about machine learning, research, and papers.'
export const defaultOgImage = '/images/profile_pic_2.jpg'

// Trailing-slash canonical form. GitHub Pages serves directory routes
// (e.g. /blog/2026/nextlat/index.html) and 301-redirects the no-slash URL to
// the trailing-slash one, so canonical URLs and the sitemap must use it too.
export function canonicalPath(path) {
  if (!path || path === '/') return '/'
  const trimmed = path.replace(/\/+$/, '')
  return `${trimmed}/`
}

// Page <title>. Avoids "Jayden Teoh | Jayden Teoh" when the page title already
// equals the site name (the home page).
export function buildTitle(title) {
  return title && title !== siteName ? `${title} | ${siteName}` : siteName
}
