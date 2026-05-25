# JaydenTeoh.github.io

Source for [jaydenteoh.github.io](https://jaydenteoh.github.io). React + Vite SPA with MDX-powered blog posts. Visual style inspired by [stephenkyang/stephenkyang.github.io](https://github.com/stephenkyang/stephenkyang.github.io).

## Stack

- React 18 + Vite 5
- React Router (BrowserRouter, clean URLs)
- MDX for blog posts (`@mdx-js/rollup`) with `remark-gfm`, `remark-math`, `rehype-katex`
- KaTeX for math rendering

## Local development

Requires Node 18+ (Node 20 recommended).

```bash
npm install
npm run dev        # local dev server with HMR
npm run build      # production build into dist/
npm run preview    # serve the built site locally
```

## Project layout

```
.
├── .github/workflows/deploy.yml   # GitHub Pages deploy on push to main
├── public/                        # static assets (served at /)
│   ├── images/                    # paper thumbnails, profile pics, favicon
│   ├── data/                      # CV, SOP PDFs
│   ├── blog/assets/               # images embedded in blog posts
│   └── .nojekyll
├── src/
│   ├── main.jsx                   # entry
│   ├── App.jsx                    # routes
│   ├── index.css / App.css        # global styles
│   ├── components/                # Header, PaperCard, PostMeta
│   ├── pages/                     # Home, Blog, PostPage, NotFound
│   ├── data/                      # papers.js, posts.js, misc.js
│   └── posts/                     # MDX blog posts
├── index.html                     # Vite entry HTML
├── vite.config.js
└── package.json
```

## Adding a blog post

1. Create `src/posts/<slug>.mdx`. Reference images under `/blog/assets/<slug>/...` (drop the files into `public/blog/assets/<slug>/`).
2. Register the post in [src/data/posts.js](src/data/posts.js):

   ```js
   {
     slug: '<slug>',
     title: '...',
     date: 'YYYY-MM-DD',
     tags: ['...'],
     excerpt: '...',
     cover: '/blog/assets/<slug>/...',
     load: () => import('../posts/<slug>.mdx'),
   }
   ```

3. The post is rendered by `PostPage` at `/blog/<slug>`.

For interactive posts, import a React component at the top of the MDX file and use it inline:

```mdx
import MyDemo from '../components/MyDemo.jsx'

# Some Post

<MyDemo />

Continue with prose...
```

## Adding a paper

Add an entry to [src/data/papers.js](src/data/papers.js) with `thumb`, `title`, `authors`, `venue`, `year`, `links`, and `blurb`. Drop the thumbnail under `public/images/<paper-id>/...`. Author entries use `me: true` to bold your name and `equal: true` for shared first-author asterisks.

## Deploy

Pushing to `main` triggers `.github/workflows/deploy.yml`, which runs `npm ci && npm run build`, copies `dist/index.html` to `dist/404.html` for SPA deep-link fallback, and deploys via `actions/deploy-pages`.

One-time setup: in GitHub repo settings → Pages, set the source to **GitHub Actions**.
