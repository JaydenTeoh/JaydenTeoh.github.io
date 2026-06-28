import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { MDXProvider } from '@mdx-js/react'
import Header from '../components/Header.jsx'
import SEO from '../components/SEO.jsx'
import PaperHero from '../components/paper/PaperHero.jsx'
import {
  TLDR,
  Abstract,
  TOC,
  Section,
  Takeaway,
  Grid,
  Figure,
  VideoFigure,
  BenchmarkSlider,
  BibTeX,
  Footnote,
  FootnoteProvider,
  Cite,
} from '../components/paper/PaperBlocks.jsx'
import { getPost } from '../data/posts.js'


const mdxComponents = {
  TLDR,
  Abstract,
  TOC,
  Section,
  Takeaway,
  Grid,
  Figure,
  VideoFigure,
  BenchmarkSlider,
  BibTeX,
  Footnote,
  Cite,
  PaperHero,
}

export default function PaperPostPage() {
  const { year, slug } = useParams()
  const post = getPost(year, slug)
  const [Mdx, setMdx] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!post) return
    let cancelled = false
    setMdx(null)
    setError(null)
    post
      .load()
      .then((mod) => {
        if (!cancelled) setMdx(() => mod.default)
      })
      .catch((e) => {
        if (!cancelled) setError(e)
      })
    return () => {
      cancelled = true
    }
  }, [post])

  if (!post) {
    return (
      <div className="app-shell">
        <Header />
        <main className="paper-page">
          <div className="paper-container">
            <p>post not found.</p>
            <Link to="/blog">back to blog</Link>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <SEO
        title={post.title}
        description={post.excerpt || `Read ${post.title} by Jayden Teoh.`}
        path={`/blog/${year}/${slug}`}
        image={post.socialImage || post.cover}
      />
      <Header />

      <main className="paper-page paper-page--sidebar">
        <p className="paper-back">
          <Link to="/blog">← back to all posts</Link>
        </p>

        {error ? (
          <div className="paper-container">
            <p>failed to load post: {String(error.message || error)}</p>
          </div>
        ) : Mdx ? (
          <FootnoteProvider>
            <MDXProvider components={mdxComponents}>
              <Mdx components={mdxComponents} />
            </MDXProvider>
          </FootnoteProvider>
        ) : (
          <div className="paper-container">
            <p className="writing-loading">loading...</p>
          </div>
        )}
      </main>
    </div>
  )
}
