import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import Header from '../components/Header.jsx'
import PostMeta from '../components/PostMeta.jsx'
import { getPost } from '../data/posts.js'

export default function PostPage() {
  const { slug } = useParams()
  const post = getPost(slug)
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
        <main className="writing-detail-wrapper">
          <p>post not found.</p>
          <Link to="/blog">back to blog</Link>
        </main>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <Header />

      <main className="writing-detail-wrapper">
        <p className="writing-back">
          <Link to="/blog">back to all posts</Link>
        </p>

        <PostMeta title={post.title} date={post.date} tags={post.tags} />

        <article className="writing-detail-content">
          {error ? (
            <p>failed to load post: {String(error.message || error)}</p>
          ) : Mdx ? (
            <Mdx />
          ) : (
            <p className="writing-loading">loading...</p>
          )}
        </article>
      </main>
    </div>
  )
}
