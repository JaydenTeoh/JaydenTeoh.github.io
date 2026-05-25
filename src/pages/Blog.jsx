import { useMemo, useState } from 'react'
import Header from '../components/Header.jsx'
import PostCard from '../components/PostCard.jsx'
import { posts, getAllTags } from '../data/posts.js'

export default function Blog() {
  const [filter, setFilter] = useState('all')
  const tags = useMemo(() => getAllTags(), [])

  const filtered = filter === 'all' ? posts : posts.filter((p) => p.tags.includes(filter))

  return (
    <div className="app-shell">
      <Header />

      <main className="writings-page">
        <div className="writings-header">
          <h1 className="page-title">blog</h1>
          <p className="page-subtitle">
            thoughts on machine learning, research, and other topics.
          </p>
        </div>

        <div className="writing-tags">
          {tags.map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setFilter(t)}
              className={`writing-tag${filter === t ? ' writing-tag-active' : ''}`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="post-list">
          {filtered.length === 0 ? (
            <p className="writing-empty">no posts under this tag yet.</p>
          ) : (
            filtered.map((p) => <PostCard key={p.slug} post={p} />)
          )}
        </div>
      </main>
    </div>
  )
}
