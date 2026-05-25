import { Link } from 'react-router-dom'

const pastelColors = [
  '#FFB3BA',
  '#BAFFC9',
  '#BAE1FF',
  '#FFFFBA',
  '#FFB3F7',
  '#B3FFE6',
  '#FFD4B3',
  '#E6B3FF',
  '#B3FFB3',
  '#FFE6B3',
  '#B3D4FF',
  '#FFB3D4',
  '#D4FFB3',
  '#B3FFF0',
  '#F0B3FF',
  '#FFE6E6',
]

function tagColor(tag) {
  let hash = 0
  for (let i = 0; i < tag.length; i++) {
    hash = tag.charCodeAt(i) + ((hash << 5) - hash)
  }
  return pastelColors[Math.abs(hash) % pastelColors.length]
}

export default function PostCard({ post }) {
  return (
    <article className="post-card">
      {post.cover ? (
        <Link to={`/blog/${post.slug}`} className="post-cover" aria-hidden="true">
          <img src={post.cover} alt="" loading="lazy" />
        </Link>
      ) : null}

      <div className="post-info">
        <h3 className="post-title">
          <Link to={`/blog/${post.slug}`}>{post.title}</Link>
        </h3>
        <p className="post-date">{post.date}</p>
        {post.excerpt ? <p className="post-excerpt">{post.excerpt}</p> : null}
        {post.tags?.length ? (
          <p className="post-tags">
            {post.tags.map((t) => (
              <span key={t} className="post-tag" style={{ background: tagColor(t) }}>
                {t}
              </span>
            ))}
          </p>
        ) : null}
      </div>
    </article>
  )
}
