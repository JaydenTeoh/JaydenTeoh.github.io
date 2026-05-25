export default function PostMeta({ title, date, tags }) {
  return (
    <header className="writing-detail-header">
      <h1 className="writing-detail-title">{title}</h1>
      <p className="writing-detail-date">
        {date}
        {tags?.length ? <> &middot; {tags.join(' / ')}</> : null}
      </p>
    </header>
  )
}
