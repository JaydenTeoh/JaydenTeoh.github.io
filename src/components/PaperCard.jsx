export default function PaperCard({ paper }) {
  return (
    <article className="paper-card">
      <div className="paper-thumb-wrap">
        <img className="paper-thumb" src={paper.thumb} alt={paper.alt} loading="lazy" />
      </div>

      <div className="paper-body">
        <a href={paper.paperUrl} className="paper-title" target="_blank" rel="noreferrer">
          {paper.title}
        </a>

        <p className="paper-authors">
          {paper.authors.map((a, i) => (
            <span key={`${a.name}-${i}`}>
              {a.me ? <strong>{a.name}</strong> : a.name}
              {a.equal ? '*' : ''}
              {i < paper.authors.length - 1 ? ', ' : ''}
            </span>
          ))}
        </p>

        <p className="paper-venue">
          <em>{paper.venue}</em>, {paper.year}
          {paper.note ? <> &nbsp;({paper.note})</> : null}
          {paper.award ? (
            <>
              {' '}
              <span className="paper-award">({paper.award})</span>
            </>
          ) : null}
        </p>

        {paper.links?.length ? (
          <p className="paper-links">
            {paper.links.map((l, i) => (
              <span key={l.href + i}>
                <a href={l.href} target="_blank" rel="noreferrer">
                  {l.label}
                </a>
                {i < paper.links.length - 1 ? ' / ' : ''}
              </span>
            ))}
          </p>
        ) : null}

        {paper.blurb ? <p className="paper-blurb">{paper.blurb}</p> : null}
      </div>
    </article>
  )
}
