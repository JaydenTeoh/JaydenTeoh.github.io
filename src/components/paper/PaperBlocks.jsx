import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'

const useIsoLayoutEffect =
  typeof window !== 'undefined' ? useLayoutEffect : useEffect

export function TLDR({ children }) {
  return (
    <section className="paper-tldr">
      <p>
        <strong>TL;DR:</strong> {children}
      </p>
    </section>
  )
}

export function Abstract({ children }) {
  return (
    <section className="paper-abstract">
      <div className="paper-container">
        <h2>Abstract</h2>
        <p>{children}</p>
      </div>
    </section>
  )
}

export function TOC({ items }) {
  const [active, setActive] = useState(items?.[0]?.slug ?? '')

  useEffect(() => {
    if (!items?.length) return

    const updateActive = () => {
      const offset = 120
      let current = items[0].slug
      for (const it of items) {
        const el = document.getElementById(it.slug)
        if (el && el.getBoundingClientRect().top <= offset) {
          current = it.slug
        }
      }
      setActive((prev) => (prev === current ? prev : current))
    }

    updateActive()
    window.addEventListener('scroll', updateActive, { passive: true })
    window.addEventListener('resize', updateActive)
    return () => {
      window.removeEventListener('scroll', updateActive)
      window.removeEventListener('resize', updateActive)
    }
  }, [items])

  if (!items?.length) return null

  const handleClick = (e, slug) => {
    e.preventDefault()
    document.getElementById(slug)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActive(slug)
    history.replaceState(null, '', `#${slug}`)
  }

  const links = items.map((it) => (
    <a
      key={it.slug}
      href={`#${it.slug}`}
      className={active === it.slug ? 'is-active' : undefined}
      aria-current={active === it.slug ? 'location' : undefined}
      onClick={(e) => handleClick(e, it.slug)}
    >
      {it.text}
    </a>
  ))

  return (
    <>
      <nav className="paper-toc paper-toc--sidebar" aria-label="Table of contents">
        <span className="paper-toc-sidebar-label">On this page</span>
        <div className="paper-toc-sidebar-links">{links}</div>
      </nav>
      <nav className="paper-toc paper-toc--mobile" aria-label="Table of contents">
        <span className="paper-toc-label">Contents:</span>
        {links}
      </nav>
    </>
  )
}

export function Section({ id, children }) {
  return (
    <section id={id} className="paper-section paper-prose">
      {children}
    </section>
  )
}

export function Takeaway({ label = 'Key Insight', children }) {
  return (
    <div className="paper-takeaway" data-label={label}>
      {children}
    </div>
  )
}

export function Grid({ cols = 2, children }) {
  return <div className={`paper-grid paper-grid-${cols}`}>{children}</div>
}

export function Figure({ src, alt = '', caption, width, ...rest }) {
  return (
    <figure className="paper-figure">
      <img src={src} alt={alt || caption || ''} style={width ? { width } : undefined} {...rest} />
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  )
}

export function VideoFigure({ src, caption, ...rest }) {
  return (
    <figure>
      <video autoPlay muted loop playsInline preload="metadata" {...rest}>
        <source src={src} type="video/mp4" />
      </video>
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  )
}

function isGifSrc(src) {
  return /\.gif($|\?)/i.test(src)
}

export function BenchmarkSlider({ slides }) {
  const [active, setActive] = useState(0)
  const count = slides?.length ?? 0

  if (!count) return null

  const slide = slides[active]
  const go = (delta) => setActive((i) => (i + delta + count) % count)

  const media =
    slide.src &&
    (slide.mediaType === 'video' && !isGifSrc(slide.src) ? (
      <video
        key={slide.src}
        autoPlay
        muted
        loop
        playsInline
        preload="metadata"
        className="paper-benchmark-slider__media"
      >
        <source src={slide.src} type="video/mp4" />
      </video>
    ) : (
      <img
        key={slide.src}
        src={slide.src}
        alt={slide.alt || slide.title || slide.label}
        className="paper-benchmark-slider__media"
      />
    ))

  return (
    <div className="paper-benchmark-slider">
      <div className="paper-benchmark-slider__tabs" role="tablist" aria-label="Benchmark results">
        {slides.map((s, i) => (
          <button
            key={s.id}
            type="button"
            role="tab"
            id={`benchmark-tab-${s.id}`}
            aria-selected={i === active}
            aria-controls={`benchmark-panel-${s.id}`}
            className={`paper-benchmark-slider__tab${i === active ? ' is-active' : ''}`}
            onClick={() => setActive(i)}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div
        className="paper-benchmark-slider__panel"
        role="tabpanel"
        id={`benchmark-panel-${slide.id}`}
        aria-labelledby={`benchmark-tab-${slide.id}`}
      >
        <div className="paper-benchmark-slider__stage">
          <button
            type="button"
            className="paper-benchmark-slider__nav paper-benchmark-slider__nav--prev"
            onClick={() => go(-1)}
            aria-label="Previous benchmark"
          >
            <i className="bi bi-chevron-left" aria-hidden="true" />
          </button>

          <figure className="paper-benchmark-slider__figure">
            {media || (
              <div className="paper-benchmark-slider__placeholder" aria-hidden="true">
                <span className="paper-benchmark-slider__placeholder-label">{slide.label}</span>
                <span className="paper-benchmark-slider__placeholder-hint">Figure coming soon</span>
              </div>
            )}
          </figure>

          <button
            type="button"
            className="paper-benchmark-slider__nav paper-benchmark-slider__nav--next"
            onClick={() => go(1)}
            aria-label="Next benchmark"
          >
            <i className="bi bi-chevron-right" aria-hidden="true" />
          </button>
        </div>

        {(slide.description || slide.caption) && (
          <p className="paper-benchmark-slider__description">
            {slide.description || slide.caption}
          </p>
        )}
      </div>

      <div className="paper-benchmark-slider__dots" aria-label="Benchmark pagination">
        {slides.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className={`paper-benchmark-slider__dot${i === active ? ' is-active' : ''}`}
            aria-label={`Show ${s.label}`}
            aria-current={i === active ? 'true' : undefined}
            onClick={() => setActive(i)}
          />
        ))}
      </div>
    </div>
  )
}

export function FootnoteProvider({ children }) {
  // Kept as a no-op wrapper for API stability. Numbering is derived from DOM
  // order, so no shared state is required.
  return children
}

export function Footnote({ children }) {
  const markerRef = useRef(null)
  const [num, setNum] = useState(0)
  const [open, setOpen] = useState(false)
  const reactId = useId()
  const tipId = `paper-fn-${reactId.replace(/:/g, '_')}`

  useIsoLayoutEffect(() => {
    const el = markerRef.current
    if (!el) return
    const root = el.closest('.paper-page') || document
    const all = root.querySelectorAll('.paper-footnote-marker')
    const idx = Array.prototype.indexOf.call(all, el)
    const next = idx + 1
    if (idx >= 0) setNum((prev) => (prev === next ? prev : next))
  }, [])

  return (
    <span className={`paper-footnote${open ? ' is-open' : ''}`}>
      <button
        ref={markerRef}
        type="button"
        className="paper-footnote-marker"
        aria-describedby={tipId}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        onBlur={(e) => {
          const next = e.relatedTarget
          if (!next || !e.currentTarget.parentNode.contains(next)) {
            setOpen(false)
          }
        }}
      >
        {num || ''}
      </button>
      <span id={tipId} role="tooltip" className="paper-footnote-tooltip">
        <span className="paper-footnote-num" aria-hidden="true">
          {num ? `${num}.` : ''}
        </span>
        <span className="paper-footnote-body">{children}</span>
      </span>
    </span>
  )
}

function citeKey(ref, fallback) {
  return ref.id || ref.href || ref.title || ref.authors || fallback
}

function buildCiteNumberMap(root) {
  const seen = new Map()
  let counter = 1
  for (const node of root.querySelectorAll('[data-cite-key]')) {
    const k = node.getAttribute('data-cite-key')
    if (!k || seen.has(k)) continue
    seen.set(k, counter)
    counter += 1
  }
  return seen
}

function numsEqual(a, b) {
  return a.length === b.length && a.every((n, i) => n === b[i])
}

function CiteTooltipBody({ authors, year, title, venue, href }) {
  const body = (
    <>
      {authors || year != null ? (
        <span className="paper-cite-meta">
          {authors}
          {year != null ? <> ({year})</> : null}
        </span>
      ) : null}
      {title ? <span className="paper-cite-title">{title}</span> : null}
      {venue ? <span className="paper-cite-venue">{venue}</span> : null}
    </>
  )

  if (!href) return body
  return (
    <a href={href} target="_blank" rel="noopener noreferrer" className="paper-cite-tooltip-link">
      {body}
    </a>
  )
}

function CiteGroup({ refs }) {
  const reactId = useId()
  const tipId = `paper-cite-group-${reactId.replace(/:/g, '_')}`
  const groupRef = useRef(null)
  const keys = useMemo(
    () => refs.map((ref, i) => citeKey(ref, `${reactId}-${i}`)),
    [refs, reactId]
  )
  const keysKey = keys.join('\0')
  const [nums, setNums] = useState(() => refs.map(() => 0))

  useIsoLayoutEffect(() => {
    const root = groupRef.current?.closest('.paper-page') || document
    const seen = buildCiteNumberMap(root)
    const next = keys.map((k) => seen.get(k) || 0)
    setNums((prev) => (numsEqual(prev, next) ? prev : next))
  }, [keysKey])

  const label = nums.filter((n) => n > 0).join(', ')

  return (
    <span className="paper-cite paper-cite-group" ref={groupRef}>
      {keys.map((k) => (
        <span key={k} data-cite-key={k} className="paper-cite-register" aria-hidden="true" />
      ))}
      <button
        type="button"
        className="paper-cite-link"
        aria-describedby={tipId}
        aria-label={`References ${label}`}
      >
        [{label || ''}]
      </button>
      <span id={tipId} role="tooltip" className="paper-cite-tooltip paper-cite-tooltip--group">
        {refs.map((ref, i) => (
          <span key={keys[i]} className="paper-cite-tooltip-entry">
            <span className="paper-cite-num" aria-hidden="true">
              {nums[i] ? `${nums[i]}.` : ''}
            </span>
            <CiteTooltipBody {...ref} />
          </span>
        ))}
      </span>
    </span>
  )
}

export function Cite({ id, href, authors, year, title, venue, refs }) {
  if (refs?.length) {
    return <CiteGroup refs={refs} />
  }

  const reactId = useId()
  const tipId = `paper-cite-${reactId.replace(/:/g, '_')}`
  const hasTip = Boolean(title || venue || authors)
  const key = citeKey({ id, href, authors, title }, reactId)

  const linkRef = useRef(null)
  const [num, setNum] = useState(0)

  useIsoLayoutEffect(() => {
    const el = linkRef.current
    if (!el) return
    const root = el.closest('.paper-page') || document
    const assigned = buildCiteNumberMap(root).get(key)
    if (assigned) setNum((prev) => (prev === assigned ? prev : assigned))
  }, [key])

  return (
    <span className="paper-cite">
      <a
        ref={linkRef}
        href={href}
        data-cite-key={key}
        className="paper-cite-link"
        target="_blank"
        rel="noopener noreferrer"
        aria-describedby={hasTip ? tipId : undefined}
      >
        [{num || ''}]
      </a>
      {hasTip ? (
        <span id={tipId} role="tooltip" className="paper-cite-tooltip">
          <CiteTooltipBody authors={authors} year={year} title={title} venue={venue} />
        </span>
      ) : null}
    </span>
  )
}

export function BibTeX({ note, children }) {
  const code = (typeof children === 'string' ? children : '').trim()
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch (_) {
      // clipboard not available; ignore
    }
  }

  return (
    <section className="paper-bibtex-section">
      <h2>BibTeX</h2>
      {note ? <p className="paper-bibtex-note">{note}</p> : null}
      <div className="paper-bibtex-wrap">
        <button
          type="button"
          className={`paper-bibtex-copy${copied ? ' is-copied' : ''}`}
          onClick={handleCopy}
          aria-label={copied ? 'Copied' : 'Copy BibTeX'}
        >
          <i className={copied ? 'bi bi-check-lg' : 'bi bi-clipboard'} aria-hidden="true" />
        </button>
        <pre className="paper-bibtex">
          <code>{highlightBibtex(code)}</code>
        </pre>
      </div>
    </section>
  )
}

function highlightBibtex(src) {
  const parts = []
  let lastIdx = 0
  let key = 0
  const re = /@\w+/g
  let m
  while ((m = re.exec(src)) !== null) {
    if (m.index > lastIdx) parts.push(src.slice(lastIdx, m.index))
    parts.push(
      <span key={key++} className="bib-entry">
        {m[0]}
      </span>
    )
    lastIdx = m.index + m[0].length
  }
  if (lastIdx < src.length) parts.push(src.slice(lastIdx))
  return parts
}
