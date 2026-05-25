import { useEffect, useRef, useState } from 'react'

export default function TypingText({
  text,
  speed = 90,
  startDelay = 200,
  className = '',
  cursor = true,
}) {
  const prefersReducedMotion =
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

  const [shown, setShown] = useState(prefersReducedMotion ? text : '')
  const [done, setDone] = useState(prefersReducedMotion)
  const indexRef = useRef(0)

  useEffect(() => {
    if (prefersReducedMotion) return
    setShown('')
    setDone(false)
    indexRef.current = 0

    const start = setTimeout(function tick() {
      indexRef.current += 1
      setShown(text.slice(0, indexRef.current))
      if (indexRef.current >= text.length) {
        setDone(true)
        return
      }
      timer = setTimeout(tick, speed)
    }, startDelay)

    let timer = start
    return () => clearTimeout(timer)
  }, [text, speed, startDelay, prefersReducedMotion])

  return (
    <span className={className} aria-label={text}>
      <span aria-hidden="true">{shown}</span>
      {cursor ? (
        <span className={`typing-cursor${done ? ' typing-cursor-blink' : ''}`} aria-hidden="true">
          |
        </span>
      ) : null}
    </span>
  )
}
