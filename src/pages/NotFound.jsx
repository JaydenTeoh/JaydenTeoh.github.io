import { Link } from 'react-router-dom'
import Header from '../components/Header.jsx'
import SEO from '../components/SEO.jsx'

export default function NotFound() {
  return (
    <div className="app-shell">
      <SEO title="Page not found" description="This page could not be found." path="/404" noindex />
      <Header />
      <main className="not-found">
        <p className="not-found-text">this page doesn't exist.</p>
        <Link to="/" className="not-found-home">go back home</Link>
      </main>
    </div>
  )
}
