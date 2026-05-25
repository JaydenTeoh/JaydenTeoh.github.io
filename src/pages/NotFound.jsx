import { Link } from 'react-router-dom'
import Header from '../components/Header.jsx'

export default function NotFound() {
  return (
    <div className="app-shell">
      <Header />
      <main className="not-found">
        <p className="not-found-text">this page doesn't exist.</p>
        <Link to="/" className="not-found-home">go back home</Link>
      </main>
    </div>
  )
}
