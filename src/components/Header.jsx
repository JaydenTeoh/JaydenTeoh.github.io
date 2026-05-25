import { Link, NavLink, useLocation } from 'react-router-dom'

export default function Header() {
  const { pathname } = useLocation()
  const isHome = pathname === '/'

  return (
    <header className={`top-bar${isHome ? ' top-bar-home' : ''}`}>
      <div className="name-block">
        {isHome ? null : (
          <Link to="/" className="brand-link">
            <p className="brand">Jayden Teoh</p>
          </Link>
        )}
      </div>
      <nav className="nav-links">
        <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`}>
          home
        </NavLink>
        <NavLink to="/blog" className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`}>
          blog
        </NavLink>
      </nav>
    </header>
  )
}
