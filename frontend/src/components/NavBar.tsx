import { useNavigate, useLocation } from 'react-router-dom';

// ── Scroll target IDs that match section ids on LandingPage ──────────────────
type SectionId = 'hero' | 'how-it-works' | 'technology' | 'about';

export default function NavBar() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const isInChat  = location.pathname === '/chat';

  /**
   * Handle a nav-link click for a landing-page section.
   *
   * • On landing (/): smooth-scroll to the section — no route change.
   * • In chat (/chat): navigate to "/" and pass the target section as
   *   router state so LandingPage can scroll once it mounts.
   */
  const handleSectionClick = (id: SectionId) => {
    if (isInChat) {
      navigate('/', { state: { scrollTo: id } });
    } else {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  /**
   * Logo click:
   * • On landing: smooth-scroll to #hero (stay on the page).
   * • In chat: navigate to "/" (marketing site).
   */
  const handleLogoClick = () => {
    if (isInChat) {
      navigate('/');
    } else {
      document.getElementById('hero')?.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-[#FAF8F4] border-b-2 border-[#111111]">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">

        {/* ── Logo ─────────────────────────────────────────────────────── */}
        <button
          onClick={handleLogoClick}
          className="text-lg font-black text-gray-900 tracking-tight hover:text-black transition-colors uppercase flex items-center gap-2"
          aria-label="Go to top"
        >
          <span style={{ color: 'var(--accent)' }}>◈</span> Mirage Detector
        </button>

        {/* ── Nav links ────────────────────────────────────────────────── */}
        <nav className="hidden md:flex items-center gap-7">
          <button
            onClick={() => handleSectionClick('how-it-works')}
            className="nav-link"
            id="nav-how-it-works"
          >
            How it Works
          </button>
          <button
            onClick={() => handleSectionClick('technology')}
            className="nav-link"
            id="nav-technology"
          >
            Technology
          </button>
          <button
            onClick={() => handleSectionClick('about')}
            className="nav-link"
            id="nav-about"
          >
            About
          </button>
          <a
            href="https://github.com/Heisenberg-Xd/Mirage"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-link flex items-center gap-1"
            id="nav-github"
          >
            GitHub
            <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>open_in_new</span>
          </a>
        </nav>

        {/* ── CTA — only shown on the landing page, never in the workspace ── */}
        {!isInChat && (
          <button
            onClick={() => navigate('/chat')}
            className="btn-primary"
            style={{ padding: '8px 20px', fontSize: '13px' }}
            id="nav-get-started"
          >
            Get Started
          </button>
        )}
      </div>
    </header>
  );
}
