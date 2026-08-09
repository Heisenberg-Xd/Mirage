import { Link } from 'react-router-dom';

/**
 * 404 — shown for any route that doesn't match "/" or "/chat".
 * Matches the Mirage Detector design system exactly.
 */
export default function NotFound() {
  return (
    <div className="min-h-screen bg-[#FAF8F4] flex flex-col items-center justify-center px-4 text-center">

      {/* Logo mark */}
      <p className="text-lg font-black text-gray-900 tracking-tight uppercase mb-10">
        <span style={{ color: 'var(--accent)' }}>◈</span> Mirage Detector
      </p>

      {/* Error badge */}
      <div className="section-badge mb-6 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>
          error
        </span>
        404 — Page Not Found
      </div>

      {/* Headline */}
      <h1 className="text-5xl sm:text-6xl font-black text-gray-900 leading-tight mb-5">
        Nothing Here.<br />
        <span style={{ color: 'var(--accent)' }}>No Hallucinations Either.</span>
      </h1>

      <p className="text-lg text-gray-500 leading-relaxed max-w-md mb-10">
        The page you're looking for doesn't exist. Head back to the landing page
        or jump straight into the workspace.
      </p>

      {/* Actions */}
      <div className="flex flex-wrap gap-4 justify-center">
        <Link to="/" className="btn-primary" id="not-found-home">
          <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>
            home
          </span>
          Back to Home
        </Link>
        <Link to="/chat" className="btn-ghost" id="not-found-chat">
          <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>
            verified_user
          </span>
          Open Workspace
        </Link>
      </div>

      {/* Decorative stat row */}
      <div className="mt-16 flex flex-wrap items-center justify-center gap-8 text-xs font-bold uppercase tracking-widest text-gray-300">
        {['No sign-up required', 'Fully open source', 'Zero LLM in verifier'].map((t) => (
          <span key={t} className="flex items-center gap-1.5">
            <span
              className="material-symbols-outlined text-sm"
              style={{ color: 'var(--accent)', fontVariationSettings: "'FILL' 1" }}
            >
              check_circle
            </span>
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}
