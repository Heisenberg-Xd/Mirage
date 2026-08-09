export default function Footer() {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <footer className="border-t-2 border-[#111111] bg-[#FAF8F4] mt-auto">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12">
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-8 mb-10">
          {/* Brand */}
          <div className="sm:col-span-1">
            <p className="text-base font-black text-gray-900 uppercase tracking-tight mb-2">
              <span style={{ color: 'var(--accent)' }}>◈</span> Mirage
            </p>
            <p className="text-xs text-gray-500 leading-relaxed">
              Deterministic AI hallucination detection powered by DeBERTa v3 NLI, Groq, and Tavily.
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="text-xs font-black uppercase tracking-widest text-gray-400 mb-3">Product</h4>
            <ul className="space-y-2">
              <li>
                <button
                  onClick={() => scrollTo('how-it-works')}
                  className="text-sm font-semibold text-gray-600 hover:text-black transition-colors text-left"
                >
                  How It Works
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollTo('technology')}
                  className="text-sm font-semibold text-gray-600 hover:text-black transition-colors text-left"
                >
                  Technology
                </button>
              </li>
              <li>
                <button
                  onClick={() => scrollTo('hero')}
                  className="text-sm font-semibold text-gray-600 hover:text-black transition-colors text-left"
                >
                  Start Verifying
                </button>
              </li>
            </ul>
          </div>

          {/* Project */}
          <div>
            <h4 className="text-xs font-black uppercase tracking-widest text-gray-400 mb-3">Project</h4>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/Heisenberg-Xd/Mirage"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-semibold text-gray-600 hover:text-black transition-colors flex items-center gap-1"
                >
                  GitHub
                  <span className="material-symbols-outlined" style={{ fontSize: '13px' }}>open_in_new</span>
                </a>
              </li>
              <li>
                <button
                  onClick={() => scrollTo('about')}
                  className="text-sm font-semibold text-gray-600 hover:text-black transition-colors text-left"
                >
                  About
                </button>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="text-xs font-black uppercase tracking-widest text-gray-400 mb-3">Legal</h4>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-sm font-semibold text-gray-600 hover:text-black transition-colors">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-sm font-semibold text-gray-600 hover:text-black transition-colors">
                  Terms of Use
                </a>
              </li>
              <li>
                <a href="#" className="text-sm font-semibold text-gray-600 hover:text-black transition-colors">
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-6 border-t-2 border-[#E5E5E5] flex flex-col sm:flex-row items-center justify-between gap-3">
          <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">
            © {new Date().getFullYear()} Mirage Detector. Open source under MIT.
          </p>
          <p className="text-xs text-gray-400 font-semibold">
            Built with DeBERTa v3 · Groq · Tavily · FastAPI · React
          </p>
        </div>
      </div>
    </footer>
  );
}
