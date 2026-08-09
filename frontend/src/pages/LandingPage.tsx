import { useState, useRef, useEffect, memo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';

// ─── Constants ────────────────────────────────────────────────────────────────

const EXAMPLE_QUERIES = [
  'Who invented the telephone?',
  'Is quantum computing commercially available?',
  'When did humans first land on the Moon?',
  'What is the speed of light?',
];

const HOW_IT_WORKS = [
  {
    step: '01',
    icon: 'help_circle',
    title: 'Ask a Question',
    desc: 'Enter any factual question. Mirage accepts anything from science to history to current events.',
  },
  {
    step: '02',
    icon: 'smart_toy',
    title: 'AI Generates Answer',
    desc: "Groq's Llama 3.3 model produces a fast, comprehensive answer — the kind any chatbot would give.",
  },
  {
    step: '03',
    icon: 'travel_explore',
    title: 'Live Evidence Search',
    desc: 'Tavily retrieves real-time web sources. Every claim is grounded against fresh, authoritative evidence.',
  },
  {
    step: '04',
    icon: 'verified_user',
    title: 'Deterministic Verdict',
    desc: 'DeBERTa v3 NLI scores each claim. A trust score and hallucination label are computed — no LLM guessing.',
  },
];

const TECH_STACK = [
  { name: 'Groq',            icon: 'bolt',            desc: 'Ultra-fast LLM inference at 500+ tokens/s' },
  { name: 'Llama 3.3 70B',  icon: 'psychology',      desc: "Meta's open-source frontier language model" },
  { name: 'DeBERTa v3',     icon: 'analytics',       desc: 'Microsoft NLI model for entailment detection' },
  { name: 'Cross-Encoder',  icon: 'swap_vert',       desc: 'Semantic relevance scoring for claims vs evidence' },
  { name: 'Tavily',         icon: 'public',          desc: 'Real-time web search API with structured results' },
  { name: 'spaCy',          icon: 'account_tree',    desc: 'NLP pipeline for entity extraction and parsing' },
  { name: 'FastAPI',        icon: 'api',             desc: 'High-performance Python async backend' },
  { name: 'React + Vite',   icon: 'code',            desc: 'Lightning-fast frontend with TypeScript' },
  { name: 'RapidFuzz',      icon: 'manage_search',   desc: 'Fuzzy string matching for entity alignment' },
];

const FEATURES = [
  { icon: 'travel_explore', title: 'Live Web Evidence',        desc: 'Tavily retrieves fresh sources for every query in real time. No static knowledge base.' },
  { icon: 'verified_user',  title: 'Deterministic NLI',        desc: 'DeBERTa v3 classifies entailment or contradiction — no LLM in the verification loop.' },
  { icon: 'split_scene',    title: 'Atomic Claim Extraction',  desc: 'spaCy parses every sentence into individual verifiable facts for precise scoring.' },
  { icon: 'speed',          title: 'Hallucination Meter',      desc: 'A confidence-scored label: Not Hallucinating, Cannot Verify, or Hallucinating.' },
  { icon: 'gavel',          title: 'Source Authority',         desc: 'Each source is scored for credibility. Wikipedia, .gov, and .edu rank highest.' },
  { icon: 'leaderboard',    title: 'Cross-Encoder Ranking',    desc: 'ms-marco CrossEncoder re-ranks evidence by semantic relevance to each claim.' },
];

const PIPELINE = [
  { icon: 'help_circle',    label: 'User Question' },
  { icon: 'smart_toy',      label: 'LLM Answer' },
  { icon: 'split_scene',    label: 'Claim Extraction' },
  { icon: 'travel_explore', label: 'Evidence Search' },
  { icon: 'analytics',      label: 'NLI Scoring' },
  { icon: 'speed',          label: 'Hallucination Meter' },
  { icon: 'verified_user',  label: 'Trust Report' },
];

// ─── Memoised sub-components ─────────────────────────────────────────────────

const HowItWorksSection = memo(() => (
  <section id="how-it-works" className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t-2 border-[#111111]">
    <div className="mb-14 text-center">
      <span className="section-badge mb-4 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>route</span>
        How It Works
      </span>
      <h2 className="text-3xl sm:text-4xl font-black text-gray-900 mt-3">
        From Question to<br />
        <span style={{ color: 'var(--accent)' }}>Verified Answer.</span>
      </h2>
    </div>

    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {HOW_IT_WORKS.map((item) => (
        <div key={item.step} className="card card-hover p-6 flex flex-col gap-4">
          <div className="flex items-start justify-between">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center border-2 border-[#111111]"
              style={{ background: 'var(--accent-light)' }}
            >
              <span
                className="material-symbols-outlined"
                style={{ color: 'var(--accent)', fontVariationSettings: "'FILL' 1", fontSize: '22px' }}
              >
                {item.icon}
              </span>
            </div>
            <span className="text-3xl font-black text-gray-100 select-none">{item.step}</span>
          </div>
          <div>
            <h3 className="text-base font-black text-gray-900 mb-2">{item.title}</h3>
            <p className="text-sm text-gray-500 leading-relaxed">{item.desc}</p>
          </div>
        </div>
      ))}
    </div>
  </section>
));

const TechnologySection = memo(() => (
  <section id="technology" className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t-2 border-[#111111]">
    <div className="mb-14 text-center">
      <span className="section-badge mb-4 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>settings</span>
        Technology
      </span>
      <h2 className="text-3xl sm:text-4xl font-black text-gray-900 mt-3">
        Built on<br />
        <span style={{ color: 'var(--accent)' }}>Production-Grade AI.</span>
      </h2>
    </div>

    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-3 gap-4">
      {TECH_STACK.map((tech) => (
        <div key={tech.name} className="card card-hover p-5 flex items-start gap-4">
          <div
            className="w-10 h-10 shrink-0 rounded-lg flex items-center justify-center border-2 border-[#111111]"
            style={{ background: 'var(--accent-light)' }}
          >
            <span
              className="material-symbols-outlined"
              style={{ color: 'var(--accent)', fontVariationSettings: "'FILL' 1", fontSize: '20px' }}
            >
              {tech.icon}
            </span>
          </div>
          <div className="min-w-0">
            <p className="text-sm font-black text-gray-900 truncate">{tech.name}</p>
            <p className="text-xs text-gray-500 leading-relaxed mt-0.5">{tech.desc}</p>
          </div>
        </div>
      ))}
    </div>
  </section>
));

const WhyMirageSection = memo(() => (
  <section className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t-2 border-[#111111]">
    <div className="mb-14 text-center">
      <span className="section-badge mb-4 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>compare</span>
        Why Mirage
      </span>
      <h2 className="text-3xl sm:text-4xl font-black text-gray-900 mt-3">
        AI Answers Aren't<br />
        <span style={{ color: 'var(--accent)' }}>Always Right.</span>
      </h2>
    </div>

    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto">
      {/* Without Mirage */}
      <div className="card p-6">
        <h3 className="text-sm font-black uppercase tracking-widest text-gray-400 mb-5">Without Verification</h3>
        <ul className="space-y-3">
          {[
            'AI answer has no source backing',
            'Hallucinations go undetected',
            'No confidence score provided',
            'You have to manually fact-check',
            'Blind trust in the model',
          ].map((item) => (
            <li key={item} className="flex items-start gap-3 text-sm text-gray-600">
              <span className="shrink-0 mt-0.5 w-5 h-5 rounded-full bg-red-100 border-2 border-red-300 flex items-center justify-center">
                <span className="material-symbols-outlined text-red-500" style={{ fontSize: '13px' }}>close</span>
              </span>
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* With Mirage */}
      <div className="card p-6 border-[var(--accent)]" style={{ borderColor: 'var(--accent)' }}>
        <h3 className="text-sm font-black uppercase tracking-widest mb-5" style={{ color: 'var(--accent)' }}>
          With Mirage
        </h3>
        <ul className="space-y-3">
          {[
            'Every claim checked against live evidence',
            'Contradictions flagged immediately',
            'Precise confidence score (0–100%)',
            'Source authority ranking included',
            'Entity drift detection built-in',
          ].map((item) => (
            <li key={item} className="flex items-start gap-3 text-sm text-gray-800 font-medium">
              <span
                className="shrink-0 mt-0.5 w-5 h-5 rounded-full border-2 flex items-center justify-center"
                style={{ background: 'var(--accent-light)', borderColor: 'var(--accent)' }}
              >
                <span className="material-symbols-outlined" style={{ fontSize: '13px', color: 'var(--accent)' }}>check</span>
              </span>
              {item}
            </li>
          ))}
        </ul>
      </div>
    </div>
  </section>
));

const FeaturesSection = memo(() => (
  <section className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t-2 border-[#111111]">
    <div className="mb-14 text-center">
      <span className="section-badge mb-4 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>auto_awesome</span>
        Features
      </span>
      <h2 className="text-3xl sm:text-4xl font-black text-gray-900 mt-3">
        Every Layer of<br />
        <span style={{ color: 'var(--accent)' }}>Fact Verification.</span>
      </h2>
    </div>

    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
      {FEATURES.map((f) => (
        <div key={f.title} className="card card-hover p-6 group">
          <div
            className="w-11 h-11 rounded-lg flex items-center justify-center border-2 border-[#111111] mb-4 transition-all duration-200 group-hover:scale-110"
            style={{ background: 'var(--accent-light)' }}
          >
            <span
              className="material-symbols-outlined"
              style={{ color: 'var(--accent)', fontVariationSettings: "'FILL' 1", fontSize: '22px' }}
            >
              {f.icon}
            </span>
          </div>
          <h3 className="text-base font-black text-gray-900 mb-2">{f.title}</h3>
          <p className="text-sm text-gray-500 leading-relaxed">{f.desc}</p>
        </div>
      ))}
    </div>
  </section>
));

const PipelineSection = memo(() => (
  <section className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t-2 border-[#111111]">
    <div className="mb-14 text-center">
      <span className="section-badge mb-4 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>schema</span>
        Verification Pipeline
      </span>
      <h2 className="text-3xl sm:text-4xl font-black text-gray-900 mt-3">
        A Deterministic<br />
        <span style={{ color: 'var(--accent)' }}>Seven-Stage Process.</span>
      </h2>
    </div>

    <div className="flex flex-col items-center max-w-xs mx-auto">
      {PIPELINE.map((stage, idx) => (
        <div key={stage.label} className="flex flex-col items-center w-full">
          <div
            className="card w-full px-5 py-3.5 flex items-center gap-4"
            style={idx === 0 || idx === PIPELINE.length - 1 ? { borderColor: 'var(--accent)' } : {}}
          >
            <div
              className="w-9 h-9 shrink-0 rounded-lg flex items-center justify-center border-2 border-[#111111]"
              style={{ background: idx === 0 || idx === PIPELINE.length - 1 ? 'var(--accent)' : 'var(--accent-light)' }}
            >
              <span
                className="material-symbols-outlined"
                style={{
                  color: idx === 0 || idx === PIPELINE.length - 1 ? '#fff' : 'var(--accent)',
                  fontVariationSettings: "'FILL' 1",
                  fontSize: '18px',
                }}
              >
                {stage.icon}
              </span>
            </div>
            <span className="text-sm font-bold text-gray-900">{stage.label}</span>
            <span
              className="ml-auto text-xs font-black px-2 py-0.5 rounded border border-[#111] bg-[#F5F5F5]"
            >
              {String(idx + 1).padStart(2, '0')}
            </span>
          </div>
          {idx < PIPELINE.length - 1 && <div className="pipeline-arrow my-1" />}
        </div>
      ))}
    </div>
  </section>
));

const AboutSection = memo(() => (
  <section id="about" className="w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t-2 border-[#111111]">
    <div className="max-w-3xl mx-auto">
      <span className="section-badge mb-6 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>info</span>
        About Mirage
      </span>
      <span className="accent-line" />
      <h2 className="text-3xl sm:text-4xl font-black text-gray-900 mb-6">
        Built to Solve<br />
        <span style={{ color: 'var(--accent)' }}>AI Hallucination.</span>
      </h2>
      <div className="space-y-4 text-gray-600 leading-relaxed">
        <p>
          Mirage Detector was built out of a simple observation: AI language models are useful,
          but they invent facts. Not sometimes — often. The hallucination problem isn't a bug that
          will be patched; it's an architectural property of how transformers work. They predict
          plausible continuations, not ground truth.
        </p>
        <p>
          Most "AI fact checkers" solve this by asking another LLM to judge the output.
          That's circular. Mirage takes a different approach: <strong>deterministic NLI</strong>.
          Every claim extracted from an AI answer is cross-referenced against live web evidence
          using DeBERTa v3 — a model trained specifically to detect entailment and contradiction,
          not to generate plausible text.
        </p>
        <p>
          The result is a pipeline with no guesswork: a structured trust score,
          a per-claim verdict, and source authority rankings — computed identically every time,
          with full transparency into every step.
        </p>
      </div>
      <div className="mt-8 grid grid-cols-3 gap-4">
        {[
          { value: '100%', label: 'Deterministic' },
          { value: '0',    label: 'LLMs in the verifier' },
          { value: '7',    label: 'Pipeline stages' },
        ].map((stat) => (
          <div key={stat.label} className="card p-4 text-center">
            <p className="text-2xl font-black text-gray-900" style={{ color: 'var(--accent)' }}>{stat.value}</p>
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mt-1">{stat.label}</p>
          </div>
        ))}
      </div>
    </div>
  </section>
));

const CTASection = memo(({ onStart }: { onStart: () => void }) => (
  <section className="w-full border-t-2 border-[#111111]">
    <div className="max-w-3xl mx-auto px-4 sm:px-6 py-28 text-center">
      <span className="section-badge mb-6 inline-flex">
        <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)' }}>rocket_launch</span>
        Get Started
      </span>
      <h2 className="text-4xl sm:text-5xl font-black text-gray-900 leading-tight mb-5">
        Ready to Verify<br />
        <span style={{ color: 'var(--accent)' }}>AI Answers?</span>
      </h2>
      <p className="text-lg text-gray-500 leading-relaxed mb-10 max-w-lg mx-auto">
        Paste any question. Mirage does the rest — live evidence, deterministic NLI, full trust report.
      </p>
      <button className="btn-primary text-base px-8 py-4" onClick={onStart} id="cta-get-started">
        <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>verified_user</span>
        Start Verifying — It's Free
      </button>
    </div>
  </section>
));


// ─── Main Page ────────────────────────────────────────────────────────────────

export default function LandingPage() {
  const [heroQuery, setHeroQuery] = useState('');
  const heroRef  = useRef<HTMLTextAreaElement>(null);
  const navigate = useNavigate();
  const location = useLocation();

  // ── Scroll-to handler after navigating back from /chat ─────────────────────
  useEffect(() => {
    const state = location.state as { scrollTo?: string } | null;
    if (state?.scrollTo) {
      // Give the DOM a tick to finish mounting before scrolling.
      const timer = setTimeout(() => {
        document.getElementById(state.scrollTo!)?.scrollIntoView({ behavior: 'smooth' });
        // Clear the state so a refresh / back-button doesn't re-scroll.
        window.history.replaceState({}, '');
      }, 80);
      return () => clearTimeout(timer);
    }
  }, [location.state]);

  // ── Routing helpers ────────────────────────────────────────────────────────

  /**
   * Navigate to the Chat Workspace.
   * If the user has typed something in the hero box, pass it as the first
   * message so Workspace can fire it automatically on mount.
   */
  const goToChat = (query?: string) => {
    const q = query ?? heroQuery.trim();
    navigate('/chat', q ? { state: { initialQuestion: q } } : undefined);
  };

  const handleHeroKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      goToChat();
    }
  };

  return (
    <>
      <NavBar />
      <main className="flex-grow flex flex-col items-center">

        {/* ── SECTION 1: Hero ──────────────────────────────────────────── */}
        <section id="hero" className="w-full max-w-6xl mx-auto px-4 sm:px-6 pt-36 pb-20 flex flex-col items-center text-center">
          {/* Badge */}
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-black uppercase tracking-widest mb-8 border-2 border-[#111111] shadow-[2px_2px_0px_0px_#111111] bg-[#F0F0F0]"
          >
            <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)', fontVariationSettings: "'FILL' 1" }}>
              auto_awesome
            </span>
            Powered by DeBERTa v3 NLI · Groq Llama 3.3 · Tavily
          </div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl font-black text-gray-900 leading-[1.05] tracking-tight mb-5">
            Verify AI Answers<br />
            <span style={{ color: 'var(--accent)' }}>Before You Trust Them.</span>
          </h1>

          <p className="text-lg text-gray-500 leading-relaxed max-w-xl mb-12">
            Mirage generates an AI answer, then runs deterministic fact-checking against live web evidence —
            no LLM in the verification loop.
          </p>

          {/* ── Hero search preview — routes to /chat on submit ───────── */}
          <div className="card w-full max-w-2xl p-3" id="hero-search">
            <textarea
              ref={heroRef}
              className="w-full bg-transparent resize-none min-h-[56px] max-h-52 px-3 py-2 text-base text-gray-900 placeholder-gray-400 focus:outline-none leading-relaxed"
              placeholder="Ask anything… e.g. 'Who invented the telephone?'"
              rows={2}
              value={heroQuery}
              onChange={(e) => setHeroQuery(e.target.value)}
              onKeyDown={handleHeroKeyDown}
              onInput={(e) => {
                const t = e.target as HTMLTextAreaElement;
                t.style.height = 'auto';
                t.style.height = Math.min(t.scrollHeight, 208) + 'px';
              }}
              aria-label="Enter a question to verify"
            />
            <div className="flex items-center justify-between pt-2 border-t-2 border-[#111111] mt-1">
              <p className="text-xs text-gray-400 font-bold uppercase tracking-wide hidden sm:block">
                Press Enter to open chat · Shift+Enter for new line
              </p>
              <button
                onClick={() => goToChat()}
                disabled={!heroQuery.trim()}
                className="btn-primary ml-auto"
                id="hero-verify-btn"
              >
                Verify
                <span className="material-symbols-outlined text-base" style={{ fontVariationSettings: "'FILL' 1" }}>send</span>
              </button>
            </div>
          </div>

          {/* Example queries */}
          <div className="flex flex-wrap gap-2 justify-center mt-5">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                onClick={() => goToChat(q)}
                className="btn-ghost text-xs"
                id={`example-query-${q.slice(0, 10).replace(/\s+/g, '-').toLowerCase()}`}
              >
                {q}
              </button>
            ))}
          </div>

          {/* Trust indicators */}
          <div className="flex flex-wrap items-center justify-center gap-6 mt-12 text-xs font-bold uppercase tracking-widest text-gray-400">
            {['No sign-up required', 'Fully open source', 'Zero LLM in verifier'].map((t) => (
              <span key={t} className="flex items-center gap-1.5">
                <span className="material-symbols-outlined text-sm" style={{ color: 'var(--accent)', fontVariationSettings: "'FILL' 1" }}>
                  check_circle
                </span>
                {t}
              </span>
            ))}
          </div>
        </section>

        {/* ── SECTION 2: How it Works ───────────────────────────────────── */}
        <HowItWorksSection />

        {/* ── SECTION 3: Technology ─────────────────────────────────────── */}
        <TechnologySection />

        {/* ── SECTION 4: Why Mirage ─────────────────────────────────────── */}
        <WhyMirageSection />

        {/* ── SECTION 5: Features ───────────────────────────────────────── */}
        <FeaturesSection />

        {/* ── SECTION 6: Pipeline Visualization ────────────────────────── */}
        <PipelineSection />

        {/* ── SECTION 7: About ──────────────────────────────────────────── */}
        <AboutSection />

        {/* ── SECTION 8: CTA — Get Started routes to /chat ─────────────── */}
        <CTASection onStart={() => goToChat()} />

      </main>
      <Footer />
    </>
  );
}
