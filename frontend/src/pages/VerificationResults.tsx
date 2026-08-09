import { useLocation, Navigate, Link } from 'react-router-dom';
import NavBar from '../components/NavBar';
import Footer from '../components/Footer';
import HallucinationMeter from '../components/HallucinationMeter';
import ClaimCard from '../components/ClaimCard';
import EvidenceCard from '../components/EvidenceCard';
import type { VerifyResponse } from '../types';

export default function VerificationResults() {
  const location = useLocation();
  const data = location.state as VerifyResponse | undefined;

  if (!data || !data.result) {
    return <Navigate to="/" />;
  }

  const { question, raw_answer, result } = data;

  return (
    <>
      <NavBar />
      <main className="flex-grow pt-24 pb-20 px-4 sm:px-6 w-full max-w-5xl mx-auto flex flex-col gap-8">
        
        {/* ── Question & Answer ────────────────────────────────────────── */}
        <section className="flex flex-col gap-6">
          <div className="flex justify-end w-full">
            <div className="bg-blue-50 border border-blue-100 rounded-2xl rounded-tr-sm p-4 max-w-3xl shadow-sm">
              <p className="text-base font-semibold text-blue-900">{question}</p>
            </div>
          </div>

          <div className="card p-6 bg-white relative">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center">
                <span className="material-symbols-outlined text-gray-700" style={{ fontVariationSettings: "'FILL' 1" }}>
                  robot_2
                </span>
              </div>
              <span className="text-sm font-bold text-gray-900 uppercase tracking-widest">Original Answer</span>
            </div>
            <div className="prose prose-slate max-w-none text-gray-800 leading-relaxed text-sm md:text-base whitespace-pre-wrap">
              {raw_answer}
            </div>
          </div>
        </section>

        {/* ── Hallucination Meter ──────────────────────────────────────── */}
        <section>
          <HallucinationMeter
            label={result.label}
            confidence={result.confidence_pct}
          />
        </section>

        {/* ── Entity Alignment & Summary ───────────────────────────────── */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Summary */}
          <div className="card p-6 bg-white">
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3">Assessment Summary</h3>
            <p className="text-sm text-gray-700 font-medium leading-relaxed">
              {result.explanation}
            </p>
          </div>

          {/* Entity Alignment */}
          <div className={`card p-6 bg-white border-l-4 ${result.entity_drift_detected ? 'border-red-500' : 'border-green-500'}`}>
            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-4">Entity Alignment</h3>
            <div className="flex flex-col gap-3 text-sm">
              <div className="flex justify-between items-start gap-4">
                <span className="text-gray-500 font-medium">Question Entity:</span>
                <span className="font-semibold text-gray-900 text-right">{result.primary_q_entity || 'None detected'}</span>
              </div>
              <div className="flex justify-between items-start gap-4">
                <span className="text-gray-500 font-medium">Answer Entity:</span>
                <span className="font-semibold text-gray-900 text-right">{result.primary_a_entity || 'None detected'}</span>
              </div>
              <div className="pt-3 border-t border-gray-100 flex justify-between items-center mt-1">
                <span className="text-gray-500 font-medium">Match Status:</span>
                {result.entity_drift_detected ? (
                  <span className="inline-flex items-center gap-1 text-red-700 font-bold bg-red-50 px-2 py-0.5 rounded text-xs">
                    <span className="material-symbols-outlined text-sm">error</span>
                    Drift Detected
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-green-700 font-bold bg-green-50 px-2 py-0.5 rounded text-xs">
                    <span className="material-symbols-outlined text-sm">check_circle</span>
                    Aligned
                  </span>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* ── Claim Analysis ───────────────────────────────────────────── */}
        {result.claim_verifications && result.claim_verifications.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900">Claim Analysis</h3>
              <span className="text-sm font-semibold text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                {result.claim_verifications.length} claims extracted
              </span>
            </div>
            <div className="flex flex-col gap-3">
              {result.claim_verifications.map((cv, idx) => (
                <ClaimCard key={idx} cv={cv} index={idx} />
              ))}
            </div>
          </section>
        )}

        {/* ── Evidence Sources ─────────────────────────────────────────── */}
        {result.evidence && result.evidence.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-4 mt-4">
              <h3 className="text-lg font-bold text-gray-900">Evidence Sources</h3>
              <span className="text-sm font-semibold text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                {result.evidence.length} sources retrieved
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {result.evidence.map((src, i) => {
                // Find best NLI score for this source across all claims
                let bestNli = 0;
                result.claim_verifications.forEach((cv) => {
                  cv.evidence_scores.forEach((es) => {
                    if (es.source_idx === i && es.nli_score) {
                      bestNli = Math.max(bestNli, es.nli_score.entailment);
                    }
                  });
                });
                const authScore = result.authority_scores[i] ?? 0.5;

                return (
                  <EvidenceCard
                    key={i}
                    source={src}
                    index={i}
                    authorityScore={authScore}
                    nliScore={bestNli}
                  />
                );
              })}
            </div>
          </section>
        )}

        {/* ── Technical Traces ─────────────────────────────────────────── */}
        {result.logic_trace && result.logic_trace.length > 0 && (
          <section className="mt-8">
            <details className="card bg-gray-50 group">
              <summary className="flex items-center justify-between p-4 cursor-pointer font-semibold text-sm text-gray-700 hover:text-gray-900 transition-colors">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-gray-400">terminal</span>
                  Verification Logic Trace
                </div>
                <span className="material-symbols-outlined text-gray-400 group-open:rotate-180 transition-transform">
                  expand_more
                </span>
              </summary>
              <div className="p-4 pt-0 border-t border-gray-200 mt-2">
                <ul className="list-disc pl-5 mt-4 text-xs font-mono text-gray-600 space-y-2">
                  {result.logic_trace.map((trace, idx) => (
                    <li key={idx}>{trace}</li>
                  ))}
                </ul>
              </div>
            </details>
          </section>
        )}
        
        {/* Bottom Actions */}
        <div className="flex justify-center mt-8">
          <Link to="/" className="btn-ghost">
            <span className="material-symbols-outlined text-sm">arrow_back</span>
            Verify Another Answer
          </Link>
        </div>

      </main>
      <Footer />
    </>
  );
}
