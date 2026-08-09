import { useState, useEffect } from 'react';

import HallucinationMeter from './HallucinationMeter';
import ClaimCard from './ClaimCard';
import EvidenceCard from './EvidenceCard';
import type { ConversationMessage } from '../types';

interface VerificationPanelProps {
  message: ConversationMessage | null;
}

const LOADING_STAGES = [
  'Generating AI Answer',
  'Extracting Claims',
  'Searching Live Web',
  'Ranking Evidence',
  'Running NLI',
  'Calculating Confidence',
  'Finalizing Report'
];

export default function VerificationPanel({ message }: VerificationPanelProps) {
  const [loadingStageIdx, setLoadingStageIdx] = useState(0);

  // Simulate loading stages progression while message is in 'loading' state
  useEffect(() => {
    if (message?.status === 'loading') {
      setLoadingStageIdx(0);
      const interval = setInterval(() => {
        setLoadingStageIdx((prev) => {
          if (prev < LOADING_STAGES.length - 1) return prev + 1;
          return prev;
        });
      }, 1500);
      return () => clearInterval(interval);
    }
  }, [message?.status]);

  if (!message) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-6 opacity-40 select-none">
        <span className="material-symbols-outlined text-5xl mb-4" style={{ fontVariationSettings: "'FILL' 1" }}>
          plagiarism
        </span>
        <h3 className="text-lg font-bold text-gray-900 uppercase tracking-widest">Verification Engine</h3>
        <p className="text-sm text-gray-500 mt-2 max-w-[200px]">
          Select an AI answer to view its live evidence report.
        </p>
      </div>
    );
  }

  if (message.status === 'loading') {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8">
        <div className="w-full max-w-xs space-y-6">
          <div className="flex flex-col items-center text-center mb-8">
            <svg className="animate-spin w-8 h-8 text-[#ff5e5b] mb-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-widest animate-pulse">
              Running Verification
            </h3>
          </div>

          <div className="space-y-4 relative">
            {/* Timeline track */}
            <div className="absolute left-[11px] top-2 bottom-2 w-0.5 bg-gray-200" />
            
            {LOADING_STAGES.map((stage, idx) => {
              const isPast = idx < loadingStageIdx;
              const isCurrent = idx === loadingStageIdx;
              const isFuture = idx > loadingStageIdx;
              
              return (
                <div key={stage} className={`flex items-center gap-4 relative z-10 transition-opacity duration-300 ${isFuture ? 'opacity-30' : 'opacity-100'}`}>
                  <div className={`
                    w-6 h-6 rounded-full flex items-center justify-center border-2 bg-white
                    ${isPast ? 'border-[#ff5e5b]' : isCurrent ? 'border-gray-900' : 'border-gray-300'}
                  `}>
                    {isPast ? (
                      <span className="material-symbols-outlined text-[12px] text-[#ff5e5b]" style={{ fontVariationSettings: "'FILL' 1" }}>check</span>
                    ) : isCurrent ? (
                      <div className="w-2 h-2 rounded-full bg-gray-900 animate-pulse" />
                    ) : null}
                  </div>
                  <span className={`text-xs font-bold uppercase tracking-wider ${isCurrent ? 'text-gray-900' : 'text-gray-500'}`}>
                    {stage}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  if (message.status === 'error' || !message.result) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <span className="material-symbols-outlined text-4xl text-red-500 mb-3">error</span>
        <h3 className="text-sm font-bold text-gray-900 uppercase tracking-widest mb-1">Verification Failed</h3>
        <p className="text-xs text-gray-500">{message.errorMessage}</p>
      </div>
    );
  }

  const { result } = message;

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-6 bg-[#FAF8F4]" style={{ minHeight: 0 }}>
      
      {/* ── Hallucination Meter ────────────────────────────────────── */}
      <section className="mb-8">
        <h3 className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">Trust Report</h3>
        <div className="bg-white border-2 border-[#111111] rounded-xl p-6 shadow-[4px_4px_0px_0px_#111111]">
          <HallucinationMeter label={result.label} confidence={result.confidence_pct} />
          
          <div className="mt-6 pt-4 border-t-2 border-gray-100 grid grid-cols-2 gap-4">
            <div>
              <p className="text-[10px] uppercase font-bold text-gray-400 tracking-widest mb-1">Claims Verified</p>
              <p className="text-xl font-black text-gray-900">{result.claims.length}</p>
            </div>
            <div>
              <p className="text-[10px] uppercase font-bold text-gray-400 tracking-widest mb-1">Live Sources</p>
              <p className="text-xl font-black text-gray-900">{result.evidence.length}</p>
            </div>
          </div>
        </div>
      </section>

      {/* ── Claim Analysis ─────────────────────────────────────────── */}
      <section className="mb-8">
        <h3 className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">Atomic Claims</h3>
        <div className="space-y-4">
          {result.claim_verifications.length > 0 ? (
            result.claim_verifications.map((cv, i) => (
              <ClaimCard key={i} cv={cv} index={i} />
            ))
          ) : (
            <div className="p-4 bg-white border-2 border-gray-200 rounded-lg text-sm text-gray-500 text-center">
              No verifiable factual claims detected in the response.
            </div>
          )}
        </div>
      </section>

      {/* ── Evidence Sources ───────────────────────────────────────── */}
      <section className="mb-8">
        <h3 className="text-xs font-black uppercase tracking-widest text-gray-400 mb-4">Evidence Sources</h3>
        <div className="flex flex-col gap-4">
          {result.evidence.map((src, i) => {
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
      
    </div>
  );
}
