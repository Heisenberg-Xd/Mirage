import { useState } from 'react';
import type { ClaimVerification, ClaimVerdict } from '../types';

interface ClaimCardProps {
  cv: ClaimVerification;
  index: number;
}

const VERDICT_CONFIG: Record<ClaimVerdict, { label: string; color: string; bg: string; icon: string }> = {
  supported:    { label: 'Supported',    color: '#15803d', bg: '#f0fdf4', icon: 'check_circle' },
  insufficient: { label: 'Insufficient', color: '#92400e', bg: '#fffbeb', icon: 'help'         },
  contradicted: { label: 'Contradicted', color: '#991b1b', bg: '#fef2f2', icon: 'cancel'       },
  ignored:      { label: 'Ignored',      color: '#6b7280', bg: '#f9fafb', icon: 'block'        },
};

export default function ClaimCard({ cv, index }: ClaimCardProps) {
  const [open, setOpen] = useState(false);
  const cfg = VERDICT_CONFIG[cv.verdict] ?? VERDICT_CONFIG.insufficient;

  // cv.claim is a Claim object (from the backend dataclass), not a string.
  const claimText   = cv.claim?.text     ?? cv.claim?.raw_text ?? '';
  const isNegated   = cv.claim?.is_negated           ?? false;
  const isRelevant  = cv.claim?.is_relevant_to_question ?? true;
  const keyEntities = cv.claim?.key_entities         ?? [];

  // Backend field names from ClaimVerification dataclass (via asdict())
  const entPct = Math.round((cv.best_nli_entailment ?? 0) * 100);
  const relPct = Math.round((cv.best_relevance_score ?? 0) * 100);
  const supCount = cv.supporting_count ?? 0;

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden">
      {/* Header — always visible */}
      <button
        className="w-full flex items-center gap-3 p-4 text-left hover:bg-gray-50 transition-colors"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="shrink-0 text-xs font-bold text-gray-400 w-5">{index + 1}.</span>
        {/* Verdict badge */}
        <span
          className="shrink-0 inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold"
          style={{ color: cfg.color, background: cfg.bg }}
        >
          <span className="material-symbols-outlined text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>
            {cfg.icon}
          </span>
          {cfg.label}
        </span>
        {/* Claim text — render .text string, NOT the object */}
        <span className="flex-1 text-sm font-medium text-gray-800 min-w-0 truncate">{claimText}</span>
        {/* Scores */}
        <div className="shrink-0 hidden sm:flex items-center gap-4 text-xs text-gray-500 mr-2">
          <span>
            <span className="font-semibold">Ent:</span>{' '}
            <span className="font-bold text-blue-600">{entPct}%</span>
          </span>
          <span>
            <span className="font-semibold">Rel:</span>{' '}
            <span className="font-bold text-purple-600">{relPct}%</span>
          </span>
        </div>
        <span
          className={`material-symbols-outlined text-gray-400 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
        >
          expand_more
        </span>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="border-t border-gray-100 px-4 pb-4 pt-4 bg-gray-50">
          {/* Full claim text */}
          <p className="text-sm font-medium text-gray-800 mb-4">{claimText}</p>

          {/* Metrics grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <MetricBox label="Verdict"     value={cfg.label}          valueStyle={{ color: cfg.color }} />
            <MetricBox label="Entailment"  value={`${entPct}%`}       valueStyle={{ color: '#2563eb' }} />
            <MetricBox label="Relevance"   value={`${relPct}%`}       valueStyle={{ color: '#7c3aed' }} />
            <MetricBox label="Supporting"  value={String(supCount)}   valueStyle={{ color: '#15803d' }} />
          </div>

          {/* Key entities */}
          {keyEntities.length > 0 && (
            <div className="flex flex-wrap gap-1 mb-3">
              {keyEntities.map((ent, i) => (
                <span key={i} className="px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-600">
                  {ent}
                </span>
              ))}
            </div>
          )}

          {/* Tags */}
          <div className="flex flex-wrap gap-2">
            {isNegated && (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-purple-100 text-purple-700">Negated</span>
            )}
            {!isRelevant && (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-gray-200 text-gray-600">Additional Context</span>
            )}
            {isRelevant && (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-700">Relevant to Question</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricBox({
  label,
  value,
  valueStyle,
}: {
  label: string;
  value: string;
  valueStyle?: React.CSSProperties;
}) {
  return (
    <div className="rounded-lg bg-white border border-gray-200 p-3 text-center">
      <p className="text-xs text-gray-500 uppercase tracking-wide font-semibold mb-1">{label}</p>
      <p className="text-base font-bold" style={valueStyle}>{value}</p>
    </div>
  );
}
