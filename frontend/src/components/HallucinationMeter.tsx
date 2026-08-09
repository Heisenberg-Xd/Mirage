import { useEffect, useRef, useState } from 'react';
import type { HallucinationLabel } from '../types';

interface HallucinationMeterProps {
  label: HallucinationLabel;
  confidence: number;
}

const CONFIG: Record<HallucinationLabel, { color: string; bg: string; icon: string; textColor: string }> = {
  'Not Hallucinating': { color: '#22C55E', bg: '#f0fdf4', icon: 'verified',  textColor: '#15803d' },
  'Cannot Verify':     { color: '#F59E0B', bg: '#fffbeb', icon: 'help',      textColor: '#92400e' },
  'Hallucinating':     { color: '#EF4444', bg: '#fef2f2', icon: 'dangerous', textColor: '#991b1b' },
};

// Map label to a 0–100 "risk" position for the gradient meter
function labelToPosition(label: HallucinationLabel, confidence: number): number {
  if (label === 'Not Hallucinating') return Math.round(confidence * 0.33);
  if (label === 'Cannot Verify') return Math.round(33 + confidence * 0.34);
  return Math.round(66 + confidence * 0.34);
}

export default function HallucinationMeter({ label, confidence }: HallucinationMeterProps) {
  const cfg = CONFIG[label];
  const position = labelToPosition(label, confidence);
  const [animated, setAnimated] = useState(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    let start = 0;
    const step = (ts: number) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / 800, 1);
      // ease-out
      setAnimated(Math.round(position * (1 - Math.pow(1 - progress, 3))));
      if (progress < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
  }, [position]);

  return (
    <div className="rounded-xl border border-gray-200 p-6 shadow-sm bg-white">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center"
            style={{ background: cfg.bg }}
          >
            <span
              className="material-symbols-outlined text-xl"
              style={{ color: cfg.color, fontVariationSettings: "'FILL' 1" }}
            >
              {cfg.icon}
            </span>
          </div>
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-500">Assessment</p>
            <p className="text-xl font-bold" style={{ color: cfg.textColor }}>{label}</p>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-500">Confidence</p>
          <p className="text-2xl font-black" style={{ color: cfg.color }}>{confidence}%</p>
        </div>
      </div>

      {/* Meter bar */}
      <div className="relative">
        {/* Gradient track */}
        <div
          className="h-4 w-full rounded-full overflow-hidden"
          style={{
            background: 'linear-gradient(to right, #22C55E 0%, #22C55E 33%, #F59E0B 33%, #F59E0B 66%, #EF4444 66%, #EF4444 100%)',
            opacity: 0.25,
          }}
        />
        {/* Filled bar */}
        <div
          className="absolute top-0 left-0 h-4 rounded-full transition-all"
          style={{
            width: `${animated}%`,
            background: `linear-gradient(to right, #22C55E 0%, #22C55E 33%, #F59E0B 33%, #F59E0B 66%, #EF4444 66%)`,
            opacity: 0.9,
          }}
        />
        {/* Thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-5 h-5 rounded-full border-2 border-white shadow-md transition-all"
          style={{ left: `calc(${animated}% - 10px)`, background: cfg.color }}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between mt-2 text-xs font-medium text-gray-400">
        <span>Not Hallucinating</span>
        <span>Cannot Verify</span>
        <span>Hallucinating</span>
      </div>
    </div>
  );
}
