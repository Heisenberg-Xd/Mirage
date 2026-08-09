import type { LoadingStep } from '../types';

interface LoadingStepsProps {
  steps: LoadingStep[];
}

export default function LoadingSteps({ steps }: LoadingStepsProps) {
  return (
    <div className="w-full max-w-sm mx-auto py-8">
      <div className="flex flex-col gap-3">
        {steps.map((step, idx) => (
          <div key={idx} className="flex items-center gap-3">
            <div className="w-6 h-6 shrink-0 flex items-center justify-center">
              {step.status === 'done' ? (
                <span className="material-symbols-outlined text-green-500 text-xl" style={{ fontVariationSettings: "'FILL' 1" }}>
                  check_circle
                </span>
              ) : step.status === 'running' ? (
                <svg className="animate-spin w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : (
                <div className="w-4 h-4 rounded-full border-2 border-gray-200" />
              )}
            </div>
            <span
              className={`text-sm font-medium transition-colors ${
                step.status === 'done'
                  ? 'text-green-600'
                  : step.status === 'running'
                  ? 'text-blue-600'
                  : 'text-gray-400'
              }`}
            >
              {step.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
