import { useToast } from '../hooks/useToast';
import type { Toast, ToastType } from '../types';

const CONFIG: Record<ToastType, { bg: string; border: string; icon: string }> = {
  error:   { bg: 'bg-red-50',    border: 'border-red-400',   icon: 'error' },
  warning: { bg: 'bg-yellow-50', border: 'border-yellow-400', icon: 'warning' },
  success: { bg: 'bg-green-50',  border: 'border-green-400',  icon: 'check_circle' },
  info:    { bg: 'bg-blue-50',   border: 'border-blue-400',   icon: 'info' },
};

const TEXT_COLOR: Record<ToastType, string> = {
  error:   'text-red-700',
  warning: 'text-yellow-700',
  success: 'text-green-700',
  info:    'text-blue-700',
};

function ToastItem({ toast }: { toast: Toast }) {
  const { removeToast } = useToast();
  const cfg = CONFIG[toast.type];
  const textColor = TEXT_COLOR[toast.type];
  return (
    <div
      className={`flex items-start gap-3 p-4 rounded-lg border ${cfg.bg} ${cfg.border} shadow-md w-80 animate-slide-in`}
      role="alert"
    >
      <span className={`material-symbols-outlined text-xl shrink-0 ${textColor}`} style={{ fontVariationSettings: "'FILL' 1" }}>
        {cfg.icon}
      </span>
      <div className="flex-1 min-w-0">
        <p className={`text-sm font-semibold ${textColor}`}>{toast.title}</p>
        <p className={`text-xs mt-0.5 ${textColor} opacity-80`}>{toast.message}</p>
      </div>
      <button
        onClick={() => removeToast(toast.id)}
        className="shrink-0 text-gray-400 hover:text-gray-600 transition-colors"
        aria-label="Dismiss"
      >
        <span className="material-symbols-outlined text-base">close</span>
      </button>
    </div>
  );
}

export default function ToastContainer() {
  const { toasts } = useToast();
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2" role="region" aria-label="Notifications">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  );
}
