import type { EvidenceSource } from '../types';

interface EvidenceCardProps {
  source: EvidenceSource;
  authorityScore: number;
  nliScore: number;
  index: number;
}

function getAuthorityLabel(score: number): { label: string; color: string; bg: string } {
  if (score >= 0.7) return { label: 'High Authority', color: '#15803d', bg: '#f0fdf4' };
  if (score >= 0.4) return { label: 'Medium Authority', color: '#92400e', bg: '#fffbeb' };
  return { label: 'Low Authority', color: '#991b1b', bg: '#fef2f2' };
}

function getDomain(url: string): string {
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return url;
  }
}

function getFaviconUrl(url: string): string {
  try {
    const { protocol, hostname } = new URL(url);
    return `${protocol}//${hostname}/favicon.ico`;
  } catch {
    return '';
  }
}

export default function EvidenceCard({ source, authorityScore, nliScore, index }: EvidenceCardProps) {
  const authority = getAuthorityLabel(authorityScore);
  const domain = getDomain(source.url);
  const favicon = getFaviconUrl(source.url);
  const snippet = source.content?.slice(0, 280) ?? '';

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm hover:shadow-md transition-shadow">
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 shrink-0 rounded bg-gray-100 flex items-center justify-center overflow-hidden">
            <img
              src={favicon}
              alt={domain}
              className="w-4 h-4"
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = 'none';
                (e.target as HTMLImageElement).parentElement!.innerHTML =
                  `<span class="text-xs font-bold text-gray-500">${domain[0]?.toUpperCase() ?? '#'}</span>`;
              }}
            />
          </div>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">{source.title || `Source ${index + 1}`}</p>
            <p className="text-xs text-gray-400">{domain}</p>
          </div>
        </div>
        <span
          className="shrink-0 px-2.5 py-1 rounded-full text-xs font-semibold"
          style={{ color: authority.color, background: authority.bg }}
        >
          {authority.label}
        </span>
      </div>

      {/* Snippet */}
      <p className="text-sm text-gray-600 leading-relaxed mb-4">
        {snippet}{snippet.length < (source.content?.length ?? 0) ? '…' : ''}
      </p>

      {/* Footer row */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span>
            <span className="font-semibold text-gray-700">NLI:</span>{' '}
            <span className="font-bold text-blue-600">{Math.round(nliScore * 100)}%</span>
          </span>
          <span>
            <span className="font-semibold text-gray-700">Auth:</span>{' '}
            <span className="font-bold" style={{ color: authority.color }}>{Math.round(authorityScore * 100)}</span>
          </span>
        </div>
        <a
          href={source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-xs font-semibold text-blue-600 hover:text-blue-800 transition-colors border border-blue-200 rounded-md px-3 py-1.5 hover:bg-blue-50"
        >
          Open Source
          <span className="material-symbols-outlined text-sm">open_in_new</span>
        </a>
      </div>
    </div>
  );
}
