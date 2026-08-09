import type { VerifyResponse } from '../types';

const API = import.meta.env.VITE_API_URL as string;

// ─────────────────────────────────────────────────────────────────────────────
// verifyAnswer
// ─────────────────────────────────────────────────────────────────────────────
export async function verifyAnswer(question: string): Promise<VerifyResponse> {
  const res = await fetch(`${API}/api/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => 'Unknown error');
    throw new Error(`Server responded with ${res.status}: ${errorText}`);
  }

  return res.json() as Promise<VerifyResponse>;
}
