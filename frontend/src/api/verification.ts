import { api } from '../utils/api';
import type { VerifyResponse } from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// verifyAnswer
// ─────────────────────────────────────────────────────────────────────────────
export async function verifyAnswer(
  question: string,
  options?: { onRetry?: (attempt: number) => void }
): Promise<VerifyResponse> {
  return api.post<VerifyResponse>(
    '/api/verify',
    { question },
    { onRetry: options?.onRetry }
  );
}
