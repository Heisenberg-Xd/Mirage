// ─────────────────────────────────────────────────────────────────────────────
// NLI Score  (matches backend NLIScore dataclass)
// ─────────────────────────────────────────────────────────────────────────────
export interface NLIScore {
  entailment: number;
  contradiction: number;
  neutral: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Claim  (matches backend Claim dataclass)
// ─────────────────────────────────────────────────────────────────────────────
export interface Claim {
  text: string;
  raw_text: string;
  is_negated: boolean;
  key_entities: string[];
  is_relevant_to_question: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Evidence Score per source  (matches backend EvidenceScore dataclass)
// ─────────────────────────────────────────────────────────────────────────────
export interface EvidenceScore {
  source_idx: number;
  raw_relevance_score: number;
  sigmoid_relevance_score: number;
  nli_score: NLIScore | null;
  verdict: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single Claim Verification  (matches backend ClaimVerification dataclass)
// ─────────────────────────────────────────────────────────────────────────────
export type ClaimVerdict = 'supported' | 'insufficient' | 'contradicted' | 'ignored';

export interface ClaimVerification {
  claim: Claim;               // nested object — NOT a string
  verdict: ClaimVerdict;
  supporting_count: number;
  contradicting_count: number;
  insufficient_count: number;
  best_relevance_score: number;
  best_nli_entailment: number;
  best_source_idx: number;
  evidence_scores: EvidenceScore[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Evidence Source  (matches backend evidence list[dict] entries)
// ─────────────────────────────────────────────────────────────────────────────
export interface EvidenceSource {
  url: string;
  title: string;
  content: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Verification Result  (matches backend VerificationResult dataclass)
// ─────────────────────────────────────────────────────────────────────────────
export type HallucinationLabel = 'Not Hallucinating' | 'Cannot Verify' | 'Hallucinating';

export interface VerificationResult {
  label: HallucinationLabel;
  confidence_score: number;
  confidence_pct: number;
  explanation: string;
  claims: Claim[];                         // list of Claim objects
  claim_verifications: ClaimVerification[];
  evidence: EvidenceSource[];
  authority_scores: number[];
  question_entities: string[];
  answer_entities: string[];
  primary_q_entity: string;
  primary_a_entity: string;
  entity_drift_detected: boolean;
  has_false_hallucination: boolean;
  has_unverified_context: boolean;
  logic_trace: string[];
  supported_count: number;
  contradicted_count: number;
  insufficient_count: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// API Response  (matches backend /api/verify return shape)
// ─────────────────────────────────────────────────────────────────────────────
export interface VerifyResponse {
  question: string;
  raw_answer: string;
  result: VerificationResult;
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast
// ─────────────────────────────────────────────────────────────────────────────
export type ToastType = 'error' | 'warning' | 'success' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Loading Step
// ─────────────────────────────────────────────────────────────────────────────
export type LoadingStepStatus = 'waiting' | 'running' | 'done';

export interface LoadingStep {
  label: string;
  status: LoadingStepStatus;
}

// ─────────────────────────────────────────────────────────────────────────────
// Workspace Conversation Message
// ─────────────────────────────────────────────────────────────────────────────
export interface ConversationMessage {
  id: string;
  question: string;
  raw_answer: string | null; // null while loading
  result: VerificationResult | null; // null while loading
  status: 'loading' | 'complete' | 'error';
  errorMessage?: string;
  timestamp: Date;
}
