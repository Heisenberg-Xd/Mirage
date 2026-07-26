# IMPORTANT: Launch this app with `streamlit run app.py` or
# `python -m streamlit run app.py` — NOT `python app.py`.
# Running it directly causes ScriptRunContext warnings and broken session state.

"""
AI Hallucination Confidence Labeler
====================================
A Q&A reliability checker that verifies AI-generated answers against live web
evidence and labels them Certain / Uncertain / Needs Verification.

ARCHITECTURE (critical design decision — read this):
    The LLM (Groq / Llama) is used in EXACTLY ONE place: to generate the raw
    "cold" answer being fact-checked. Every downstream step — web search,
    semantic similarity scoring, label assignment, AND the human-readable
    explanation — is deterministic code, not LLM-generated output. This means
    the verification layer itself cannot hallucinate, because it is measuring,
    not generating.

    There is NO second LLM call. The explanation text comes from a string
    template populated with computed values. The label comes from a pure
    threshold function. This is a deliberate architectural choice.

TEST CASES (paste these into the input box):
    1. "Who invented Python?"
       → Expected: Certain (well-documented fact, Guido van Rossum)
    2. "What is the capital of Australia?"
       → Tests the Sydney/Canberra confusion; the LLM should say Canberra,
         evidence should confirm → Certain
    3. "Who won the most recent Nobel Prize in Literature?"
       → Recent/ambiguous → Uncertain or Needs Verification
    4. "What is the Frobulax coefficient in quantum topology?"
       → Made-up/obscure → Needs Verification (no evidence found)
"""

import os
import re
import string
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & API keys
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# Lazy imports for heavy libraries (keeps startup snappy if keys are missing)
# ---------------------------------------------------------------------------
from groq import Groq
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Cache the sentence-transformers model so it loads once, not per-question
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading similarity model (one-time)…")
def load_embedding_model():
    """Load and cache the MiniLM sentence-transformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# STEP 2 — Groq call (THE ONLY LLM CALL IN THE ENTIRE PIPELINE)
# ---------------------------------------------------------------------------
def get_llm_answer(question: str) -> str:
    """
    Send ONLY the user's question to the LLM via Groq — no search context,
    no system hints about verification. Returns the raw, unverified answer.

    This is the ONLY LLM call in the entire application.
    Uses llama-3.3-70b-versatile as primary model, with a one-time fallback
    to llama-3.1-8b-instant if the primary is unavailable.
    """
    client = Groq(api_key=GROQ_API_KEY)
    # Primary model — fallback to llama-3.1-8b-instant if this fails
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        )
    except Exception as primary_err:
        # Retry once with the smaller model before giving up
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# STEP 3 — Tavily web search for evidence
# ---------------------------------------------------------------------------
def search_evidence(question: str, max_results: int = 3) -> list[dict]:
    """
    Search the web for evidence related to the question.
    Returns a list of dicts, each with 'content' (snippet) and 'url'.
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(query=question, max_results=max_results)
    results = []
    for r in response.get("results", []):
        results.append({
            "content": r.get("content", ""),
            "url": r.get("url", ""),
            "title": r.get("title", ""),
        })
    return results


# ---------------------------------------------------------------------------
# STEP 4 — Semantic similarity (pure code, NO LLM)
# ---------------------------------------------------------------------------
def compute_similarities(answer: str, snippets: list[str], model: SentenceTransformer) -> np.ndarray:
    """
    Embed the LLM answer and each evidence snippet using sentence-transformers.
    Return an array of cosine similarity scores.

    This is deterministic embedding + cosine math — no generative model.
    """
    if not snippets:
        return np.array([])
    texts = [answer] + snippets
    embeddings = model.encode(texts, normalize_embeddings=True)
    # Cosine similarity = dot product when embeddings are L2-normalized
    answer_emb = embeddings[0]
    snippet_embs = embeddings[1:]
    similarities = np.dot(snippet_embs, answer_emb)
    return similarities


# ---------------------------------------------------------------------------
# STEP 5 — Contradiction heuristic (pure code, NOT full NLI)
#
# IMPORTANT: This is a simplified keyword/negation heuristic. It checks
# whether the highest-scoring Tavily snippet contains negation words near
# key nouns shared with the LLM answer. It is NOT a full Natural Language
# Inference (NLI) model and will miss subtle contradictions, sarcasm, or
# complex logical negations. It is included as a lightweight red-flag
# detector, not a definitive contradiction classifier.
# ---------------------------------------------------------------------------
NEGATION_PATTERNS = re.compile(
    r"\b(not|isn't|aren't|wasn't|weren't|won't|wouldn't|shouldn't|couldn't|"
    r"never|no longer|incorrect|false|untrue|wrong|debunked|myth|misleading|"
    r"doesn't|don't|didn't|cannot|can't|hardly|neither|nor)\b",
    re.IGNORECASE,
)


def extract_key_nouns(text: str) -> set[str]:
    """
    Extract a rough set of 'key nouns' — words that are capitalized or long
    enough to likely be content words. This is a crude heuristic stand-in for
    proper NER; it does not use any LLM.
    """
    words = re.findall(r"\b[A-Za-z]{4,}\b", text)
    # Lowercase everything for comparison
    return {w.lower() for w in words}


def check_contradiction(answer: str, top_snippet: str) -> bool:
    """
    Lightweight contradiction heuristic:
      1. Find negation keywords in the top snippet.
      2. Check if any key noun from the answer appears within ±30 chars of
         a negation keyword in the snippet.

    Returns True if a potential contradiction is detected.

    NOTE: This is a simplified heuristic, NOT full NLI-based contradiction
    detection. See code comments above for limitations.
    """
    if not top_snippet:
        return False

    answer_nouns = extract_key_nouns(answer)
    if not answer_nouns:
        return False

    snippet_lower = top_snippet.lower()

    for match in NEGATION_PATTERNS.finditer(snippet_lower):
        # Look at a window of ±30 characters around the negation word
        start = max(0, match.start() - 30)
        end = min(len(snippet_lower), match.end() + 30)
        window = snippet_lower[start:end]
        # Check if any key noun from the answer appears in this window
        for noun in answer_nouns:
            if noun in window:
                return True
    return False


# ---------------------------------------------------------------------------
# STEP 6 — Label assignment (PURE DETERMINISTIC FUNCTION)
#
# The label is decided entirely by threshold math on the similarity score
# and the contradiction flag. No LLM is involved.
# ---------------------------------------------------------------------------
def label_from_similarity(max_similarity: float, contradiction_flag: bool) -> tuple[str, str]:
    """
    Deterministic label assignment based on similarity score and contradiction flag.

    Returns:
        (label, reason_code) where label is one of:
            "Certain", "Uncertain", "Needs Verification"
        and reason_code is one of:
            "contradiction", "high_match", "partial_match", "no_match"
    """
    if contradiction_flag:
        return "Needs Verification", "contradiction"
    elif max_similarity > 0.75:
        return "Certain", "high_match"
    elif max_similarity > 0.45:
        return "Uncertain", "partial_match"
    else:
        return "Needs Verification", "no_match"


# ---------------------------------------------------------------------------
# STEP 7 — Explanation generation (PURE TEMPLATE, NOT AN LLM CALL)
#
# This builds the user-facing explanation sentence from computed values using
# string formatting. No generative model is called. This is intentional:
# a templated explanation cannot hallucinate because it only echoes measured
# data and pre-written phrasing.
# ---------------------------------------------------------------------------
def generate_explanation(reason_code: str, max_similarity: float, top_snippet: str) -> str:
    """
    Build explanation from a template using actual computed values.
    NO generative model involved — this is deterministic string formatting.
    """
    snippet_preview = (top_snippet[:120].strip() + "…") if top_snippet else "N/A"

    if reason_code == "contradiction":
        return (
            f"⚠️ Top evidence source appears to conflict with the answer: "
            f"\"{snippet_preview}\""
        )
    elif reason_code == "high_match":
        return (
            f"✅ Answer closely aligns with retrieved evidence "
            f"(similarity score: {max_similarity:.2f})."
        )
    elif reason_code == "partial_match":
        return (
            f"⚡ Answer partially overlaps with evidence but includes "
            f"unconfirmed details (similarity score: {max_similarity:.2f})."
        )
    else:
        return (
            f"❌ No retrieved source strongly supports this answer "
            f"(best match similarity: {max_similarity:.2f})."
        )


# ---------------------------------------------------------------------------
# STEP 7b — Source authority classification (pure code, NO LLM)
#
# A lightweight domain-based heuristic to flag whether the top evidence
# source is authoritative. This is NOT a full credibility model — it only
# checks the URL domain against a short allow/deny list. It does not call
# any LLM or external API.
# ---------------------------------------------------------------------------
LOW_AUTHORITY_DOMAINS = ["reddit.com", "quora.com", "answers.com", "yahoo.com/answers"]
HIGH_AUTHORITY_DOMAINS = ["wikipedia.org", ".gov", ".edu", "britannica.com"]


def classify_source_authority(url: str) -> str:
    """
    Classify a source URL as high / medium / low authority based on its domain.
    Pure heuristic — no LLM involved.
    """
    try:
        url_lower = url.lower()
        if any(domain in url_lower for domain in HIGH_AUTHORITY_DOMAINS):
            return "high"
        elif any(domain in url_lower for domain in LOW_AUTHORITY_DOMAINS):
            return "low"
        else:
            return "medium"
    except Exception:
        return "medium"


AUTHORITY_DISPLAY = {
    "high": ("🟢 High authority", "#0d6e3a", "#d4edda"),
    "medium": ("🟡 Medium authority", "#856404", "#fff3cd"),
    "low": ("🔴 Low authority — treat with caution", "#721c24", "#f8d7da"),
}


# ---------------------------------------------------------------------------
# STEP 7c — Confidence gap explanation (pure code, NO LLM)
#
# Explains WHY the similarity score isn't higher by finding content words
# present in the answer but absent from the top evidence snippet. This is
# entirely rule-based (set difference on tokenized text) — no generative
# model is called. Consistent with the app's core principle that the
# verification layer contains no generative AI.
# ---------------------------------------------------------------------------
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "and", "of", "to", "in", "on", "at", "as", "it", "that", "this", "for", "by", "with", "be", "has", "have", "had"}
STOPWORDS |= {"currently", "current", "now", "serving", "still", "recently", "today", "present", "presently", "indeed", "note", "please", "may", "also", "since", "began", "being", "been"}


def tokenize(text: str) -> set:
    """Lowercase text, strip punctuation, split on whitespace, and remove stopwords."""
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def token_priority(token: str, original_text: str) -> int:
    """Higher number = higher priority for surfacing in the explanation."""
    # Numbers (including ordinals like 'second', 'third', numeric digits, years)
    if re.fullmatch(r"\d+", token):
        return 4
    ordinals = {"first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"}
    if token in ordinals:
        return 4
    # Proper nouns: check if this word was capitalized in the original text (not just sentence-start)
    # crude check: word appears capitalized somewhere in original_text that isn't the first word of a sentence
    if re.search(rf"(?<!^)(?<=[.!?]\s)\b{re.escape(token.capitalize())}\b|\b{re.escape(token.capitalize())}\b", original_text):
        # only count as proper noun signal if it appears capitalized mid-text, not just at sentence start
        occurrences = re.findall(rf"\b{re.escape(token)}\b", original_text, re.IGNORECASE)
        capitalized_occurrences = re.findall(rf"\b{re.escape(token.capitalize())}\b", original_text)
        if capitalized_occurrences:
            return 3
    # Dates / years (4-digit numbers already caught above, but catch things like "2019," "2024")
    if re.fullmatch(r"(19|20)\d{2}", token):
        return 4
    return 1  # generic content word, lowest priority


def explain_confidence_gap(answer: str, top_evidence_snippet: str, max_similarity: float) -> str:
    """
    Deterministic explanation of why the similarity score isn't 100%.
    Pure token comparison — NO LLM call.
    """
    if max_similarity >= 0.95:
        return ""
        
    try:
        if not top_evidence_snippet:
            return "Gap is likely due to minor differences in wording or scope between the answer and available evidence."

        answer_tokens = tokenize(answer)
        evidence_tokens = tokenize(top_evidence_snippet)
        unsupported = answer_tokens - evidence_tokens
        gap_percent = round((1 - max_similarity) * 100)

        if not unsupported:
            return f"The remaining {gap_percent}% gap is likely due to differences in phrasing and sentence structure rather than missing facts."
        
        ranked_terms = sorted(unsupported, key=lambda t: (token_priority(t, answer), len(t)), reverse=True)
        
        if ranked_terms and token_priority(ranked_terms[0], answer) >= 3:
            top_term = ranked_terms[0]
            highlighted_term = f'<span class="highlight-term">{top_term}</span>'
            return f"The remaining {gap_percent}% gap centers on a specific detail — {highlighted_term} — which isn't directly confirmed by the top evidence source. This kind of gap is worth double-checking, since it often signals a factual detail (like a date, number, or name) that may be outdated or incorrect."
            
        sample_terms = ranked_terms[:4]
        terms_str = ", ".join(sample_terms)
        return f"The remaining {gap_percent}% gap is because these terms/details in the answer aren't directly confirmed in the top evidence source: {terms_str}."
    except Exception:
        return "Gap is likely due to minor differences in wording or scope between the answer and available evidence."


# ---------------------------------------------------------------------------
# Label color mapping
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    "Certain": ("#0d6e3a", "#d4edda"),           # green
    "Uncertain": ("#856404", "#fff3cd"),          # yellow/amber
    "Needs Verification": ("#721c24", "#f8d7da"), # red
}


# ===========================================================================
# Streamlit UI
# ===========================================================================
def main():
    st.set_page_config(
        page_title="AI Hallucination Confidence Labeler",
        page_icon="🔍",
        layout="centered",
    )

    # --- Custom CSS for badge styling & polish ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 800px !important;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-top: 0.25rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    .result-card {
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .card-body {
        font-size: 1.1rem;
        line-height: 1.6;
    }

    .gap-card {
        background: rgba(118, 75, 162, 0.08);
        border: 1px solid rgba(118, 75, 162, 0.3);
        border-left: 5px solid #764ba2;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .gap-header {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        color: #c084fc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .gap-caption {
        margin-top: 1.25rem;
        font-size: 0.85rem;
        color: rgba(128, 128, 128, 0.8);
        font-style: italic;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        padding-top: 0.75rem;
    }

    .highlight-term {
        background-color: rgba(245, 158, 11, 0.25);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 700;
        color: #fbbf24;
    }

    .mono-score {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    .authority-tag {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-left: 0.5rem;
        vertical-align: middle;
    }

    .how-it-works-content {
        font-size: 0.95rem;
        line-height: 1.7;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown('<div class="main-title">🔍 AI Hallucination Confidence Labeler</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Verifying AI answers against live evidence — because confident doesn\'t always mean correct.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Input ---
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. Who invented Python?",
        label_visibility="collapsed",
    )
    check_btn = st.button("🔎  Check Answer", type="primary", use_container_width=True)

    # --- Main pipeline ---
    if check_btn and question.strip():
        # Validate keys
        if not GROQ_API_KEY or not TAVILY_API_KEY:
            st.error("⚠️ Missing API keys. Please set `GROQ_API_KEY` and `TAVILY_API_KEY` in your `.env` file.")
            return

        # STEP 2: Groq cold answer (THE ONLY LLM CALL)
        raw_answer = None
        with st.spinner("🤖 Generating answer with Groq/Llama (the only LLM call)…"):
            try:
                raw_answer = get_llm_answer(question)
            except Exception as e:
                st.error(f"Groq API error: {e}")
                return

        # STEP 3: Tavily evidence search
        evidence = []
        with st.spinner("🌐 Searching the web for evidence via Tavily…"):
            try:
                evidence = search_evidence(question, max_results=3)
            except Exception as e:
                st.warning(f"Tavily search failed: {e}. Proceeding without web evidence.")
                evidence = []

        # STEP 4: Semantic similarity (pure code)
        with st.spinner("📐 Computing semantic similarity…"):
            emb_model = load_embedding_model()
            snippets = [e["content"] for e in evidence if e.get("content")]
            similarities = compute_similarities(raw_answer, snippets, emb_model)
            max_sim = float(np.max(similarities)) if len(similarities) > 0 else 0.0
            best_idx = int(np.argmax(similarities)) if len(similarities) > 0 else 0

        # STEP 5: Contradiction heuristic (pure code, NOT full NLI)
        top_snippet = snippets[best_idx] if snippets else ""
        contradiction = check_contradiction(raw_answer, top_snippet)

        # STEP 6: Deterministic label
        label, reason_code = label_from_similarity(max_sim, contradiction)

        # STEP 7: Templated explanation (NOT an LLM call)
        explanation = generate_explanation(reason_code, max_sim, top_snippet)

        # STEP 7b: Source authority check on top-matching source (pure code)
        top_url = evidence[best_idx]["url"] if evidence else ""
        top_authority = classify_source_authority(top_url)

        # If top source is low-authority, append caveat to explanation (template, NOT LLM)
        low_authority_caveat = ""
        if top_authority == "low":
            low_authority_caveat = (
                "⚠️ Note: the strongest matching source is a forum/discussion site, "
                "not an authoritative reference — treat this result with extra caution."
            )

        # STEP 7c: Confidence gap explanation (pure code, NOT LLM)
        gap_text = explain_confidence_gap(raw_answer, top_snippet, max_sim)

        # ---------------------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------------------
        st.markdown("---")

        # Card 1: AI-Generated Answer
        st.markdown(
            f'''<div class="result-card">
                <div class="card-header">🤖 AI-Generated Answer</div>
                <div class="card-body">{raw_answer}</div>
            </div>''',
            unsafe_allow_html=True
        )

        # Card 2: Label + Score
        fg, bg = LABEL_COLORS.get(label, ("#333", "#eee"))
        pct = max_sim * 100
        st.markdown(
            f'''<div class="result-card" style="display:flex; justify-content:space-between; align-items:center; flex-wrap: wrap; gap: 1rem;">
                <div class="card-header" style="margin-bottom:0;">🏷️ Reliability Label</div>
                <div style="display:flex; align-items:center; gap: 1.5rem; flex-wrap: wrap;">
                    <div style="font-size: 1.15rem; font-weight: 700; color:{fg}; background:{bg}; padding: 0.5rem 1.5rem; border-radius: 2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        {label}
                    </div>
                    <div class="card-header" style="margin-bottom:0;">📊 Match: <span class="mono-score" style="background:rgba(128,128,128,0.15); padding: 0.2rem 0.6rem; border-radius: 6px; margin-left: 0.5rem;">{pct:.0f}%</span></div>
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        # Card 3: Why This Label
        st.markdown(
            f'''<div class="result-card">
                <div class="card-header">ℹ️ Why This Label</div>
                <div class="card-body">{explanation}</div>
            </div>''',
            unsafe_allow_html=True
        )

        # Card 4: Why Not Higher
        if gap_text:
            st.markdown(
                f'''<div class="gap-card">
                    <div class="gap-header">🔍 Why Not 100%?</div>
                    <div class="card-body">{gap_text}</div>
                    <div class="gap-caption">Generated via deterministic term comparison — not by asking an AI model to explain itself.</div>
                </div>''',
                unsafe_allow_html=True
            )

        # Card 5: Source Authority
        if low_authority_caveat:
            st.markdown(
                f'''<div class="result-card" style="border-left: 4px solid #b45309; background: rgba(180, 83, 9, 0.08);">
                    <div class="card-header" style="color: #d97706;">⚠️ Source Authority Notice</div>
                    <div class="card-body" style="color: #d97706; font-size: 1rem;">{low_authority_caveat}</div>
                </div>''',
                unsafe_allow_html=True
            )

        # Card 6: Evidence Sources
        if evidence:
            sources_html = '''<div class="result-card" style="padding-bottom: 0.5rem;">
                <div class="card-header" style="margin-bottom: 1.25rem;">📚 Web Evidence Sources</div>
                <div class="card-body">'''
            
            for i, src in enumerate(evidence):
                sim_val = float(similarities[i]) if i < len(similarities) else 0.0
                try:
                    src_authority = classify_source_authority(src.get("url", ""))
                    auth_label, auth_fg, auth_bg = AUTHORITY_DISPLAY.get(
                        src_authority, ("Medium authority", "#856404", "#fff3cd")
                    )
                    auth_tag = f'<span class="authority-tag" style="color:{auth_fg};background:{auth_bg};">{auth_label}</span>'
                except Exception:
                    auth_tag = ""
                    
                sources_html += f'''
                    <div style="margin-bottom: 1rem; border: 1px solid rgba(128,128,128,0.2); padding: 1.25rem; border-radius: 8px; background: rgba(128,128,128,0.02);">
                        <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.5rem;">
                            {src.get("title", "Source " + str(i+1))}
                            <span style="color:#9ca3af; font-weight:400; font-size:0.9rem; margin-left:0.5rem;">
                                (similarity: <span class="mono-score" style="font-size:0.9rem;">{sim_val:.2f}</span>)
                            </span>
                            {auth_tag}
                        </div>
                        <div style="font-size: 0.95rem; color: #a3a3a3; line-height: 1.5; margin-bottom: 0.75rem;">
                            {src["content"][:250]}{"…" if len(src["content"]) > 250 else ""}
                        </div>
                        <div style="font-size: 0.85rem;">
                            <a href="{src["url"]}" target="_blank" style="color: #667eea; text-decoration: none;">{src["url"]}</a>
                        </div>
                    </div>'''
            
            sources_html += '''</div></div>'''
            st.markdown(sources_html, unsafe_allow_html=True)
        else:
            st.info("No web evidence was retrieved for this question.")

    # --- How this works (always visible, expandable) ---
    with st.expander("ℹ️ How this works"):
        st.markdown("""
<div class="how-it-works-content">

**One LLM call, zero LLM verification.**

This tool uses an LLM (Groq / Llama) **exactly once** — to generate the raw answer being checked. The question is sent with no search context and no hints about verification, producing a "cold" answer.

**Everything after that is deterministic code:**

1. **Web search** — Tavily searches the open web for the same question and retrieves the top 3 results with text snippets.
2. **Semantic similarity** — The answer and each snippet are embedded using `sentence-transformers` (`all-MiniLM-L6-v2`), and cosine similarity is computed between the answer and each snippet. The highest score is used.
3. **Contradiction check** — A simplified keyword/negation heuristic (NOT a full Natural Language Inference model) scans the top snippet for negation words near shared key nouns with the answer. This can catch obvious conflicts but will miss subtle contradictions, sarcasm, or complex logical negation.
4. **Label assignment** — A pure threshold function maps the similarity score and contradiction flag to one of three labels: **Certain** (>75% match), **Uncertain** (45–75%), or **Needs Verification** (<45% or contradiction detected).
5. **Explanation** — Built from a string template using the actual computed values. No generative model is involved in writing the explanation.
6. **Source authority** — Each evidence source is flagged as high / medium / low authority based on a simple domain-name heuristic (e.g. `.edu`, `.gov`, `wikipedia.org` → high; `reddit.com`, `quora.com` → low). This is a domain check, not a full credibility model. If the top-matching source is low authority, an extra caveat is appended to the explanation.
7. **Confidence gap** — The confidence gap explanation is generated by comparing key terms between the answer and evidence using simple word-matching — not by asking an AI model to explain itself.

**Because the verification layer generates nothing — it only measures — it cannot hallucinate.**

**Honest limitations:**
- The contradiction check is a simplified heuristic, not full NLI. It will miss nuanced contradictions.
- If Tavily's own sources contain incorrect information, the system may wrongly label a hallucinated answer as "Certain."
- If a hallucinated answer happens to phrase itself similarly to correct-sounding evidence, it may score high on similarity despite being wrong.
- Semantic similarity measures phrasing overlap, not factual accuracy — two sentences can be semantically similar while making different factual claims.
- Source authority is based on domain name only — a `.edu` page can still contain errors, and a Reddit post can still be factually correct.
- The confidence gap explanation identifies missing *words*, not missing *facts* — it is a lexical heuristic, not a semantic one.

</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
