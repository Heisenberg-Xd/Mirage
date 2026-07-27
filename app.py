# IMPORTANT: Launch this app with `streamlit run app.py` or
# `python -m streamlit run app.py` — NOT `python app.py`.
# Running it directly causes ScriptRunContext warnings and broken session state.

"""
Mirage — AI Hallucination Confidence Labeler
=============================================
A Q&A reliability checker that verifies AI-generated answers against live web
evidence and labels them Certain / Likely Certain / Uncertain / Needs Verification.
"""

import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & API keys
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---------------------------------------------------------------------------
# External API clients
# ---------------------------------------------------------------------------
from groq import Groq
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Hybrid verification engine
# ---------------------------------------------------------------------------
from verification import run_verification
from verification.authority import get_authority_label, get_authority_tier
from verification.templates import generate_claim_summary
from verification.models import VerificationResult
from verification.search import search_evidence_expanded


# ---------------------------------------------------------------------------
# STEP 2 — Groq call (THE ONLY LLM CALL IN THE ENTIRE PIPELINE)
# ---------------------------------------------------------------------------
def get_llm_answer(question: str) -> str:
    client = Groq(api_key=GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        )
    except Exception:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
        )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Label color mapping (SaaS Palette)
# ---------------------------------------------------------------------------
LABEL_COLORS: dict[str, tuple[str, str]] = {
    "Certain":            ("#22C55E", "rgba(34, 197, 94, 0.15)"),
    "Likely Certain":     ("#4F8CFF", "rgba(79, 140, 255, 0.15)"),
    "Uncertain":          ("#F59E0B", "rgba(245, 158, 11, 0.15)"),
    "Needs Verification": ("#EF4444", "rgba(239, 68, 68, 0.15)"),
}

AUTHORITY_DISPLAY: dict[str, tuple[str, str, str]] = {
    "high":   ("🟢 High Authority",        "#22C55E", "rgba(34, 197, 94, 0.15)"),
    "medium": ("🟡 Medium Authority",       "#F59E0B", "rgba(245, 158, 11, 0.15)"),
    "low":    ("🔴 Low Authority (Caution)","#EF4444", "rgba(239, 68, 68, 0.15)"),
}

VERDICT_COLORS: dict[str, str] = {
    "supported":    "#22C55E",
    "insufficient": "#F59E0B",
    "contradicted": "#EF4444",
    "ignored":      "#64748B",
}

VERDICT_ICONS: dict[str, str] = {
    "supported":    "✅",
    "insufficient": "⚠️",
    "contradicted": "❌",
    "ignored":      "🚫",
}


# ===========================================================================
# UI Components & CSS (unchanged from previous SaaS redesign)
# ===========================================================================

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global Typography & Backgrounds */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0B0F17;
        color: #F8FAFC;
    }

    .stAppHeader {
        display: none;
    }

    /* Container Spacing */
    .block-container {
        max-width: 1200px !important;
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        color: #F8FAFC !important;
        letter-spacing: -0.02em;
    }

    p {
        color: #94A3B8;
        font-weight: 400;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] p {
        color: #94A3B8;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.2rem !important;
        margin-bottom: 2rem !important;
    }

    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #171F2F !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4F8CFF !important;
        box-shadow: 0 0 0 3px rgba(79, 140, 255, 0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4F8CFF !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #3b76e8 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(79, 140, 255, 0.25) !important;
    }

    /* Cards */
    .saas-card {
        background-color: #171F2F;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    .saas-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: rgba(255, 255, 255, 0.15);
    }

    .card-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .card-body {
        font-size: 1rem;
        color: #94A3B8;
        line-height: 1.6;
    }

    /* KPI Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        background-color: #171F2F;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem;
        height: 100%;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .kpi-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Claim verdict row */
    .claim-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.85rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .claim-row:last-child { border-bottom: none; }
    .claim-verdict-icon { font-size: 1.1rem; margin-top: 0.1rem; flex-shrink: 0; }
    .claim-text { color: #F8FAFC; font-size: 0.95rem; line-height: 1.5; }
    .claim-meta { color: #64748B; font-size: 0.8rem; margin-top: 0.2rem; }

    /* Score bar */
    .score-bar-wrap { margin: 0.5rem 0 0.25rem 0; }
    .score-bar-bg {
        width: 100%; height: 6px; background: rgba(255,255,255,0.08);
        border-radius: 99px; overflow: hidden;
    }
    .score-bar-fill { height: 6px; border-radius: 99px; transition: width 0.4s ease; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94A3B8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #4F8CFF !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        background-color: rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #4F8CFF;
    }

    /* Custom Badges */
    .saas-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .source-link a {
        color: #4F8CFF;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    .source-link a:hover {
        color: #3b76e8;
        text-decoration: underline;
    }

    /* Component breakdown bar */
    .component-row {
        display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;
    }
    .component-name {
        width: 130px; font-size: 0.82rem; color: #94A3B8; flex-shrink: 0;
    }
    .component-bar-wrap { flex: 1; }
    .component-pct {
        width: 42px; text-align: right; font-size: 0.82rem;
        color: #F8FAFC; font-weight: 600; flex-shrink: 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_kpi_card(title: str, value: str, color_hex: str = "#F8FAFC"):
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">{title}</div>
        <div class="kpi-value" style="color: {color_hex}">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_saas_card(title: str, content: str, icon: str = ""):
    st.markdown(f"""
    <div class="saas-card">
        <div class="card-header">{icon} {title}</div>
        <div class="card-body" style="color: #F8FAFC;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def render_score_bar(score_pct: int, color: str = "#4F8CFF"):
    st.markdown(f"""
    <div class="score-bar-wrap">
        <div class="score-bar-bg">
            <div class="score-bar-fill"
                 style="width:{score_pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_component_bar(name: str, value: float, color: str = "#4F8CFF"):
    pct = round(value * 100)
    st.markdown(f"""
    <div class="component-row">
        <div class="component-name">{name}</div>
        <div class="component-bar-wrap">
            <div class="score-bar-bg">
                <div class="score-bar-fill"
                     style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        <div class="component-pct">{pct}%</div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# Streamlit UI — main()
# ===========================================================================
def main():
    st.set_page_config(
        page_title="Mirage — AI Confidence Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h1>⚡ Mirage Labeler</h1>", unsafe_allow_html=True)
        st.markdown(
            "Hybrid deterministic fact-checking engine. "
            "One LLM call. Zero LLM verification."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Engine")
        st.markdown(
            '<div class="card-body" style="font-size:0.85rem; line-height:1.8;">'
            '🧠 <b style="color:#F8FAFC">LLM</b>: Groq / Llama-3.3-70b<br>'
            '🔍 <b style="color:#F8FAFC">Search</b>: Tavily (Top-5 Expanded)<br>'
            '⚙️ <b style="color:#F8FAFC">Filter</b>: CE Relevance Ranking<br>'
            '🧠 <b style="color:#F8FAFC">Voting</b>: DeBERTa v3 NLI<br>'
            '📏 <b style="color:#F8FAFC">Alignment</b>: RapidFuzz Entities<br>'
            '📊 <b style="color:#F8FAFC">Score</b>: 8-Component Composite'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Profile")
        st.markdown(
            '<div style="display:flex; align-items:center; gap:10px; margin-top:10px;">'
            '<div style="width:32px; height:32px; border-radius:50%; background:#4F8CFF;'
            ' display:flex; align-items:center; justify-content:center;'
            ' color:white; font-weight:bold;">A</div>'
            '<div><div style="font-weight:600; color:#F8FAFC; font-size:0.9rem;">Admin User</div>'
            '<div style="font-size:0.8rem; color:#64748B;">Workspace</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # --- Header ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            '<h1 style="margin-bottom:0.5rem; font-size: 2.2rem;">Verification Dashboard</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size: 1.1rem; margin-bottom: 2rem;">'
            'NLI-powered deterministic fact-checking — with entity drift & hallucination detection.'
            '</p>',
            unsafe_allow_html=True,
        )
    with col2:
        pass

    # --- Input Section ---
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    question = st.text_input(
        "Query",
        placeholder="Enter a claim or question to verify...",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        check_btn = st.button("Run Verification", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Main Pipeline
    # -----------------------------------------------------------------------
    if check_btn and question.strip():
        if not GROQ_API_KEY or not TAVILY_API_KEY:
            st.error(
                "Missing API keys. "
                "Please set `GROQ_API_KEY` and `TAVILY_API_KEY` in `.env`."
            )
            return

        with st.status("Analyzing claim...", expanded=True) as status:
            st.write("Generating answer with Llama model...")
            try:
                raw_answer = get_llm_answer(question)
            except Exception as e:
                status.update(label=f"Groq API Error: {e}", state="error")
                return

            st.write("Expanding search queries and retrieving top-5 evidence...")
            try:
                evidence = search_evidence_expanded(question)
            except Exception as e:
                st.warning(f"Tavily search failed: {e}. Proceeding without evidence.")
                evidence = []

            st.write("Extracting atomic claims and analyzing entities...")
            st.write("Running CrossEncoder for relevance filtering...")
            st.write("Running DeBERTa v3 for Natural Language Inference (NLI)...")
            st.write("Aggregating 8-component confidence score...")

            try:
                result: VerificationResult = run_verification(raw_answer, evidence, question)
            except Exception as e:
                status.update(label=f"Verification engine error: {e}", state="error")
                st.exception(e)
                return

            status.update(label="Analysis complete", state="complete", expanded=False)

        # -------------------------------------------------------------------
        # DISPLAY DASHBOARD RESULTS
        # -------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)

        # KPI Row — 5 cards
        fg, bg = LABEL_COLORS.get(result.label, ("#94A3B8", "transparent"))
        top_url = result.evidence[0]["url"] if result.evidence else ""
        top_auth_tier = get_authority_tier(
            result.authority_scores[0] if result.authority_scores else 0.5
        )
        top_auth_display = AUTHORITY_DISPLAY.get(top_auth_tier, ("Unknown", "#94A3B8", ""))

        # Only count relevant claims in the total for the KPI
        relevant_claims_count = sum(1 for c in result.claims if getattr(c, 'is_relevant_to_question', True))

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            render_kpi_card("Reliability", result.label, color_hex=fg)
        with kpi2:
            render_kpi_card("Confidence", f"{result.confidence_pct}%", color_hex=fg)
        with kpi3:
            render_kpi_card(
                "Claims",
                f"{result.supported_count}/{relevant_claims_count} verified",
                color_hex="#F8FAFC",
            )
        with kpi4:
            render_kpi_card(
                "Contradictions",
                str(result.contradicted_count) if result.contradicted_count else "None",
                color_hex="#EF4444" if result.contradicted_count else "#22C55E",
            )
        with kpi5:
            render_kpi_card(
                "Top Source",
                top_auth_display[0].split(" ", 1)[1] if len(top_auth_display[0].split(" ", 1)) > 1 else top_auth_display[0],
                color_hex=top_auth_display[1],
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Generated answer card
        render_saas_card("Generated Answer", raw_answer, icon="🤖")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", "Claims Analysis", "Evidence Sources", "How it Works"
        ])

        # ---- Tab 1: Overview ----
        with tab1:
            if result.entity_drift_detected:
                st.markdown("""
                <div class="saas-card" style="border-left: 4px solid #F59E0B; margin-top: 1rem; margin-bottom: 0;">
                    <div class="card-header" style="color:#F59E0B;">⚠️ Entity Drift Detected</div>
                    <div class="card-body">The answer introduced major entities not found in the original question.</div>
                </div>
                """, unsafe_allow_html=True)

            if result.has_hallucinated_claims:
                st.markdown("""
                <div class="saas-card" style="border-left: 4px solid #F59E0B; margin-top: 1rem; margin-bottom: 0;">
                    <div class="card-header" style="color:#F59E0B;">⚠️ Hallucinated Expansion Detected</div>
                    <div class="card-body">The answer contains factual claims that are irrelevant to the user's question.</div>
                </div>
                """, unsafe_allow_html=True)

            # Label explanation card
            st.markdown(f"""
            <div class="saas-card" style="margin-top: 1rem;">
                <div class="card-header">ℹ️ Verification Summary</div>
                <div class="card-body" style="color:#F8FAFC;">{result.explanation}</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence component breakdown
            st.markdown("""
            <div class="saas-card" style="border-left: 4px solid #4F8CFF;">
                <div class="card-header" style="color:#4F8CFF;">📊 8-Component Confidence Breakdown</div>
                <div class="card-body">
            """, unsafe_allow_html=True)

            components = {
                "NLI Entailment": result.nli_score,
                "CE Relevance": result.ce_score,
                "Entity Alignment": result.entity_score,
                "Q-Relevance": result.q_relevance_score,
                "Source Authority": result.authority_avg,
                "Support Ratio": result.support_ratio,
                "Src Diversity": result.diversity_score,
                "No Contradictions": result.contradiction_penalty,
            }
            component_colors = {
                "NLI Entailment": "#4F8CFF",
                "CE Relevance": "#22C55E",
                "Entity Alignment": "#A78BFA",
                "Q-Relevance": "#F472B6",
                "Source Authority": "#F59E0B",
                "Support Ratio": "#38BDF8",
                "Src Diversity": "#34D399",
                "No Contradictions": "#FB7185"
            }
            for name, val in components.items():
                render_component_bar(name, val, color=component_colors[name])

            st.markdown("</div></div>", unsafe_allow_html=True)

        # ---- Tab 2: Claims Analysis ----
        with tab2:
            claim_rows = generate_claim_summary(result.claim_verifications)

            if not claim_rows:
                st.info("No atomic claims were extracted from the answer.")
            else:
                st.markdown(f"""
                <div class="saas-card" style="margin-top: 1rem;">
                    <div class="card-header">🔬 Atomic Claims Extracted: {len(claim_rows)}</div>
                    <div class="card-body">
                """, unsafe_allow_html=True)

                for row in claim_rows:
                    icon    = VERDICT_ICONS.get(row["verdict"], "❓")
                    color   = VERDICT_COLORS.get(row["verdict"], "#94A3B8")
                    neg_tag = ' <span style="color:#A78BFA; font-size:0.75rem;">[negated]</span>' if row["is_negated"] else ""
                    rel_tag = ' <span style="color:#EF4444; font-size:0.75rem;">[hallucinated]</span>' if not row["is_relevant"] else ""
                    
                    if not row["is_relevant"]:
                        meta_text = f"Claim ignored for factual verification — irrelevant to question"
                    else:
                        meta_text = (f"NLI Entailment: <b style='color:#F8FAFC'>{row['best_nli_pct']}%</b> &nbsp;·&nbsp; "
                                     f"CE Relevance: <b style='color:#F8FAFC'>{row['best_score_pct']}%</b> &nbsp;·&nbsp; "
                                     f"Supporting: <b style='color:#22C55E'>{row['supporting']}</b> &nbsp;·&nbsp; "
                                     f"Verdict: <b style='color:{color}'>{row['verdict'].capitalize()}</b>")

                    st.markdown(f"""
                    <div class="claim-row">
                        <div class="claim-verdict-icon">{icon}</div>
                        <div>
                            <div class="claim-text">{row["claim"]}{neg_tag}{rel_tag}</div>
                            <div class="claim-meta">
                                {meta_text}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div></div>", unsafe_allow_html=True)

        # ---- Tab 3: Evidence Sources ----
        with tab3:
            if result.evidence:
                for i, src in enumerate(result.evidence):
                    auth_score = result.authority_scores[i] if i < len(result.authority_scores) else 0.5
                    auth_tier  = get_authority_tier(auth_score)
                    auth_label_str = get_authority_label(src.get("url", ""))
                    auth_txt, a_fg, a_bg = AUTHORITY_DISPLAY.get(
                        auth_tier, ("Unknown", "#94A3B8", "transparent")
                    )

                    # Find best claim score against this source
                    best_nli_for_source = 0.0
                    for cv in result.claim_verifications:
                        for es in cv.evidence_scores:
                            if es.source_idx == i:
                                best_nli_for_source = max(best_nli_for_source, es.nli_score.entailment if es.nli_score else 0.0)

                    st.markdown(f"""
                    <div class="saas-card" style="margin-top: 1rem;">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 1rem;">
                            <div style="font-weight:600; color:#F8FAFC; font-size:1.05rem;">
                                {src.get("title", f"Source {i+1}")}
                            </div>
                            <div style="display:flex; gap: 0.5rem; align-items:center; flex-shrink:0; margin-left:1rem;">
                                <span class="saas-badge" style="background:rgba(255,255,255,0.05); color:#F8FAFC;">
                                    NLI: {round(best_nli_for_source*100)}%
                                </span>
                                <span class="saas-badge" style="background:{a_bg}; color:{a_fg};">
                                    {auth_txt}
                                </span>
                                <span class="saas-badge" style="background:rgba(255,255,255,0.05); color:#94A3B8;">
                                    Auth: {round(auth_score*100)}
                                </span>
                            </div>
                        </div>
                        <div class="card-body" style="margin-bottom: 1rem;">
                            {src.get("content", "")[:350]}{"…" if len(src.get("content","")) > 350 else ""}
                        </div>
                        <div class="source-link">
                            <a href="{src['url']}" target="_blank">View Original Source ↗</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No web evidence found.")

        # ---- Tab 4: How it Works ----
        with tab4:
            st.markdown("""
            <div class="saas-card" style="margin-top: 1rem;">
                <div class="card-body">
                    <p style="color:#F8FAFC; font-weight:600; font-size:1.05rem; margin-bottom:1rem;">
                        One LLM call. Zero LLM verification. Fully deterministic.
                    </p>
                    <p>The LLM (Groq / Llama) is invoked exactly once to generate a cold answer.
                    All downstream steps are deterministic algorithms:</p>
                    <ol style="color:#94A3B8; margin-left:1.5rem; margin-top:0.75rem; line-height:1.9;">
                        <li><strong style="color:#F8FAFC">Entity Alignment</strong> —
                            RapidFuzz ensures entities in the answer match those in the question to detect drift.</li>
                        <li><strong style="color:#F8FAFC">Question Relevance Filter</strong> —
                            A CrossEncoder filters out hallucinated extra information that doesn't answer the prompt.</li>
                        <li><strong style="color:#F8FAFC">Query Expansion</strong> —
                            Questions are deterministically expanded to maximize search surface.</li>
                        <li><strong style="color:#F8FAFC">Natural Language Inference (NLI)</strong> —
                            DeBERTa v3 evaluates whether each evidence source entails or contradicts the claims.</li>
                        <li><strong style="color:#F8FAFC">Negation Handling</strong> —
                            Negated claims are extracted via spaCy dependency arcs. Evidence confirming a positive entity naturally supports the negated claim.</li>
                        <li><strong style="color:#F8FAFC">Weighted composite confidence</strong> —
                            Eight signals are combined (NLI, CE, Entity Drift, Authority, etc.).</li>
                    </ol>
                    <p style="margin-top:1rem; color:#64748B; font-size:0.9rem; font-style:italic;">
                        Because the verification layer generates nothing — it only measures —
                        it cannot hallucinate. Every confidence score is reproducible.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
