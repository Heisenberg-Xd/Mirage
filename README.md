# Mirage — AI Hallucination Detector

A Q&A hallucination checker that verifies AI-generated answers against live web evidence and labels them **Not Hallucinating** / **Cannot Verify** / **Hallucinating**.

## Output States

| State | Meaning | Confidence |
|---|---|---|
| ✅ **Not Hallucinating** | All relevant claims are supported by evidence | 95–100% |
| ⚠️ **Cannot Verify** | Insufficient evidence to support or contradict | 40–60% |
| ❌ **Hallucinating** | One or more relevant claims contradicted by evidence | 10–30% |

> The confidence score represents confidence **in the hallucination assessment**, not in the answer quality.

## How It Works

1. You ask a question.
2. The LLM (Groq / Llama) generates a "cold" answer (no search context, no hints).
3. Tavily searches the web for evidence on the same question.
4. A CrossEncoder filters claims to only those that **directly answer the question** — conversational filler is ignored.
5. DeBERTa v3 NLI determines whether each evidence source **entails or contradicts** each relevant claim.
6. Hallucination decision:
   - Any contradicted relevant claim → **Hallucinating**
   - All claims supported → **Not Hallucinating**
   - No conclusive evidence either way → **Cannot Verify**

> **Key design principle**: The LLM is used exactly once — to produce the answer being checked. Every verification step is deterministic code, so the verification layer itself cannot hallucinate.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file from the template
cp .env.example .env
# Then edit .env and add your API keys:
#   GROQ_API_KEY=your_key_here
#   TAVILY_API_KEY=your_key_here

# 3. Run the app (IMPORTANT: use streamlit run, NOT python app.py)
streamlit run app.py
# or if streamlit isn't on PATH:
python -m streamlit run app.py
```

## API Keys Required

| Service | Get a key at |
|---------|-------------|
| Groq | https://console.groq.com/keys |
| Tavily Search | https://tavily.com |

## Test Cases

- **"Who invented Python?"** — should be Not Hallucinating (well-documented fact)
- **"What is the capital of Australia?"** — tests Sydney/Canberra hallucination
- **A recent/ambiguous event** — should show Cannot Verify
- **A deliberately false claim** — should show Hallucinating
