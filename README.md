# AI Hallucination Confidence Labeler

A Q&A reliability checker that verifies AI-generated answers against live web evidence and labels them **Certain** / **Uncertain** / **Needs Verification**.

## How It Works

1. You ask a question.
2. The LLM (Groq / Llama) generates a "cold" answer (no search context, no hints).
3. Tavily searches the web for evidence on the same question.
4. The system computes semantic similarity between the answer and evidence using sentence-transformers — **no LLM involved**.
5. A deterministic threshold function assigns a confidence label.
6. A template (not an LLM) generates the explanation text.

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

- **"Who invented Python?"** — should be Certain (well-documented fact)
- **"What is the capital of Australia?"** — tests Sydney/Canberra confusion
- **A recent/ambiguous event** — should show Uncertain or Needs Verification
- **A deliberately obscure question** — tests the "no evidence found" path
