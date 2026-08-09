"""
backend/main.py
===============
FastAPI wrapper around the existing verification engine.
- Does NOT contain any verification logic.
- Delegates everything to verification.* package.
"""

import os
import math
import traceback
from dataclasses import asdict
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    print("[STARTUP] WARNING: GROQ_API_KEY is not set in .env")
if not TAVILY_API_KEY:
    print("[STARTUP] WARNING: TAVILY_API_KEY is not set in .env")

# ----------------------------
# Import verification engine
# (no logic lives here)
# ----------------------------
from verification import run_verification
from verification.search import search_evidence_expanded

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Mirage Verification API", version="1.0.0")

# ----------------------------
# CORS – dev only
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request schema
# ----------------------------
class VerifyRequest(BaseModel):
    question: str
    answer: Optional[str] = None


# ----------------------------
# JSON-safe serializer
# Converts every Python type that JSON cannot handle into a safe primitive.
# Called recursively so nested dataclasses / numpy scalars / enums all work.
# ----------------------------
def _to_json_safe(obj: Any) -> Any:
    """Recursively convert obj to a JSON-serializable structure."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        # Guard against nan / inf which JSON does not allow
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    # numpy scalars – convert to Python native
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return [_to_json_safe(x) for x in obj.tolist()]
    except ImportError:
        pass
    # dataclass → dict
    try:
        from dataclasses import asdict as _asdict, fields  # noqa: PLC0415
        fields(obj)  # raises TypeError if not a dataclass
        return _to_json_safe(_asdict(obj))
    except TypeError:
        pass
    # Enum
    try:
        return obj.value
    except AttributeError:
        pass
    # Last resort
    return str(obj)


# ----------------------------
# LLM answer generator
# (only called when frontend sends no answer)
# ----------------------------
def _get_llm_answer(question: str) -> str:
    from groq import Groq  # noqa: PLC0415

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    client = Groq(api_key=GROQ_API_KEY)
    print("[verify] Calling Groq LLM...")

    last_exc: Optional[Exception] = None
    for model in ("llama-3.3-70b-versatile", "llama-3.1-8b-instant"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.3,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            print(f"[verify] Model {model} failed: {exc}")
            last_exc = exc

    raise RuntimeError(f"All Groq models failed. Last error: {last_exc}") from last_exc


# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health():
    return {"status": "ok", "service": "Mirage Verification API"}


# ----------------------------
# POST /api/verify
# ----------------------------
@app.post("/api/verify")
async def verify_endpoint(req: VerifyRequest):
    print(f"\n[verify] question={req.question!r}")

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    # ── Step 1: Get or generate the answer ──────────────────────────────────
    answer = req.answer
    if not answer or not answer.strip():
        try:
            answer = _get_llm_answer(req.question)
            print(f"[verify] LLM answer ({len(answer)} chars): {answer[:80]}...")
        except Exception:
            traceback.print_exc()
            raise

    # ── Step 2: Evidence search ──────────────────────────────────────────────
    try:
        print("[verify] Searching evidence...")
        evidence = search_evidence_expanded(req.question)
        print(f"[verify] Evidence count: {len(evidence)}")
    except Exception:
        traceback.print_exc()
        raise

    # ── Step 3: Verification pipeline ───────────────────────────────────────
    try:
        print("[verify] Running verification pipeline...")
        result = run_verification(
            answer=answer,
            evidence=evidence,
            question=req.question,
        )
        print(f"[verify] label={result.label}, confidence_pct={result.confidence_pct}")
    except Exception:
        traceback.print_exc()
        raise

    # ── Step 4: Serialize ────────────────────────────────────────────────────
    # VerificationResult is a @dataclass, NOT a Pydantic BaseModel.
    # asdict() converts it to a plain dict; _to_json_safe() handles every
    # nested type (NLIScore, EvidenceScore, numpy scalars, etc.).
    try:
        print("[verify] Serializing result...")
        result_dict = _to_json_safe(asdict(result))
        payload = {
            "question": req.question,
            "raw_answer": answer,
            "result": result_dict,
        }
        print("[verify] Done – returning response")
        return payload
    except Exception:
        traceback.print_exc()
        raise
