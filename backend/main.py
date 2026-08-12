"""
backend/main.py
===============
FastAPI wrapper around the existing verification engine.
- Does NOT contain any verification logic.
- Delegates everything to verification.* package.
- Instrumented with step-by-step timing logs for production debugging.

Production hardening (v2):
- lifespan context manager (replaces deprecated @on_event)
- Models pre-loaded at startup (eliminates first-request cold start)
- /health/ready endpoint for Railway health probe
- Evidence search failure degrades gracefully (returns Cannot Verify, not 500)
- Traceback is logged server-side ONLY — clients receive a safe error message
- asyncio timeout wrapper on the full verify pipeline
- In-memory per-IP rate limiting (10 req/min)
- numpy import moved to module level
"""

import asyncio
import gc
import json
import math
import os
import sys
import time
import traceback
import logging
from contextlib import asynccontextmanager
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("mirage.api")

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
VERIFY_TIMEOUT = int(os.getenv("VERIFY_TIMEOUT_SECONDS", "120"))

logger.info("[ENV] GROQ_API_KEY    : %s", "Loaded (len=%d)" % len(GROQ_API_KEY)    if GROQ_API_KEY    else "MISSING")
logger.info("[ENV] TAVILY_API_KEY  : %s", "Loaded (len=%d)" % len(TAVILY_API_KEY)  if TAVILY_API_KEY  else "MISSING")
logger.info("[ENV] VERIFY_TIMEOUT  : %ds", VERIFY_TIMEOUT)

# ---------------------------------------------------------------------------
# Import verification engine
# ---------------------------------------------------------------------------
logger.info("[IMPORT] Loading verification engine...")
try:
    from verification import run_verification
    from verification.search import search_evidence_expanded
    logger.info("[IMPORT] verification engine imported OK")
except Exception as _import_err:
    logger.error("[IMPORT] FAILED to import verification engine!")
    traceback.print_exc()
    raise

# ---------------------------------------------------------------------------
# Readiness flag — set to True after startup models are loaded
# ---------------------------------------------------------------------------
_is_ready = False

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (10 requests/min per IP)
# ---------------------------------------------------------------------------
_rate_store: dict[str, list[float]] = defaultdict(list)
_RATE_LIMIT = 10      # max requests
_RATE_WINDOW = 60.0   # per N seconds

def _check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = time.time()
    window_start = now - _RATE_WINDOW
    timestamps = _rate_store[ip]
    # Prune old entries
    _rate_store[ip] = [t for t in timestamps if t > window_start]
    if len(_rate_store[ip]) >= _RATE_LIMIT:
        return False
    _rate_store[ip].append(now)
    return True

# ---------------------------------------------------------------------------
# Lifespan — pre-load models so the first request is fast
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm spaCy and NLI models at startup."""
    global _is_ready
    logger.info("\n[STARTUP] Warming up models...")

    # Pre-load spaCy (small, fast ~200ms)
    try:
        from verification.spacy_loader import get_spacy_model
        get_spacy_model()
        logger.info("[STARTUP] spaCy model loaded OK")
    except Exception as e:
        logger.warning("[STARTUP] spaCy preload failed (will lazy-load): %s", e)

    # Pre-load NLI model (heavy, ~2–30s depending on cache)
    try:
        from verification.nli import load_nli_model
        model = load_nli_model()
        if model is not None:
            logger.info("[STARTUP] NLI model loaded OK")
        else:
            logger.warning("[STARTUP] NLI model is disabled or failed to load")
    except Exception as e:
        logger.warning("[STARTUP] NLI preload failed (will lazy-load): %s", e)

    _is_ready = True
    logger.info("[STARTUP] Server ready.\n")

    yield  # <-- application runs here

    # Shutdown cleanup
    logger.info("[SHUTDOWN] Releasing model resources...")
    gc.collect()
    logger.info("[SHUTDOWN] Done.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mirage Verification API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

# ---------------------------------------------------------------------------
# Global exception handler — never leak tracebacks to clients
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("[UNCAUGHT] %s: %s\n%s", type(exc).__name__, exc, tb)
    # Return a SAFE generic error — no traceback, no internal paths
    return JSONResponse(
        status_code=500,
        content={
            "error": "An internal server error occurred.",
            "type": type(exc).__name__,
            # stage hint only — no traceback
        },
    )

# ---------------------------------------------------------------------------
# CORS — explicit origins only (no wildcard)
# ---------------------------------------------------------------------------
_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://mirage-one-psi.vercel.app",
]
_ALLOWED_ORIGIN_REGEX = r"https://.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_origin_regex=_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
    expose_headers=["X-Request-Time"],
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    question: str
    answer: Optional[str] = None

# ---------------------------------------------------------------------------
# JSON-safe serializer
# ---------------------------------------------------------------------------
def _to_json_safe(obj: Any) -> Any:
    """Recursively convert obj to a JSON-serializable structure."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_json_safe(i) for i in obj]

    # numpy scalars / arrays
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)

    # torch tensors
    type_str = type(obj).__module__ + "." + type(obj).__qualname__
    if "torch" in type_str:
        try:
            return obj.tolist()
        except Exception:
            return None

    # spaCy objects
    if "spacy" in type_str:
        return str(obj)

    # dataclass → dict
    try:
        from dataclasses import asdict as _asdict, fields
        fields(obj)          # raises TypeError if not a dataclass
        return _to_json_safe(_asdict(obj))
    except TypeError:
        pass

    # Enum
    try:
        return obj.value
    except AttributeError:
        pass

    return str(obj)


def _verify_serializable(payload: dict, stage: str) -> None:
    """Attempt json.dumps and raise with context if it fails."""
    try:
        json.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[{stage}] json.dumps failed: {exc}\n"
            f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}"
        ) from exc

# ---------------------------------------------------------------------------
# LLM answer generator
# ---------------------------------------------------------------------------
def _get_llm_answer(question: str) -> str:
    from groq import Groq

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")

    client = Groq(api_key=GROQ_API_KEY)
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
            logger.warning("[LLM] Model %s failed: %s: %s", model, type(exc).__name__, exc)
            last_exc = exc

    raise RuntimeError(f"All Groq models failed. Last: {last_exc}") from last_exc

# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    """Lean health check for Railway — returns 200 immediately."""
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    """
    Readiness probe — returns 503 until startup models are loaded.
    Railway can use this as the health check path.
    """
    if not _is_ready:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "message": "Models are still loading..."},
        )
    return {
        "ready": True,
        "service": "Mirage Verification API",
        "python": sys.version.split()[0],
        "env": {
            "GROQ_API_KEY":   "set" if GROQ_API_KEY   else "MISSING",
            "TAVILY_API_KEY": "set" if TAVILY_API_KEY else "MISSING",
        },
    }

# ---------------------------------------------------------------------------
# POST /api/verify   fully instrumented with timeout
# ---------------------------------------------------------------------------
@app.post("/api/verify")
async def verify_endpoint(req: VerifyRequest, request: Request):
    req_t0 = time.time()

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending another request.",
        )

    logger.info("\n%s", "=" * 60)
    logger.info("REQUEST START  ip=%s", client_ip)
    logger.info("Question: %r", req.question[:120])
    logger.info("%s", "=" * 60)

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    try:
        payload = await asyncio.wait_for(
            _run_pipeline(req, req_t0),
            timeout=VERIFY_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("[TIMEOUT] Request exceeded %ds timeout", VERIFY_TIMEOUT)
        raise HTTPException(
            status_code=504,
            detail=f"Verification timed out after {VERIFY_TIMEOUT} seconds. Please try again.",
        )

    total = time.time() - req_t0
    logger.info("[STEP 5] Returning Response")
    logger.info("%s", "=" * 60)
    logger.info("REQUEST END  (total %.2fs)", total)
    logger.info("%s\n", "=" * 60)
    return payload


async def _run_pipeline(req: VerifyRequest, req_t0: float) -> dict:
    """Inner pipeline — separated so asyncio.wait_for can cancel it cleanly."""
    loop = asyncio.get_running_loop()

    # ── STEP 1: LLM Answer ─────────────────────────────────────────────────
    answer = req.answer
    if not answer or not answer.strip():
        t = time.time()
        logger.info("[STEP 1] Generating LLM Answer ...")
        try:
            answer = await loop.run_in_executor(None, _get_llm_answer, req.question)
            logger.info("[STEP 1] SUCCESS  (%.2fs)  len=%d", time.time() - t, len(answer))
        except Exception as exc:
            logger.error("[STEP 1] FAILED  (%.2fs)  %s: %s", time.time() - t, type(exc).__name__, exc)
            traceback.print_exc()
            raise HTTPException(status_code=502, detail={
                "stage": "llm_generation",
                "type": type(exc).__name__,
                "msg": str(exc),
            })
    else:
        logger.info("[STEP 1] Answer provided by client (len=%d)  skipping LLM", len(answer))

    # ── STEP 2: Evidence Search (graceful degradation on failure) ──────────
    t = time.time()
    logger.info("[STEP 2] Searching Evidence ...")
    evidence = []
    try:
        evidence = await loop.run_in_executor(None, search_evidence_expanded, req.question)
        logger.info("[STEP 2] SUCCESS  (%.2fs)  n_evidence=%d", time.time() - t, len(evidence))
    except Exception as exc:
        logger.warning(
            "[STEP 2] Evidence search FAILED (%.2fs)  %s: %s — proceeding with empty evidence",
            time.time() - t, type(exc).__name__, exc,
        )
        # Do NOT raise — degraded verification (Cannot Verify) is better than a 500

    # ── STEP 3: Verification Pipeline ──────────────────────────────────────
    t = time.time()
    logger.info("[STEP 3] Running Verification Pipeline ...")
    try:
        result = await loop.run_in_executor(
            None, run_verification, answer, evidence, req.question
        )
        logger.info(
            "[STEP 3] SUCCESS  (%.2fs)  label=%r  pct=%d",
            time.time() - t, result.label, result.confidence_pct,
        )
    except Exception as exc:
        logger.error("[STEP 3] FAILED  (%.2fs)  %s: %s", time.time() - t, type(exc).__name__, exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "stage": "verification_pipeline",
            "type": type(exc).__name__,
            "msg": str(exc),
        })

    # ── STEP 4: Serialization ───────────────────────────────────────────────
    t = time.time()
    logger.info("[STEP 4] Serializing Result ...")
    try:
        result_dict = _to_json_safe(asdict(result))
        payload = {
            "question":   req.question,
            "raw_answer": answer,
            "result":     result_dict,
        }
        _verify_serializable(payload, "STEP 4")
        logger.info("[STEP 4] SUCCESS  (%.2fs)", time.time() - t)
    except Exception as exc:
        logger.error("[STEP 4] FAILED  (%.2fs)  %s: %s", time.time() - t, type(exc).__name__, exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "stage": "serialization",
            "type": type(exc).__name__,
            "msg": str(exc),
        })

    return payload

# ---------------------------------------------------------------------------
# Debug endpoints — each tests ONE subsystem in isolation
# ---------------------------------------------------------------------------

@app.get("/debug/env")
def debug_env():
    """Check environment variables and Python info."""
    import platform
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "not installed"
        cuda_available = False
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except ImportError:
        st_version = "not installed"
    try:
        import spacy
        spacy_version = spacy.__version__
        spacy_models = list(spacy.util.get_installed_models())
    except ImportError:
        spacy_version = "not installed"
        spacy_models = []

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "ready": _is_ready,
        "env": {
            "GROQ_API_KEY":       f"set (len={len(GROQ_API_KEY)})"   if GROQ_API_KEY   else "MISSING",
            "TAVILY_API_KEY":     f"set (len={len(TAVILY_API_KEY)})" if TAVILY_API_KEY else "MISSING",
            "HF_HOME":            os.getenv("HF_HOME", "not set"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE", "not set"),
            "VERIFY_TIMEOUT":     f"{VERIFY_TIMEOUT}s",
        },
        "torch_version": torch_version,
        "torch_cuda_available": cuda_available,
        "sentence_transformers_version": st_version,
        "spacy_version": spacy_version,
        "spacy_installed_models": spacy_models,
    }


@app.get("/debug/groq")
def debug_groq():
    """Test ONLY Groq LLM connectivity."""
    t = time.time()
    try:
        ans = _get_llm_answer("What is 2+2? Reply with only the digit.")
        return {"status": "ok", "answer": ans, "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/search")
def debug_search():
    """Test ONLY Tavily evidence search."""
    t = time.time()
    try:
        ev = search_evidence_expanded("Is the sky blue?")
        return {"status": "ok", "evidence_count": len(ev), "sample": ev[:1], "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/entity")
def debug_entity():
    """Test ONLY spaCy entity extraction."""
    t = time.time()
    try:
        from verification.entity_extractor import extract_entities
        entities = extract_entities("Elon Musk founded SpaceX in California.")
        return {"status": "ok", "entities": entities, "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/nli")
def debug_nli():
    """Test ONLY the NLI model (DeBERTa) — does it load and score correctly?"""
    t = time.time()
    try:
        from verification.nli import load_nli_model, score_nli
        model = load_nli_model()
        if model is None:
            return {"status": "disabled", "msg": "NLI model is disabled via DISABLE_NLI env var or previously failed to load.", "time_s": round(time.time() - t, 2)}
        scores = score_nli(
            "Paris is the capital of France.",
            ["Paris is located in France and serves as the capital city."],
            model,
        )
        return {
            "status": "ok",
            "entailment": scores[0].entailment,
            "contradiction": scores[0].contradiction,
            "neutral": scores[0].neutral,
            "time_s": round(time.time() - t, 2),
        }
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/verification")
def debug_verification():
    """Test ONLY the verification pipeline (no LLM, no Search)."""
    t = time.time()
    try:
        res = run_verification(
            question="Is Paris in France?",
            answer="Paris is the capital city of France.",
            evidence=[{"url": "https://example.com", "title": "Test", "content": "Paris is in France."}],
        )
        return {"status": "ok", "label": res.label, "confidence_pct": res.confidence_pct, "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/serialization")
def debug_serialization():
    """Test _to_json_safe on tricky objects: numpy, spaCy Doc."""
    t = time.time()
    try:
        data = {
            "np_int":    np.int32(42),
            "np_float":  np.float32(3.14),
            "np_array":  np.array([1, 2, 3]),
            "np_nan":    float("nan"),
            "a_set":     {1, 2, 3},
        }

        try:
            import spacy
            nlp = spacy.blank("en")
            doc = nlp("Hello world")
            data["spacy_doc"] = doc
        except ImportError:
            pass

        safe = _to_json_safe(data)
        json.dumps(safe)   # final proof
        return {"status": "ok", "serialized": safe, "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}
