"""
backend/main.py
===============
FastAPI wrapper around the existing verification engine.
- Does NOT contain any verification logic.
- Delegates everything to verification.* package.
- Instrumented with step-by-step timing logs for production debugging.
"""

import json
import math
import os
import sys
import time
import traceback
import logging
from dataclasses import asdict
from typing import Any, Optional

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

logger.info("[ENV] GROQ_API_KEY    : %s", "Loaded (len=%d)" % len(GROQ_API_KEY)    if GROQ_API_KEY    else "MISSING ")
logger.info("[ENV] TAVILY_API_KEY  : %s", "Loaded (len=%d)" % len(TAVILY_API_KEY)  if TAVILY_API_KEY  else "MISSING ")

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
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Mirage Verification API", version="1.0.0")


# ---------------------------------------------------------------------------
# Global exception handler  never return a naked 502
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error("[UNCAUGHT] %s: %s\n%s", type(exc).__name__, exc, tb)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type":  type(exc).__name__,
            "traceback": tb,
            "stage": "unknown",
        },
    )


# ---------------------------------------------------------------------------
# Startup  pre-load models so the first request is fast
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def preload_models():
    logger.info("\n[STARTUP]  Pre-loading ML models ...")
    t0 = time.time()

    # 1. spaCy
    try:
        import spacy
        logger.info("[STARTUP] Checking spaCy model 'en_core_web_sm' ...")
        if not spacy.util.is_package("en_core_web_sm"):
            logger.warning("[STARTUP] en_core_web_sm not found  running spacy download ...")
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
        # actually load it to warm the cache
        nlp = spacy.load("en_core_web_sm")
        logger.info("[STARTUP] spaCy en_core_web_sm loaded (vocab size=%d) ", len(nlp.vocab))
    except Exception as exc:
        logger.error("[STARTUP] spaCy FAILED: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        # Non-fatal  claim_extractor has a fallback

    # 2. CrossEncoder (relevance)
    try:
        logger.info("[STARTUP] Loading CrossEncoder ...")
        from verification.cross_encoder import load_cross_encoder
        ce = load_cross_encoder()
        logger.info("[STARTUP] CrossEncoder loaded: %s ", type(ce).__name__)
    except Exception as exc:
        logger.error("[STARTUP] CrossEncoder FAILED: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        # Non-fatal  inference will fall back to zeros

    # 3. NLI model (DeBERTa)
    try:
        logger.info("[STARTUP] Loading NLI model ...")
        from verification.nli import load_nli_model
        nli = load_nli_model()
        logger.info("[STARTUP] NLI model loaded: %s ", type(nli).__name__)
    except Exception as exc:
        logger.error("[STARTUP] NLI model FAILED: %s: %s", type(exc).__name__, exc)
        traceback.print_exc()
        # Non-fatal

    logger.info("[STARTUP]  Initialization complete in %.2fs\n", time.time() - t0)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://mirage-one-psi.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class VerifyRequest(BaseModel):
    question: str
    answer: Optional[str] = None


# ---------------------------------------------------------------------------
# JSON-safe serializer
# Converts every Python type that JSON cannot handle into a safe primitive.
# Called recursively so nested dataclasses / numpy scalars / enums all work.
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
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return [_to_json_safe(x) for x in obj.tolist()]
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

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

    # dataclass  dict
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

    # Last resort
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
# Health check
# ---------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "Mirage Verification API",
        "python": sys.version,
        "env": {
            "GROQ_API_KEY":   "set" if GROQ_API_KEY   else "MISSING",
            "TAVILY_API_KEY": "set" if TAVILY_API_KEY else "MISSING",
        },
    }


# ---------------------------------------------------------------------------
# POST /api/verify   fully instrumented
# ---------------------------------------------------------------------------
@app.post("/api/verify")
async def verify_endpoint(req: VerifyRequest):
    req_t0 = time.time()
    logger.info("\n%s", "=" * 60)
    logger.info("REQUEST START")
    logger.info("Question: %r", req.question[:120])
    logger.info("%s", "=" * 60)

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    #  STEP 1  Generate LLM answer 
    answer = req.answer
    if not answer or not answer.strip():
        t = time.time()
        logger.info("[STEP 1] Generating LLM Answer ...")
        try:
            answer = _get_llm_answer(req.question)
            logger.info("[STEP 1] SUCCESS  (%.2fs)  len=%d", time.time() - t, len(answer))
        except Exception as exc:
            logger.error("[STEP 1] FAILED  (%.2fs)", time.time() - t)
            logger.error("         Type   : %s", type(exc).__name__)
            logger.error("         Message: %s", exc)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail={
                "stage": "llm_generation",
                "type": type(exc).__name__,
                "msg": str(exc),
            })
    else:
        logger.info("[STEP 1] Answer provided by client (len=%d)  skipping LLM", len(answer))

    #  STEP 2  Evidence search 
    t = time.time()
    logger.info("[STEP 2] Searching Evidence ...")
    try:
        evidence = search_evidence_expanded(req.question)
        logger.info("[STEP 2] SUCCESS  (%.2fs)  n_evidence=%d", time.time() - t, len(evidence))
    except Exception as exc:
        logger.error("[STEP 2] FAILED  (%.2fs)", time.time() - t)
        logger.error("         Type   : %s", type(exc).__name__)
        logger.error("         Message: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "stage": "evidence_search",
            "type": type(exc).__name__,
            "msg": str(exc),
        })

    #  STEP 3  Verification pipeline 
    t = time.time()
    logger.info("[STEP 3] Running Verification Pipeline ...")
    try:
        result = run_verification(
            answer=answer,
            evidence=evidence,
            question=req.question,
        )
        logger.info("[STEP 3] SUCCESS  (%.2fs)  label=%r  pct=%d", time.time() - t, result.label, result.confidence_pct)
    except Exception as exc:
        logger.error("[STEP 3] FAILED  (%.2fs)", time.time() - t)
        logger.error("         Type   : %s", type(exc).__name__)
        logger.error("         Message: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "stage": "verification_pipeline",
            "type": type(exc).__name__,
            "msg": str(exc),
        })

    #  STEP 4  Serialization 
    t = time.time()
    logger.info("[STEP 4] Serializing Result ...")
    try:
        result_dict = _to_json_safe(asdict(result))
        payload = {
            "question":   req.question,
            "raw_answer": answer,
            "result":     result_dict,
        }
        # Verify it's truly JSON-safe before returning
        _verify_serializable(payload, "STEP 4")
        logger.info("[STEP 4] SUCCESS  (%.2fs)", time.time() - t)
    except Exception as exc:
        logger.error("[STEP 4] FAILED  (%.2fs)", time.time() - t)
        logger.error("         Type   : %s", type(exc).__name__)
        logger.error("         Message: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "stage": "serialization",
            "type": type(exc).__name__,
            "msg": str(exc),
        })

    total = time.time() - req_t0
    logger.info("[STEP 5] Returning Response")
    logger.info("%s", "=" * 60)
    logger.info("REQUEST END  (total %.2fs)", total)
    logger.info("%s\n", "=" * 60)
    return payload


# ---------------------------------------------------------------------------
# Debug endpoints  each tests ONE subsystem in isolation
# ---------------------------------------------------------------------------

@app.get("/debug/env")
def debug_env():
    """Check environment variables and Python info."""
    import platform
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"
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
        "env": {
            "GROQ_API_KEY":   f"set (len={len(GROQ_API_KEY)})"   if GROQ_API_KEY   else "MISSING",
            "TAVILY_API_KEY": f"set (len={len(TAVILY_API_KEY)})" if TAVILY_API_KEY else "MISSING",
            "HF_HOME":        os.getenv("HF_HOME", "not set"),
            "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE", "not set"),
        },
        "torch_version": torch_version,
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
        traceback.print_exc()
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/search")
def debug_search():
    """Test ONLY Tavily evidence search."""
    t = time.time()
    try:
        ev = search_evidence_expanded("Is the sky blue?")
        return {"status": "ok", "evidence_count": len(ev), "sample": ev[:1], "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}


@app.get("/debug/serialization")
def debug_serialization():
    """Test _to_json_safe on tricky objects: numpy, spaCy Doc."""
    t = time.time()
    try:
        import numpy as np
        import spacy
        nlp = spacy.blank("en")
        doc = nlp("Hello world")
        data = {
            "np_int":    np.int32(42),
            "np_float":  np.float32(3.14),
            "np_array":  np.array([1, 2, 3]),
            "np_nan":    float("nan"),
            "spacy_doc": doc,
            "a_set":     {1, 2, 3},
        }
        safe = _to_json_safe(data)
        json.dumps(safe)   # final proof
        return {"status": "ok", "serialized": safe, "time_s": round(time.time() - t, 2)}
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "type": type(exc).__name__, "msg": str(exc), "time_s": round(time.time() - t, 2)}
