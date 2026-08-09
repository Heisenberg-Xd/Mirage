import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any

# Import verification logic
from verification import run_verification
from verification.search import search_evidence_expanded
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

app = FastAPI(title="Mirage Verification API")

# Add CORS so React can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    question: str

class VerifyResponse(BaseModel):
    question: str
    raw_answer: str
    result: dict

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

@app.post("/api/verify")
def verify_question(req: VerifyRequest):
    if not GROQ_API_KEY or not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys in .env")

    # 1. Generate answer
    try:
        raw_answer = get_llm_answer(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

    # 2. Search evidence
    try:
        evidence = search_evidence_expanded(req.question)
    except Exception as e:
        evidence = []
        print(f"Tavily search failed: {e}. Proceeding without evidence.")

    # 3. Verification
    try:
        result_obj = run_verification(raw_answer, evidence, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verification engine error: {str(e)}")

    # Convert VerificationResult to dict (using pydantic model dump)
    # VerificationResult is a pydantic BaseModel in models.py
    if hasattr(result_obj, "model_dump"):
        result_dict = result_obj.model_dump()
    else:
        result_dict = result_obj.dict()

    return {
        "question": req.question,
        "raw_answer": raw_answer,
        "result": result_dict
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
