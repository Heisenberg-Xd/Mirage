"""
Simulate exactly what FastAPI's async endpoint does:
call run_verification from a NEW THREAD (not MainThread),
which is what uvicorn's worker does.
This reproduces the 500 error deterministically.
"""
import os
import sys
import traceback
import threading
import json
from dataclasses import asdict

from dotenv import load_dotenv
load_dotenv()

from fastapi.encoders import jsonable_encoder
from verification.search import search_evidence_expanded
from verification import run_verification

result_holder = {}

def simulate_fastapi_thread():
    try:
        print(f"[thread] Thread name: {threading.current_thread().name}")

        print("[thread] Step 1: Generating answer via Groq...")
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Who is the PM of India?"}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        print(f"[thread] Answer: {answer[:80]}")

        print("[thread] Step 2: Searching evidence...")
        evidence = search_evidence_expanded("Who is the PM of India?")
        print(f"[thread] Evidence count: {len(evidence)}")

        print("[thread] Step 3: Running verification pipeline...")
        result = run_verification(
            answer=answer,
            evidence=evidence,
            question="Who is the PM of India?",
        )
        print(f"[thread] Verification label: {result.label}")

        print("[thread] Step 4: asdict(result)...")
        result_dict = asdict(result)
        print("[thread] asdict OK")

        print("[thread] Step 5: jsonable_encoder...")
        encoded = jsonable_encoder({
            "success": True,
            "question": "Who is the PM of India?",
            "answer": answer,
            "evidence_count": len(evidence),
            "result": result_dict
        })
        print("[thread] jsonable_encoder OK")

        print("[thread] Step 6: json.dumps...")
        json.dumps(encoded)
        print("[thread] json.dumps OK")

        result_holder["success"] = True

    except Exception as e:
        print(f"\n[thread] *** EXCEPTION CAUGHT ***")
        traceback.print_exc()
        result_holder["error"] = str(e)

# Run in a separate thread like uvicorn does
t = threading.Thread(target=simulate_fastapi_thread, name="uvicorn_worker_sim")
t.start()
t.join(timeout=300)

print("\n--- RESULT ---")
if result_holder.get("success"):
    print("SUCCESS: The full pipeline works from a worker thread.")
else:
    print(f"FAILED: {result_holder.get('error', 'unknown error')}")
