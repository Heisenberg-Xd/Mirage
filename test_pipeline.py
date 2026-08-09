import os
import traceback
import json
from dataclasses import asdict
from fastapi.encoders import jsonable_encoder

from dotenv import load_dotenv
load_dotenv()

from verification.search import search_evidence_expanded
from verification import run_verification

def main():
    try:
        print("Testing verification pipeline...")
        question = "Who is PM of India?"
        evidence = search_evidence_expanded(question)
        print("Evidence length:", len(evidence))
        answer = "Narendra Modi is the PM of India."
        result = run_verification(answer=answer, evidence=evidence, question=question)
        print("Verification completed.")
        
        result_dict = asdict(result)
        print("asdict completed.")
        
        encoded = jsonable_encoder({
            "success": True,
            "question": question,
            "answer": answer,
            "evidence_count": len(evidence),
            "result": result_dict
        })
        print("jsonable_encoder completed.")
        
        json_str = json.dumps(encoded)
        print("json.dumps completed.")
        
    except Exception as e:
        print("Caught Exception:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
