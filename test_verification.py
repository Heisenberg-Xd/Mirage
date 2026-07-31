import unittest
import logging
import sys
from verification import run_verification

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

class TestVerificationEngine(unittest.TestCase):
    def setUp(self):
        # Mock evidence so tests remain 100% deterministic and don't require API calls.
        self.evidence_india_pm = [
            {"url": "https://en.wikipedia.org/wiki/Narendra_Modi", "content": "Narendra Damodardas Modi is an Indian politician who has served as the 14th prime minister of India since May 2014."},
            {"url": "https://www.bbc.com/news/world-asia-india", "content": "Narendra Modi became the Prime Minister of India in 2014 and was re-elected in 2019."}
        ]
        
        self.evidence_france = [
            {"url": "https://en.wikipedia.org/wiki/Paris", "content": "Paris is the capital and most populous city of France."},
            {"url": "https://www.britannica.com/place/Paris", "content": "Paris, city and capital of France, situated in the north-central part of the country."}
        ]
        
        self.evidence_python = [
            {"url": "https://en.wikipedia.org/wiki/Guido_van_Rossum", "content": "Guido van Rossum is a Dutch programmer best known as the creator of the Python programming language."},
            {"url": "https://docs.python.org/3/faq/general.html", "content": "Python was created in the early 1990s by Guido van Rossum at Stichting Mathematisch Centrum (CWI) in the Netherlands."}
        ]

    def test_1_india_pm_correct(self):
        question = "Who was PM of India in 2015?"
        answer = "The PM of India in 2015 was Narendra Modi."
        result = run_verification(answer, self.evidence_india_pm, question)
        self.assertEqual(result.label, "Not Hallucinating")

    def test_2_india_pm_incorrect(self):
        question = "Who was PM of India in 2015?"
        answer = "The PM of India in 2015 was Rahul Gandhi."
        result = run_verification(answer, self.evidence_india_pm, question)
        self.assertEqual(result.label, "Hallucinating")

    def test_3_france_capital_correct(self):
        question = "Capital of France?"
        answer = "The capital of France is Paris."
        result = run_verification(answer, self.evidence_france, question)
        self.assertEqual(result.label, "Not Hallucinating")

    def test_4_france_capital_incorrect(self):
        question = "Capital of France?"
        answer = "The capital of France is Berlin."
        result = run_verification(answer, self.evidence_france, question)
        self.assertEqual(result.label, "Hallucinating")

    def test_5_python_creator_correct(self):
        question = "Who invented Python?"
        answer = "Python was created by Guido van Rossum."
        result = run_verification(answer, self.evidence_python, question)
        self.assertEqual(result.label, "Not Hallucinating")


if __name__ == "__main__":
    unittest.main()
