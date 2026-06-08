"""Debug script — run from backend/ with: python test_practice_debug.py"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(override=True)

print("=" * 60)
print("STEP 1: Config constants")
from app.config import GEMINI_API_KEY, GEMINI_MODEL, GROQ_API_KEY, GROQ_MODEL
print(f"  GEMINI_API_KEY: {'SET (' + GEMINI_API_KEY[:15] + '...)' if GEMINI_API_KEY else 'MISSING'}")
print(f"  GEMINI_MODEL:   {GEMINI_MODEL}")
print(f"  GROQ_API_KEY:   {'SET (' + GROQ_API_KEY[:15] + '...)' if GROQ_API_KEY else 'MISSING'}")
print(f"  GROQ_MODEL:     {GROQ_MODEL}")

print("\nSTEP 2: Test Groq directly")
try:
    from groq import Groq
    c = Groq(api_key=GROQ_API_KEY)
    r = c.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' only"}],
        max_tokens=10
    )
    print(f"  Groq response: {r.choices[0].message.content!r} ✅")
except Exception as e:
    print(f"  Groq FAILED: {e} ❌")

print("\nSTEP 3: Test _generate_with_gemini (includes Groq fallback)")
try:
    from app.services.rag_pipeline import _generate_with_gemini
    result = _generate_with_gemini("Return the word: HELLO", max_tokens=20)
    print(f"  Result: {result!r} {'✅' if result else '❌ NONE'}")
except Exception as e:
    print(f"  FAILED: {e} ❌")

print("\nSTEP 4: Test MCQ generation with fake context")
FAKE_CONTEXT = """
SQL (Structured Query Language) is used to manage relational databases.
A SELECT statement retrieves data from one or more tables.
The WHERE clause filters rows based on a condition.
JOIN combines rows from two or more tables based on a related column.
An INDEX speeds up data retrieval operations on database tables.
The GROUP BY clause groups rows sharing a property so aggregate functions can be applied.
PRIMARY KEY uniquely identifies each row in a table.
FOREIGN KEY establishes a link between two tables.
"""

try:
    from app.services.practice_generator import _classify_difficulty_prompt, _generate_mcq_problems
    config = _classify_difficulty_prompt("medium", "mcq")
    problems = _generate_mcq_problems(FAKE_CONTEXT, 3, "medium", config, "SQL", None)
    print(f"  Generated {len(problems)} problems")
    for i, p in enumerate(problems):
        print(f"\n  Q{i+1}: {p.question[:100]}")
        if p.options:
            print(f"    A: {p.options[0].text[:80]}")
            print(f"    B: {p.options[1].text[:80]}")
except Exception as e:
    import traceback
    print(f"  FAILED:")
    traceback.print_exc()

print("\n" + "=" * 60)
