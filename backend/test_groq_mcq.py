import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)

CONTEXT = """
SQL stands for Structured Query Language used to manage relational databases.
SELECT retrieves data. WHERE filters rows. PRIMARY KEY uniquely identifies records.
FOREIGN KEY links tables. JOIN combines rows from multiple tables.
INDEX speeds up queries. GROUP BY aggregates rows. ORDER BY sorts results.
"""

prompt = (
    "Generate exactly 3 MCQ questions from this context. "
    "Return ONLY a valid JSON array:\n"
    '[{"question":"...","options":{"A":"...","B":"...","C":"...","D":"..."},"correct_answer":"B","explanation":"...","hints":["hint"]}]\n\n'
    "CONTEXT:\n" + CONTEXT + "\n\nJSON array only:"
)

import json
from groq import Groq

key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
print("Key:", key[:20] if key else "MISSING")
print("Model:", model)

c = Groq(api_key=key)
r = c.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=4000,
    temperature=0.5
)
result = r.choices[0].message.content
print("Finish reason:", r.choices[0].finish_reason)
print("Result length:", len(result))
print("--- START ---")
print(result[:1000].encode("ascii", errors="replace").decode("ascii"))
print("--- END ---")

# Parse
clean = result.strip()
if "```" in clean:
    for part in clean.split("```"):
        stripped = part.lstrip("json").strip()
        if "[" in stripped:
            clean = stripped
            break

idx = clean.find("[")
end = clean.rfind("]")
print("idx:", idx, "end:", end)

if idx != -1 and end != -1 and end > idx:
    json_str = clean[idx:end+1]
    try:
        data = json.loads(json_str)
        print("PARSED OK:", len(data), "questions")
        if data:
            print("Q1:", data[0].get("question","?")[:100])
            print("Opts:", list(data[0].get("options", {}).keys()))
    except Exception as e:
        print("PARSE ERROR:", str(e))
        print("JSON start:", json_str[:200].encode("ascii", errors="replace").decode("ascii"))
else:
    print("No JSON array brackets found")
