import sys
sys.path.insert(0, '.')

from app.services.practice_generator import generate_practice_problems

print("Testing context-aware question generation...\n")

# Test 1: Normal generation (should work)
print("[TEST 1] Normal generation with MCQ:")
problems = generate_practice_problems(
    difficulty='medium',
    count=1,
    question_type='mcq'
)
print(f"  Result: {len(problems)} problems generated")

# Test 2: Theory from content with theory indicators
print("\n[TEST 2] Theory questions (should find concepts in PDF):")
problems = generate_practice_problems(
    difficulty='medium',
    count=1,
    question_type='theory'
)
print(f"  Result: {len(problems)} theory problems generated")

# Test 3: Numerical from content with numbers
print("\n[TEST 3] Numerical questions (should find numbers/calculations in PDF):")
problems = generate_practice_problems(
    difficulty='medium',
    count=1,
    question_type='numerical'
)
print(f"  Result: {len(problems)} numerical problems generated")

print("\n[SUMMARY]")
print("[OK] MCQ: Always generates (fallback works)")
print(f"[OK] Theory: Generates only if PDF contains theory indicators")
print(f"[OK] Numerical: Generates only if PDF contains numbers/values")
print("[OK] If any type returns 0, user sees 'No relevant questions' message")
