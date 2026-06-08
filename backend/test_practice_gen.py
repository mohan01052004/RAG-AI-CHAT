import sys
sys.path.insert(0, '.')
from app.services.practice_generator import generate_practice_problems

print("Testing practice generation...\n")

for q_type in ['mcq', 'theory', 'numerical']:
    print(f"[{q_type.upper()}] Generating...")
    problems = generate_practice_problems(
        subject='Computer Science',
        difficulty='medium',
        count=2,
        question_type=q_type
    )
    print(f"[OK] Generated {len(problems)} {q_type} problems")
    if problems:
        for i, p in enumerate(problems[:1]):
            print(f"  Q{i+1}: {p.question[:70]}...")
            print(f"  Answer: {str(p.correct_answer)[:60]}...")
    print()
