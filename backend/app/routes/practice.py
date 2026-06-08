from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.database import get_db
from app.models import PracticeAttempt
from app.schemas import (
    PracticeRequest, PracticeResponse, PracticeProblem,
    PracticeSubmission, PracticeSubmissionResponse,
    PerformanceStats, MCQOption
)
from app.services.practice_generator import generate_practice_problems
from collections import defaultdict
import uuid

router = APIRouter()


@router.post("/practice/generate", response_model=PracticeResponse)
def generate_practice(req: PracticeRequest, db: Session = Depends(get_db)):
    """Generate practice problems based on uploaded content with specified difficulty"""
    try:
        problems = generate_practice_problems(
            topic=req.topic,
            subject=req.subject,
            difficulty=req.difficulty,
            count=req.count,
            question_type=req.question_type,
            document_id=req.document_id,
            document_ids=req.document_ids
        )
        
        # Check if no problems were generated
        if not problems:
            question_type_display = req.question_type.capitalize()
            raise HTTPException(
                status_code=400,
                detail=f"No relevant {question_type_display} questions found in the uploaded PDF. "
                       f"The PDF may not contain suitable content for {question_type_display.lower()} "
                       f"questions at {req.difficulty} difficulty level. "
                       f"Try uploading a different PDF or selecting a different question type."
            )
        
        return PracticeResponse(
            problems=problems,
            total_count=len(problems),
            difficulty=req.difficulty
        )
    except HTTPException:
        raise  # Re-raise HTTPException as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Practice generation failed: {str(e)}")


@router.post("/practice/submit", response_model=PracticeSubmissionResponse)
def submit_practice_answer(submission: PracticeSubmission, db: Session = Depends(get_db)):
    """Submit practice answer and get feedback with performance tracking"""
    try:
        # Determine correctness
        is_correct = False
        score = 0
        
        if submission.question_type == "mcq":
            is_correct = submission.user_answer.strip().upper() == submission.correct_answer.strip().upper()
            score = 100 if is_correct else 0
        elif submission.question_type in ["theory", "numerical"]:
            # For theory/numerical, use similarity or exact match
            user_ans = submission.user_answer.strip().lower()
            correct_ans = submission.correct_answer.strip().lower()
            
            # Simple similarity check (can be enhanced with embeddings)
            if user_ans == correct_ans:
                is_correct = True
                score = 100
            elif len(user_ans) > 0:
                # Partial credit based on keyword overlap
                user_words = set(user_ans.split())
                correct_words = set(correct_ans.split())
                overlap = len(user_words & correct_words)
                total = len(correct_words)
                score = int((overlap / max(total, 1)) * 100) if total > 0 else 0
                is_correct = score >= 70
        
        # Generate feedback
        if is_correct:
            feedback = "✅ Excellent! Your answer is correct."
        elif score >= 50:
            feedback = "⚠️ Partially correct. Review the solution for complete understanding."
        else:
            feedback = "❌ Incorrect. Study the solution carefully and try again."
        
        # Add difficulty-based feedback
        if submission.difficulty == "hard" and is_correct:
            feedback += " Great job on this challenging problem!"
        elif submission.difficulty == "easy" and not is_correct:
            feedback += " This is a fundamental concept - make sure to review the basics."
        
        # Save attempt to database
        attempt = PracticeAttempt(
            question=submission.question,
            question_type=submission.question_type,
            difficulty=submission.difficulty,
            subject=submission.subject,
            topic=submission.topic,
            user_answer=submission.user_answer,
            correct_answer=submission.correct_answer,
            is_correct=1 if is_correct else 0,
            time_taken=submission.time_taken,
            score=score
        )
        db.add(attempt)
        db.commit()
        
        return PracticeSubmissionResponse(
            is_correct=is_correct,
            correct_answer=submission.correct_answer,
            explanation=f"Score: {score}/100",
            score=score,
            feedback=feedback
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Submission failed: {str(e)}")


@router.get("/practice/stats", response_model=PerformanceStats)
def get_performance_stats(
    subject: str = None,
    topic: str = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get comprehensive performance statistics and analytics"""
    try:
        # Base query
        query = db.query(PracticeAttempt)
        if subject:
            query = query.filter(PracticeAttempt.subject == subject)
        if topic:
            query = query.filter(PracticeAttempt.topic == topic)
        
        attempts = query.all()
        
        if not attempts:
            return PerformanceStats(
                total_attempts=0,
                correct_attempts=0,
                accuracy=0.0,
                average_score=0.0,
                average_time=0.0,
                by_difficulty={},
                by_subject={},
                by_topic={},
                recent_attempts=[],
                weak_areas=[],
                strong_areas=[]
            )
        
        # Calculate overall stats
        total_attempts = len(attempts)
        correct_attempts = sum(1 for a in attempts if a.is_correct == 1)
        accuracy = (correct_attempts / total_attempts) * 100
        average_score = sum(a.score for a in attempts) / total_attempts
        average_time = sum(a.time_taken for a in attempts) / total_attempts
        
        # Stats by difficulty
        by_difficulty = defaultdict(lambda: {"total": 0, "correct": 0, "avg_score": 0})
        for a in attempts:
            diff = a.difficulty or "medium"
            by_difficulty[diff]["total"] += 1
            by_difficulty[diff]["correct"] += a.is_correct
            by_difficulty[diff]["avg_score"] += a.score
        
        for diff in by_difficulty:
            total = by_difficulty[diff]["total"]
            by_difficulty[diff]["avg_score"] = int(by_difficulty[diff]["avg_score"] / total)
            by_difficulty[diff]["accuracy"] = int((by_difficulty[diff]["correct"] / total) * 100)
        
        # Stats by subject
        by_subject = defaultdict(lambda: {"total": 0, "correct": 0, "avg_score": 0})
        for a in attempts:
            subj = a.subject or "General"
            by_subject[subj]["total"] += 1
            by_subject[subj]["correct"] += a.is_correct
            by_subject[subj]["avg_score"] += a.score
        
        for subj in by_subject:
            total = by_subject[subj]["total"]
            by_subject[subj]["avg_score"] = int(by_subject[subj]["avg_score"] / total)
            by_subject[subj]["accuracy"] = int((by_subject[subj]["correct"] / total) * 100)
        
        # Stats by topic
        by_topic = defaultdict(lambda: {"total": 0, "correct": 0, "avg_score": 0})
        for a in attempts:
            top = a.topic or "General"
            by_topic[top]["total"] += 1
            by_topic[top]["correct"] += a.is_correct
            by_topic[top]["avg_score"] += a.score
        
        for top in by_topic:
            total = by_topic[top]["total"]
            by_topic[top]["avg_score"] = int(by_topic[top]["avg_score"] / total)
            by_topic[top]["accuracy"] = int((by_topic[top]["correct"] / total) * 100)
        
        # Recent attempts
        recent = db.query(PracticeAttempt).order_by(desc(PracticeAttempt.attempted_at)).limit(limit).all()
        recent_attempts = [
            {
                "question": a.question[:100] + "..." if len(a.question) > 100 else a.question,
                "difficulty": a.difficulty,
                "subject": a.subject,
                "is_correct": bool(a.is_correct),
                "score": a.score,
                "time_taken": a.time_taken,
                "attempted_at": a.attempted_at.isoformat() if a.attempted_at else None
            }
            for a in recent
        ]
        
        # Identify weak areas (low accuracy topics)
        weak_areas = [
            {"topic": topic, "accuracy": stats["accuracy"], "total": stats["total"]}
            for topic, stats in by_topic.items()
            if stats["accuracy"] < 60 and stats["total"] >= 3
        ]
        weak_areas = sorted(weak_areas, key=lambda x: x["accuracy"])[:5]
        
        # Identify strong areas (high accuracy topics)
        strong_areas = [
            {"topic": topic, "accuracy": stats["accuracy"], "total": stats["total"]}
            for topic, stats in by_topic.items()
            if stats["accuracy"] >= 80 and stats["total"] >= 3
        ]
        strong_areas = sorted(strong_areas, key=lambda x: x["accuracy"], reverse=True)[:5]
        
        return PerformanceStats(
            total_attempts=total_attempts,
            correct_attempts=correct_attempts,
            accuracy=round(accuracy, 2),
            average_score=round(average_score, 2),
            average_time=round(average_time, 2),
            by_difficulty=dict(by_difficulty),
            by_subject=dict(by_subject),
            by_topic=dict(by_topic),
            recent_attempts=recent_attempts,
            weak_areas=weak_areas,
            strong_areas=strong_areas
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")
