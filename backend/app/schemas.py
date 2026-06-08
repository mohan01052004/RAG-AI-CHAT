from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str
    subject: Optional[str] = None
    document_id: Optional[int] = None
    document_ids: Optional[List[int]] = None
    # Advanced RAG options
    use_self_query: bool = True  # Enable Self-Query for metadata extraction
    use_hyde: bool = True  # Enable HyDE for hypothetical documents
    use_fusion: bool = True  # Enable RAG Fusion to combine strategies
    enable_advanced_rag: bool = False  # Override to use new advanced pipeline

class MCQOption(BaseModel):
    label: str
    text: str

class MCQQuestion(BaseModel):
    question: str
    options: List[MCQOption]
    correct_answer: str
    explanation: Optional[str] = None

class MCQResponse(BaseModel):
    questions: List[MCQQuestion]

class MCQAnswerRequest(BaseModel):
    question_index: int
    selected_answer: str
    correct_answer: str

class MCQAnswerResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str


class PracticeRequest(BaseModel):
    topic: Optional[str] = None
    subject: Optional[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    count: int = 5
    question_type: str = "mcq"  # mcq, theory, numerical
    document_id: Optional[int] = None
    document_ids: Optional[List[int]] = None


class PracticeProblem(BaseModel):
    id: str
    question: str
    question_type: str
    difficulty: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    options: Optional[List[MCQOption]] = None
    correct_answer: Optional[str] = None
    solution: Optional[str] = None
    hints: Optional[List[str]] = None


class PracticeResponse(BaseModel):
    problems: List[PracticeProblem]
    total_count: int
    difficulty: str


class PracticeSubmission(BaseModel):
    problem_id: str
    question: str
    question_type: str
    difficulty: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    user_answer: str
    correct_answer: str
    time_taken: int  # seconds


class PracticeSubmissionResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    score: int
    feedback: str


class PerformanceStats(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy: float
    average_score: float
    average_time: float
    by_difficulty: Dict[str, Dict[str, int]]
    by_subject: Dict[str, Dict[str, int]]
    by_topic: Dict[str, Dict[str, int]]
    recent_attempts: List[Dict]
    weak_areas: List[Dict]
    strong_areas: List[Dict]
