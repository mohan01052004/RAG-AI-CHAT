from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from app.database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String)
    subject = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


class DocumentTopic(Base):
    __tablename__ = "document_topics"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer)
    subject = Column(String)
    filename = Column(String)
    section = Column(String)
    topic = Column(String)
    subtopic = Column(String)
    page = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class PracticeAttempt(Base):
    __tablename__ = "practice_attempts"

    id = Column(Integer, primary_key=True)
    question = Column(Text)
    question_type = Column(String)  # mcq, theory, numerical
    difficulty = Column(String)  # easy, medium, hard
    subject = Column(String)
    topic = Column(String)
    user_answer = Column(Text)
    correct_answer = Column(Text)
    is_correct = Column(Integer)  # 0 or 1 for boolean
    time_taken = Column(Integer)  # seconds
    score = Column(Integer)  # 0-100
    attempted_at = Column(DateTime, default=datetime.utcnow)
