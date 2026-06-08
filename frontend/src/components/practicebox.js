import { useState, useEffect } from "react";
import { generatePractice, submitPractice, getDocuments } from "../api";
import "./practicebox.css";

export default function PracticeBox({ subject, documentIds }) {
  const [difficulty, setDifficulty] = useState("medium");
  const [questionType, setQuestionType] = useState("mcq");
  const [count, setCount] = useState(5);
  const [problems, setProblems] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [userAnswer, setUserAnswer] = useState("");
  const [showSolution, setShowSolution] = useState(false);
  const [showHints, setShowHints] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [startTime, setStartTime] = useState(null);
  const [score, setScore] = useState(null);

  // Fetch all docs from backend so quiz always has context
  const [allDocs, setAllDocs] = useState([]);
  const [selectedIds, setSelectedIds] = useState([]); // which docs to quiz from

  useEffect(() => {
    getDocuments()
      .then(res => {
        const docs = res.data?.documents || [];
        setAllDocs(docs);
        // Default: use all available doc IDs
        setSelectedIds(docs.map(d => d.id));
      })
      .catch(() => setAllDocs([]));
  }, []);

  const currentProblem = problems[currentIndex];

  const handleGenerate = async () => {
    if (selectedIds.length === 0) {
      alert("No documents selected. Please upload a PDF first.");
      return;
    }
    setIsGenerating(true);
    setProblems([]);
    setCurrentIndex(0);
    setFeedback(null);
    setShowSolution(false);
    setShowHints(false);
    setUserAnswer("");
    setScore(null);

    try {
      const res = await generatePractice({
        subject,
        difficulty,
        count: parseInt(count),
        question_type: questionType,
        document_ids: selectedIds   // always send the selected doc IDs
      });
      setProblems(res.data.problems || []);
      setStartTime(Date.now());
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "Failed to generate practice problems. Try again.";
      alert(errorMsg);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSubmit = async () => {
    if (!userAnswer.trim()) {
      alert("Please provide an answer");
      return;
    }

    setIsSubmitting(true);
    const timeTaken = Math.floor((Date.now() - startTime) / 1000);

    try {
      const res = await submitPractice({
        problem_id: currentProblem.id,
        question: currentProblem.question,
        question_type: currentProblem.question_type,
        difficulty: currentProblem.difficulty,
        subject: currentProblem.subject,
        topic: currentProblem.topic,
        user_answer: userAnswer,
        correct_answer: currentProblem.correct_answer,
        time_taken: timeTaken
      });
      
      setFeedback(res.data);
      setScore(res.data.score);
      setShowSolution(true);
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "Submission failed. Try again.";
      alert(errorMsg);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleNext = () => {
    if (currentIndex < problems.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setUserAnswer("");
      setShowSolution(false);
      setShowHints(false);
      setFeedback(null);
      setScore(null);
      setStartTime(Date.now());
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      setUserAnswer("");
      setShowSolution(false);
      setShowHints(false);
      setFeedback(null);
      setScore(null);
      setStartTime(Date.now());
    }
  };

  const handleReset = () => {
    setUserAnswer("");
    setShowSolution(false);
    setShowHints(false);
    setFeedback(null);
    setScore(null);
    setStartTime(Date.now());
  };

  return (
    <div className="practicebox-container">
      <div className="practicebox-header">
        <h2>🎯 Self-Assessment Quiz</h2>
        <p>Generate custom quizzes based on your uploaded document content</p>
      </div>

      {/* Document selector */}
      {allDocs.length > 0 && (
        <div style={{
          marginBottom: "16px",
          padding: "12px 16px",
          backgroundColor: "rgba(79,70,229,0.08)",
          borderRadius: "8px",
          border: "1px solid rgba(79,70,229,0.2)",
          fontSize: "13px",
          color: "#c7d2fe"
        }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontWeight: "600" }}>📂 Documents to quiz from:</span>
            <button
              onClick={() =>
                selectedIds.length === allDocs.length
                  ? setSelectedIds([])
                  : setSelectedIds(allDocs.map(d => d.id))
              }
              style={{
                background: "none",
                border: "1px solid rgba(129,140,248,0.4)",
                borderRadius: "6px",
                color: "#818cf8",
                fontSize: "11px",
                fontWeight: "600",
                padding: "3px 10px",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {selectedIds.length === allDocs.length ? "✕ Deselect All" : "✓ Select All"}
            </button>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
            {allDocs.map(doc => {
              const checked = selectedIds.includes(doc.id);
              const name = (doc.subject && doc.subject !== "General") ? doc.subject : doc.filename;
              return (
                <label key={doc.id} style={{
                  display: "flex", alignItems: "center", gap: "6px",
                  padding: "4px 10px", borderRadius: "9999px", cursor: "pointer",
                  fontSize: "12px", fontWeight: checked ? "600" : "400",
                  backgroundColor: checked ? "rgba(79,70,229,0.25)" : "rgba(255,255,255,0.05)",
                  border: checked ? "1px solid rgba(79,70,229,0.5)" : "1px solid rgba(255,255,255,0.1)",
                  color: checked ? "#c7d2fe" : "#6b7280",
                  transition: "all 0.15s"
                }}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => setSelectedIds(prev =>
                      prev.includes(doc.id) ? prev.filter(id => id !== doc.id) : [...prev, doc.id]
                    )}
                    style={{ accentColor: "#818cf8", cursor: "pointer" }}
                  />
                  📄 {name}
                </label>
              );
            })}
          </div>
          <div style={{ marginTop: "6px", fontSize: "11px", color: "#6b7280" }}>
            {selectedIds.length === 0
              ? "⚠️ Select at least one document to generate questions"
              : `${selectedIds.length} of ${allDocs.length} document(s) selected`}
          </div>
        </div>
      )}

      <div className="practicebox-controls">
        <div className="practicebox-control-group">
          <label>Difficulty</label>
          <select value={difficulty} onChange={(e) => setDifficulty(e.target.value)}>
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>

        <div className="practicebox-control-group">
          <label>Type</label>
          <select value={questionType} onChange={(e) => setQuestionType(e.target.value)}>
            <option value="mcq">MCQ</option>
            <option value="theory">Theory</option>
            <option value="numerical">Numerical</option>
          </select>
        </div>

        <div className="practicebox-control-group">
          <label>Count</label>
          <input
            type="number"
            min="1"
            max="20"
            value={count}
            onChange={(e) => setCount(e.target.value)}
          />
        </div>

        <button 
          className="practicebox-generate-button"
          onClick={handleGenerate}
          disabled={isGenerating}
        >
          {isGenerating ? "Generating..." : "Generate Problems"}
        </button>
      </div>

      {problems.length > 0 && (
        <div className="practicebox-problem-area">
          <div className="practicebox-progress">
            Problem {currentIndex + 1} of {problems.length}
            {currentProblem && (
              <span className={`practicebox-difficulty-badge ${currentProblem.difficulty}`}>
                {currentProblem.difficulty.toUpperCase()}
              </span>
            )}
          </div>

          <div className="practicebox-question">
            <h3>{currentProblem.question}</h3>
          </div>

          {currentProblem.question_type === "mcq" && currentProblem.options && (
            <div className="practicebox-options">
              {currentProblem.options.map((option) => (
                <label key={option.label} className="practicebox-option">
                  <input
                    type="radio"
                    name="answer"
                    value={option.label}
                    checked={userAnswer === option.label}
                    onChange={(e) => setUserAnswer(e.target.value)}
                    disabled={showSolution}
                  />
                  <span>{option.label}. {option.text}</span>
                </label>
              ))}
            </div>
          )}

          {currentProblem.question_type !== "mcq" && (
            <div className="practicebox-textarea-wrapper">
              <textarea
                className="practicebox-textarea"
                value={userAnswer}
                onChange={(e) => setUserAnswer(e.target.value)}
                placeholder="Enter your answer here..."
                rows="6"
                disabled={showSolution}
              />
            </div>
          )}

          {!showSolution && (
            <div className="practicebox-actions">
              <button 
                className="practicebox-hint-button"
                onClick={() => setShowHints(!showHints)}
              >
                {showHints ? "Hide" : "Show"} Hints 💡
              </button>
              <button 
                className="practicebox-submit-button"
                onClick={handleSubmit}
                disabled={!userAnswer.trim() || isSubmitting}
              >
                {isSubmitting ? "Submitting..." : "Submit Answer"}
              </button>
            </div>
          )}

          {showHints && currentProblem.hints && currentProblem.hints.length > 0 && (
            <div className="practicebox-hints">
              <h4>💡 Hints:</h4>
              <ul>
                {currentProblem.hints.map((hint, idx) => (
                  <li key={idx}>{hint}</li>
                ))}
              </ul>
            </div>
          )}

          {feedback && (
            <div className={`practicebox-feedback ${feedback.is_correct ? "correct" : "incorrect"}`}>
              <div className="practicebox-feedback-header">
                <strong>{feedback.feedback}</strong>
                {score !== null && <span className="practicebox-score">Score: {score}/100</span>}
              </div>
              <p><strong>Correct Answer:</strong> {feedback.correct_answer}</p>
            </div>
          )}

          {showSolution && currentProblem.solution && (
            <div className="practicebox-solution">
              <h4>📖 Solution:</h4>
              <p>{currentProblem.solution}</p>
            </div>
          )}

          <div className="practicebox-navigation">
            <button 
              onClick={handlePrevious} 
              disabled={currentIndex === 0}
              className="practicebox-nav-button"
            >
              ← Previous
            </button>
            <button 
              onClick={handleReset}
              className="practicebox-nav-button practicebox-reset-button"
            >
              🔄 Reset
            </button>
            <button 
              onClick={handleNext} 
              disabled={currentIndex === problems.length - 1}
              className="practicebox-nav-button"
            >
              Next →
            </button>
          </div>
        </div>
      )}

      {problems.length === 0 && !isGenerating && (
        <div className="practicebox-empty">
          <p>Configure settings above and click "Generate Problems" to start practicing</p>
        </div>
      )}
    </div>
  );
}
