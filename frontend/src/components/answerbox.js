import { useEffect, useState } from "react";
import "./answerbox.css";

export default function AnswerBox({ text }) {
  const [selected, setSelected] = useState({});
  const [showAnswers, setShowAnswers] = useState({});
  const [showSources, setShowSources] = useState(true);

  useEffect(() => {
    setSelected({});
    setShowAnswers({});
  }, [text]);

  if (!text) {
    return null;
  }

  if (typeof text === "object" && text.kind === "loading") {
    return (
      <div className="answer-box loading">
        <div className="loading-spinner"></div>
        <p>Thinking...</p>
      </div>
    );
  }

  // Helper: Render Confidence Bar
  const renderConfidence = (score) => {
    if (score === undefined || score === null) return null;
    const percentage = Math.round(score * 100);
    let colorClass = "low";
    let statusText = "Low Grounding";
    let emoji = "🔴";

    if (score >= 0.8) {
      colorClass = "high";
      statusText = "Highly Grounded";
      emoji = "🟢";
    } else if (score >= 0.6) {
      colorClass = "moderate";
      statusText = "Moderately Grounded";
      emoji = "🟡";
    }

    return (
      <div className={`confidence-meter ${colorClass}`}>
        <span className="confidence-emoji">{emoji}</span>
        <span className="confidence-label">{statusText}</span>
        <div className="confidence-progress-bg">
          <div className="confidence-progress-bar" style={{ width: `${percentage}%` }} />
        </div>
        <span className="confidence-percentage">{percentage}%</span>
      </div>
    );
  };

  // Helper: Render Answer text with clickable inline citations
  const renderAnswerText = (responseText, sources = []) => {
    if (!responseText) return null;

    const regex = /\[(\d+)\]/g;
    const parts = responseText.split(regex);

    return parts.map((part, idx) => {
      // Odd indices are captured source_ids
      if (idx % 2 === 1) {
        const sourceNum = parseInt(part, 10);
        const source = sources.find((s) => s.source_id === sourceNum);

        const pagesList = source?.pages && Array.isArray(source.pages) ? source.pages : [];
        const tooltipText = source
          ? `${source.filename}${pagesList.length > 0 ? ` (Page ${pagesList.join(", ")})` : ""}`
          : `Source ${sourceNum}`;

        return (
          <sup key={idx} className="citation-sup">
            <button
              className="citation-badge"
              title={tooltipText}
              onClick={() => {
                const element = document.getElementById(`source-card-${sourceNum}`);
                if (element) {
                  element.scrollIntoView({ behavior: "smooth", block: "center" });
                  element.classList.add("highlight-source");
                  setTimeout(() => {
                    element.classList.remove("highlight-source");
                  }, 2000);
                }
              }}
            >
              {sourceNum}
            </button>
          </sup>
        );
      }

      return part;
    });
  };

  // Helper: Render Sources References Grid
  const renderSourcesSection = (cleanSources) => {
    if (!cleanSources || cleanSources.length === 0) return null;

    return (
      <div className="sources-container">
        <div className="sources-header" onClick={() => setShowSources(!showSources)}>
          <span className="sources-title">📚 References & Sources ({cleanSources.length})</span>
          <span className="sources-toggle-icon">{showSources ? "▼" : "▲"}</span>
        </div>

        {showSources && (
          <div className="sources-grid">
            {cleanSources.map((src) => {
              const pagesList = src.pages && Array.isArray(src.pages) ? src.pages : [];
              return (
                <div key={src.source_id} id={`source-card-${src.source_id}`} className="source-card">
                  <div className="source-card-header">
                    <span className="source-number-badge">[{src.source_id}]</span>
                    <span className="source-filename" title={src.filename}>
                      {src.filename}
                    </span>
                  </div>
                  <div className="source-card-meta">
                    {src.subject && <span className="source-tag subject-tag">{src.subject}</span>}
                    {pagesList.length > 0 && (
                      <span className="source-tag page-tag">Page {pagesList.join(", ")}</span>
                    )}
                  </div>
                  {src.chunk_preview && (
                    <p className="source-preview" title="Source context snippet">
                      "{src.chunk_preview}"
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  // MCQ Rendering Mode
  if (typeof text === "object" && text.kind === "mcq") {
    const questions = text.value?.questions || [];

    const handleReset = (key) => {
      const newSelected = { ...selected };
      delete newSelected[key];
      setSelected(newSelected);

      const newShowAnswers = { ...showAnswers };
      delete newShowAnswers[key];
      setShowAnswers(newShowAnswers);
    };

    const handleShowAnswer = (key, correctAnswer) => {
      setShowAnswers({ ...showAnswers, [key]: true });
      setSelected({ ...selected, [key]: correctAnswer });
    };

    return (
      <div className="answer-box mcq-container">
        <h3 className="mcq-header">📝 Self-Assessment Quiz ({questions.length})</h3>

        {questions.map((q, idx) => {
          const key = `q-${idx}`;
          const selectedOption = selected[key];
          const isCorrect = selectedOption && selectedOption === q.correct_answer;
          const isAnswered = selectedOption !== undefined;

          return (
            <div key={key} className="mcq-question">
              <div className="mcq-question-text">
                <span className="mcq-number">Q{idx + 1}</span>
                {q.question}
              </div>

              <div className="mcq-options">
                {q.options.map((opt) => {
                  const isSelected = selectedOption === opt.label;
                  const isCorrectOption = opt.label === q.correct_answer;

                  let optionClass = "mcq-option";
                  if (isAnswered) {
                    if (isSelected && isCorrect) {
                      optionClass += " correct";
                    } else if (isSelected && !isCorrect) {
                      optionClass += " wrong";
                    } else if (isCorrectOption) {
                      optionClass += " correct-answer";
                    }
                  } else if (isSelected) {
                    optionClass += " selected";
                  }

                  return (
                    <button
                      key={opt.label}
                      className={optionClass}
                      onClick={() => !isAnswered && setSelected({ ...selected, [key]: opt.label })}
                      disabled={isAnswered}
                    >
                      <span className="option-label">{opt.label}</span>
                      <span className="option-text">{opt.text}</span>
                    </button>
                  );
                })}
              </div>

              <div className="mcq-action-buttons">
                {!isAnswered && (
                  <button className="btn-show-answer" onClick={() => handleShowAnswer(key, q.correct_answer)}>
                    💡 Show Answer & Explanation
                  </button>
                )}
                {isAnswered && (
                  <button className="btn-reset" onClick={() => handleReset(key)}>
                    🔄 Reset
                  </button>
                )}
              </div>

              {isAnswered && (
                <>
                  <div className={`mcq-result ${isCorrect ? "correct" : "wrong"}`}>
                    {isCorrect ? <>✅ Correct! Well done!</> : <>❌ Incorrect. The correct answer is {q.correct_answer}</>}
                  </div>
                  {q.explanation && (
                    <div className="mcq-explanation">
                      <div className="explanation-icon">💡</div>
                      <div className="explanation-content">
                        <h4 className="explanation-title">Why is {q.correct_answer} correct?</h4>
                        <p className="explanation-text">{q.explanation}</p>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  // Text Answer Mode
  if (typeof text === "object" && text.kind === "text") {
    const rawVal = text.value;

    let answerBody = "";
    let cleanSources = [];
    let confidenceScore = null;

    if (rawVal && typeof rawVal === "object") {
      const fullText = rawVal.answer || "";
      const splitParts = fullText.split("📚 **Sources:**");
      answerBody = splitParts[0];

      // Strip confidence footer lines
      answerBody = answerBody
        .split("*🟢 High confidence")[0]
        .split("*🟡 Moderate confidence")[0]
        .split("*🔴 Low confidence")[0]
        .trim();

      cleanSources = rawVal.sources || [];
      confidenceScore = rawVal.confidence?.overall_confidence ?? null;
    } else if (typeof rawVal === "string") {
      const splitParts = rawVal.split("📚 **Sources:**");
      answerBody = splitParts[0];
      answerBody = answerBody
        .split("*🟢 High confidence")[0]
        .split("*🟡 Moderate confidence")[0]
        .split("*🔴 Low confidence")[0]
        .trim();
    }

    return (
      <div className="answer-box text-answer">
        <div className="answer-header-row">
          <div className="answer-icon">🤖</div>
          {renderConfidence(confidenceScore)}
        </div>
        <div className="answer-content">
          {renderAnswerText(answerBody, cleanSources)}
        </div>

        {renderSourcesSection(cleanSources)}
      </div>
    );
  }

  const value = typeof text === "object" ? JSON.stringify(text, null, 2) : text;

  return <div className="answer-box">{value}</div>;
}
