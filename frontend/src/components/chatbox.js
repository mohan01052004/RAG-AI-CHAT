import { useState, useEffect } from "react";
import { askQuestion, getMcqs } from "../api";
import "./chatbox.css";

export default function ChatBox({
  onAnswer,
  documentId,
  subject,
  documents = [],
  selectedDocumentIds = [],
  onSelectDocuments
}) {
  const [q, setQ] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showDocPanel, setShowDocPanel] = useState(false);
  const [isGeneralChat, setIsGeneralChat] = useState(true);

  useEffect(() => {
    setIsGeneralChat(documents.length === 0);
  }, [documents.length]);

  const searchAll = documents.length > 0 && selectedDocumentIds.length === documents.length;

  const effectiveSelectedIds = selectedDocumentIds;

  const handleTogglePanel = () => {
    setShowDocPanel(prev => !prev);
  };

  const handleDocumentToggle = (docId) => {
    if (onSelectDocuments) {
      const isSelected = selectedDocumentIds.includes(docId);
      const next = isSelected
        ? selectedDocumentIds.filter(id => id !== docId)
        : [...selectedDocumentIds, docId];
      onSelectDocuments(next);
    }
  };

  const handleSearchAllToggle = (checked) => {
    if (onSelectDocuments) {
      if (checked) {
        onSelectDocuments(documents.map(doc => doc.id));
      } else {
        onSelectDocuments([]);
      }
    }
  };

  const ask = async () => {
    if (!q.trim()) return;

    setIsLoading(true);
    onAnswer({ kind: "loading" });

    const isMcq = q.toLowerCase().includes("mcq");
    const docIds = isGeneralChat ? [] : (effectiveSelectedIds.length ? effectiveSelectedIds : null);
    const subjectFilter = !isGeneralChat && docIds && docIds.length === 1 ? subject : null;

    try {
      if (isMcq) {
        const res = await getMcqs(q, {
          subject: subjectFilter,
          document_id: docIds && docIds.length === 1 ? docIds[0] : null,
          document_ids: docIds
        });
        onAnswer({ kind: "mcq", value: res.data });
      } else {
        const res = await askQuestion(q, {
          subject: subjectFilter,
          document_id: docIds && docIds.length === 1 ? docIds[0] : null,
          document_ids: docIds
        });
        onAnswer({ kind: "text", value: res.data.answer });
      }
    } catch (error) {
      const errorMsg = error.response?.data?.detail || error.message || "An error occurred";
      if (error.response?.status === 429 || errorMsg.toLowerCase().includes("rate") || errorMsg.toLowerCase().includes("limit")) {
        onAnswer({ kind: "text", value: "⚠️ Rate limit reached. Please wait a moment and try again." });
      } else {
        onAnswer({ kind: "text", value: `❌ Error: ${errorMsg}` });
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  };

  const placeholderText = isGeneralChat
    ? "Ask anything to the AI Assistant..."
    : "Ask a question about your documents...";

  return (
    <div className="chatbox-container">
      <div className="chatbox-documents">
        <div className="chatbox-documents-header">
          <div className="chatbox-mode-toggle-group">
            <span className="chatbox-mode-label">Mode:</span>
            <button
              type="button"
              className={`chatbox-mode-toggle-btn ${isGeneralChat ? "active" : ""}`}
              onClick={() => setIsGeneralChat(true)}
            >
              🤖 General AI
            </button>
            <button
              type="button"
              className={`chatbox-mode-toggle-btn ${!isGeneralChat ? "active" : ""} ${documents.length === 0 ? "disabled" : ""}`}
              onClick={() => documents.length > 0 && setIsGeneralChat(false)}
              disabled={documents.length === 0}
              title={documents.length === 0 ? "Upload a PDF to enable Document Search" : ""}
            >
              📂 Document Search
            </button>
          </div>
          
          {!isGeneralChat && documents.length > 0 && (
            <button 
              className="chatbox-documents-button"
              onClick={handleTogglePanel}
              type="button"
            >
              {effectiveSelectedIds.length === 0 || (documents.length > 0 && effectiveSelectedIds.length === documents.length)
                ? "All documents" 
                : `${effectiveSelectedIds.length} selected`}
            </button>
          )}
        </div>
        
        {!isGeneralChat && showDocPanel && documents.length > 0 && (
          <div className="chatbox-documents-panel">
            <label className="chatbox-documents-checkbox-item chatbox-documents-all">
              <input
                type="checkbox"
                checked={searchAll}
                onChange={(e) => handleSearchAllToggle(e.target.checked)}
              />
              <span>Search all documents</span>
            </label>
            
            <div className="chatbox-documents-divider" />
            
            <div className="chatbox-documents-list">
              {documents.map((doc) => (
                <label key={doc.id} className="chatbox-documents-checkbox-item">
                  <input
                    type="checkbox"
                    checked={selectedDocumentIds.includes(doc.id)}
                    onChange={() => handleDocumentToggle(doc.id)}
                  />
                  <span>{doc.subject ? `${doc.subject} — ` : ""}{doc.filename}</span>
                </label>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="chatbox-input-wrapper">
        <input
          className="chatbox-input"
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder={placeholderText}
          disabled={isLoading}
        />
        <button 
          className="chatbox-send-button" 
          onClick={ask}
          disabled={!q.trim() || isLoading}
        >
          {isLoading ? '⏳' : 'Send'}
        </button>
      </div>
      <p className="chatbox-hint">
        {isGeneralChat
          ? "💡 Ask a general question or upload a document to get context-specific answers."
          : "💡 Try: \"Summarize the key findings\" or \"What are the main concepts?\""}
      </p>
    </div>
  );
}
