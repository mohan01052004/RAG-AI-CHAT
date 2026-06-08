import { useEffect, useState } from "react";
import ChatBox from "./components/chatbox";
import AnswerBox from "./components/answerbox";
import PracticeBox from "./components/practicebox";
import ChatWithDocs from "./components/ChatWithDocs";
import { getDocuments } from "./api";
import './App.css';

function App() {
  const [answer, setAnswer] = useState(null);
  const [documentId, setDocumentId] = useState(null);
  const [subject, setSubject] = useState("");
  const [hasDocument, setHasDocument] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [activeTab, setActiveTab] = useState("query"); // "query" or "practice"

  const loadDocuments = async () => {
    try {
      const res = await getDocuments();
      const docs = res.data?.documents || [];
      setDocuments(docs);
      if (docs.length > 0) {
        setHasDocument(true);
        setSelectedDocumentIds(prev => prev.length === 0 ? docs.map(d => d.id) : prev);
      }
    } catch {
      setDocuments([]);
    }
  };


  useEffect(() => {
    loadDocuments();
  }, []);

  return (
    <div className="app-container">
      <div className="header">
        <div className="header-icon">🧠</div>
        <h1 className="header-title">RAG AI Chat</h1>
        <p className="header-subtitle">Upload your PDF documents and ask questions with interactive citations</p>
      </div>

      <div className="content-container">
        <div className="mode-tabs">
          <button
            className={`mode-tab ${activeTab === "query" ? "active" : ""}`}
            onClick={() => setActiveTab("query")}
          >
            <span className="tab-icon">💬</span>
            Chat Assistant
          </button>
          <button
            className={`mode-tab ${activeTab === "practice" ? "active" : ""} ${!hasDocument ? "disabled" : ""}`}
            onClick={() => hasDocument && setActiveTab("practice")}
            disabled={!hasDocument}
            title={!hasDocument ? "Upload a PDF to unlock quizzes" : ""}
          >
            <span className="tab-icon">📝</span>
            Self-Assessment Quiz
          </button>
        </div>

        {/* Always mounted — CSS hidden when not active to preserve chat state */}
        <div
          className="chat-section"
          style={{ width: "100%", marginTop: "16px", display: activeTab === "query" ? "block" : "none" }}
        >
          <ChatWithDocs
            onUploaded={(docId) => {
              setHasDocument(true);
              loadDocuments();
            }}
          />
        </div>

        {hasDocument && activeTab === "practice" && (
          <div className="practice-section">
            <PracticeBox
              documentIds={selectedDocumentIds}
              subject={subject}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
