import React, { useState, useEffect, useRef } from "react";
import API from "../api";

export default function ChatWithDocs({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [docName, setDocName] = useState(""); // user-given name for the document
  const [uploadStatus, setUploadStatus] = useState(""); // "", "uploading", "done", "error"
  const [uploadError, setUploadError] = useState(""); // actual error message to display
  const [uploadedDocs, setUploadedDocs] = useState([]);

  const [allDocs, setAllDocs] = useState([]);
  // selectedDocIds stores numbers (matching doc.id type from DB)
  const [selectedDocIds, setSelectedDocIds] = useState([]);
  const [searchAll, setSearchAll] = useState(false); // default: nothing selected
  const [deletingDocId, setDeletingDocId] = useState(null); // track which doc is being deleted
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  const [message, setMessage] = useState("");
  const [documentHistory, setDocumentHistory] = useState([]);
  const [generalHistory, setGeneralHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneralMode, setIsGeneralMode] = useState(false);

  const chatEndRef = useRef(null);

  // ─── Fetch all docs from server ───────────────────────────────────────────
  const fetchAllDocuments = async () => {
    try {
      const response = await API.get("/documents");
      setAllDocs(response.data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents:", err);
    }
  };

  useEffect(() => {
    fetchAllDocuments();
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [documentHistory, generalHistory, isLoading]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // ─── Document Selection Logic ─────────────────────────────────────────────
  const handleDocCheckboxChange = (docId) => {
    const numId = Number(docId);
    setSelectedDocIds((prev) => {
      // If searchAll is active, start with all documents selected
      let currentSelection;
      if (searchAll) {
        currentSelection = allDocs.map((d) => Number(d.id));
      } else {
        currentSelection = prev;
      }

      const isSelected = currentSelection.includes(numId);
      const newSelection = isSelected
        ? currentSelection.filter((id) => id !== numId)
        : [...currentSelection, numId];

      // If all documents are selected, or none are selected, default to searchAll
      if (newSelection.length === allDocs.length || newSelection.length === 0) {
        setSearchAll(true);
        return [];
      } else {
        setSearchAll(false);
        return newSelection;
      }
    });
  };

  const handleSearchAllChange = () => {
    if (searchAll) {
      setSearchAll(false);
      setSelectedDocIds([]);
    } else {
      setSearchAll(true);
      setSelectedDocIds([]);
    }
  };

  // ─── Delete Document ──────────────────────────────────────────────────────
  const handleDeleteDoc = async (docId, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this document? This cannot be undone.")) return;
    setDeletingDocId(docId);
    try {
      await API.delete(`/api/documents/${docId}`);
      // Remove from selected if it was selected
      setSelectedDocIds((prev) => prev.filter((id) => id !== docId));
      // Refresh document list
      await fetchAllDocuments();
    } catch (err) {
      console.error("Failed to delete document:", err);
      alert("Failed to delete document. Please try again.");
    } finally {
      setDeletingDocId(null);
    }
  };

  // ─── File Input ───────────────────────────────────────────────────────────
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      setFile(f);
      setUploadStatus("");
      // Pre-fill doc name from filename (strip extension)
      if (!docName) {
        setDocName(f.name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  // ─── Upload ───────────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return;
    const nameToUse = docName.trim() || file.name.replace(/\.[^/.]+$/, "");
    setUploadStatus("uploading");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("subject", nameToUse); // send as subject/label

    try {
      // DO NOT set Content-Type manually — Axios sets it automatically with the
      // correct multipart boundary when FormData is passed. Manually setting it
      // without the boundary causes the backend to fail parsing the form.
      const response = await API.post("/api/documents/upload", formData);

      const { doc_id, chunk_count } = response.data;
      const newDoc = {
        doc_id,
        filename: file.name,
        display_name: nameToUse,
        chunk_count,
      };

      setUploadedDocs((prev) => [...prev, newDoc]);
      setUploadStatus("done");
      setUploadError("");
      setFile(null);
      setDocName("");
      fetchAllDocuments();

      if (onUploaded) onUploaded(doc_id);

      const fileInput = document.getElementById("file-upload-input");
      if (fileInput) fileInput.value = "";
    } catch (err) {
      console.error("Upload error:", err);
      // Show the actual backend error message if available
      const detail = err?.response?.data?.detail;
      const status = err?.response?.status;
      const networkErr = !err?.response ? "Network error — cannot reach server" : null;
      const msg = networkErr || (detail ? `${detail}` : status ? `Server error ${status}` : err.message || "Unknown error");
      setUploadError(msg);
      setUploadStatus("error");
    }
  };

  // ─── Send Message ─────────────────────────────────────────────────────────
  const handleSendMessage = async (e) => {
    if (e) e.preventDefault();
    if (!message.trim()) return;

    const currentMsg = message;
    setMessage("");

    const activeHistory = isGeneralMode ? generalHistory : documentHistory;
    const setHistory = isGeneralMode ? setGeneralHistory : setDocumentHistory;

    const updatedHistory = [
      ...activeHistory,
      { role: "user", content: currentMsg },
    ];
    setHistory(updatedHistory);
    setIsLoading(true);

    try {
      let response;
      if (isGeneralMode) {
        response = await API.post("/api/chat", {
          message: currentMsg,
        });
      } else {
        // doc_id: null means search ALL; otherwise pass array of selected IDs as strings
        const docIdPayload =
          searchAll || selectedDocIds.length === 0
            ? null
            : selectedDocIds.map(String);

        response = await API.post("/api/chat/rag", {
          message: currentMsg,
          conversation_history: updatedHistory.map((h) => ({
            role: h.role,
            content: h.content,
          })),
          doc_id: docIdPayload,
        });
      }

      const data = response.data;
      const answer =
        typeof data === "string"
          ? data
          : data.answer || data.response || "";
      const mode = data.mode || (isGeneralMode ? "general" : "document");
      const sources = data.sources || [];

      setHistory((prev) => [
        ...prev,
        { role: "assistant", content: answer, mode, sources },
      ]);
    } catch (err) {
      console.error(err);
      setHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "⚠️ Error: Failed to generate response from server.",
          mode: "general",
          sources: [],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Strip the prefix warning from general answers before displaying
  const getDisplayAnswer = (msg) => {
    const warningPrefix =
      "⚠️ This question doesn't seem related to your uploaded documents.\nHere's a general answer:\n\n";
    if (msg.mode === "general" && msg.content.startsWith(warningPrefix)) {
      return msg.content.substring(warningPrefix.length);
    }
    return msg.content;
  };

  // ─── Dropdown label ───────────────────────────────────────────────────────
  const dropdownLabel = () => {
    if (searchAll) return "All Documents";
    if (selectedDocIds.length === 0) return "Select documents…";
    if (selectedDocIds.length === 1) {
      const found = allDocs.find((d) => d.id === selectedDocIds[0]);
      return found?.subject && found.subject !== "General"
        ? found.subject
        : found?.filename || "1 Document";
    }
    return `${selectedDocIds.length} Documents Selected`;
  };

  return (
    <div className="chat-card">
      {/* ── 1. Document Upload Section ─────────────────────────────────── */}
      <div className="chat-section-card">
        <h2 className="chat-section-title">
          <span>📂</span> Document Center
        </h2>

        {/* File picker row */}
        <div
          className="upload-row file-picker-row"
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "10px",
            marginBottom: "10px",
          }}
        >
          <input
            id="file-upload-input"
            type="file"
            accept=".pdf,.docx,.txt"
            onChange={handleFileChange}
            style={{ fontSize: "13px", color: "#c7d2fe", cursor: "pointer" }}
          />
        </div>

        {/* Document name + upload button row */}
        <div
          className="upload-row doc-name-row"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            flexWrap: "wrap",
          }}
        >
          <input
            type="text"
            placeholder="Document name (e.g. Aptitude Resource)"
            value={docName}
            onChange={(e) => setDocName(e.target.value)}
            className="chat-input-text"
          />

          <button
            onClick={handleUpload}
            disabled={!file || uploadStatus === "uploading"}
            className="chat-btn"
          >
            {uploadStatus === "uploading" ? "Uploading…" : "Upload File"}
          </button>

          {/* Status badge */}
          {uploadStatus && (
            <span
              style={{
                fontSize: "12px",
                fontWeight: "500",
                padding: "2px 10px",
                borderRadius: "9999px",
                backgroundColor:
                  uploadStatus === "uploading"
                    ? "rgba(245,158,11,0.2)"
                    : uploadStatus === "done"
                    ? "rgba(16,185,129,0.2)"
                    : "rgba(239,68,68,0.2)",
                color:
                  uploadStatus === "uploading"
                    ? "#f59e0b"
                    : uploadStatus === "done"
                    ? "#10b981"
                    : "#ef4444",
              }}
            >
              {uploadStatus === "uploading" && "Uploading document…"}
              {uploadStatus === "done" && "✓ Done"}
              {uploadStatus === "error" && `✗ ${uploadError || "Upload failed"}`}
            </span>
          )}
        </div>

        {/* Uploaded this session */}
        {uploadedDocs.length > 0 && (
          <div style={{ marginTop: "12px" }}>
            <p
              style={{
                fontSize: "12px",
                color: "#9ca3af",
                marginBottom: "8px",
              }}
            >
              Uploaded this session ({uploadedDocs.length}):
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
              {uploadedDocs.map((doc, idx) => (
                <span
                  key={idx}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    padding: "4px 10px",
                    borderRadius: "9999px",
                    fontSize: "12px",
                    fontWeight: "600",
                    backgroundColor: "rgba(79,70,229,0.3)",
                    color: "#c7d2fe",
                    border: "1px solid rgba(79,70,229,0.4)",
                    gap: "6px",
                  }}
                >
                  📄 {doc.display_name || doc.filename}
                  <span
                    style={{
                      fontSize: "10px",
                      color: "#818cf8",
                    }}
                  >
                    ID#{doc.doc_id} · {doc.chunk_count} chunks
                  </span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── Mode Toggle + Document Selector ────────────────────────────── */}
      <div className="chat-controls-row">
        {/* Mode toggle */}
        <div className="chat-mode-toggle-container">
          <button
            onClick={() => setIsGeneralMode(false)}
            className={`chat-mode-btn ${!isGeneralMode ? "active" : ""}`}
          >
            🔍 Document Search
          </button>
          <button
            onClick={() => setIsGeneralMode(true)}
            className={`chat-mode-btn ${isGeneralMode ? "active" : ""}`}
          >
            🌐 General AI
          </button>
        </div>

        {/* Document selector dropdown — only in Document Search mode */}
        {!isGeneralMode && (
          <div ref={dropdownRef} style={{ position: "relative" }}>
            <button
              onClick={() => setIsDropdownOpen((o) => !o)}
              className="chat-dropdown-trigger"
            >
              📂 {dropdownLabel()}
              <span style={{ fontSize: "10px", marginLeft: "4px" }}>▼</span>
            </button>

            {isDropdownOpen && (
              <div
                className="document-dropdown-menu"
                style={{
                  position: "absolute",
                  top: "calc(100% + 6px)",
                  right: 0,
                  zIndex: 50,
                  width: "320px",
                  backgroundColor: "rgba(17,12,46,0.97)",
                  backdropFilter: "blur(16px)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  boxShadow: "0 10px 25px -5px rgba(0,0,0,0.6)",
                  padding: "12px",
                  maxHeight: "380px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "8px",
                }}
              >
                {/* Search All option */}
                <label
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    padding: "8px 10px",
                    backgroundColor: searchAll
                      ? "rgba(59,130,246,0.15)"
                      : "rgba(255,255,255,0.05)",
                    borderRadius: "6px",
                    fontSize: "13px",
                    fontWeight: "600",
                    color: "#ffffff",
                    cursor: "pointer",
                    border: searchAll
                      ? "1px solid rgba(59,130,246,0.35)"
                      : "1px solid transparent",
                    transition: "all 0.15s",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={searchAll}
                    onChange={handleSearchAllChange}
                    style={{ cursor: "pointer", accentColor: "#3b82f6" }}
                  />
                  🔍 Search all documents
                </label>

                <div
                  style={{
                    height: "1px",
                    backgroundColor: "rgba(255,255,255,0.1)",
                    margin: "2px 0",
                  }}
                />

                {/* Documents list */}
                <div
                  style={{
                    overflowY: "auto",
                    display: "flex",
                    flexDirection: "column",
                    gap: "4px",
                    maxHeight: "250px",
                    paddingRight: "4px",
                  }}
                >
                  {allDocs.length === 0 ? (
                    <div
                      style={{
                        fontSize: "12px",
                        color: "#6b7280",
                        padding: "16px 10px",
                        textAlign: "center",
                      }}
                    >
                      No uploaded documents found.
                      <br />
                      Upload a document above to get started.
                    </div>
                  ) : (
                    allDocs.map((doc) => {
                      const numId = Number(doc.id);
                      const isChecked = searchAll || selectedDocIds.includes(numId);
                      const displayName =
                        doc.subject && doc.subject !== "General"
                          ? doc.subject
                          : doc.filename;
                      return (
                        <div
                          key={doc.id}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "6px",
                            padding: "6px 10px",
                            borderRadius: "5px",
                            fontSize: "12px",
                            transition: "background-color 0.15s",
                            backgroundColor: isChecked
                              ? "rgba(59,130,246,0.12)"
                              : "transparent",
                            border: isChecked
                              ? "1px solid rgba(59,130,246,0.25)"
                              : "1px solid transparent",
                          }}
                        >
                          {/* Checkbox + label (clickable area) */}
                          <label
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: "8px",
                              flex: 1,
                              cursor: "pointer",
                              overflow: "hidden",
                              color: isChecked ? "#ffffff" : "#9ca3af",
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={() => handleDocCheckboxChange(numId)}
                              style={{
                                cursor: "pointer",
                                accentColor: "#3b82f6",
                                flexShrink: 0,
                              }}
                            />
                            <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
                              <span
                                style={{
                                  textOverflow: "ellipsis",
                                  overflow: "hidden",
                                  whiteSpace: "nowrap",
                                  fontWeight: isChecked ? "600" : "400",
                                }}
                              >
                                📄 {displayName}
                              </span>
                              <span style={{ fontSize: "10px", color: "#6b7280" }}>
                                ID #{doc.id} · {doc.filename}
                              </span>
                            </div>
                          </label>

                          {/* Delete button */}
                          <button
                            onClick={(e) => handleDeleteDoc(numId, e)}
                            disabled={deletingDocId === numId}
                            title="Delete document"
                            style={{
                              flexShrink: 0,
                              background: "none",
                              border: "1px solid rgba(239,68,68,0.3)",
                              borderRadius: "4px",
                              color: deletingDocId === numId ? "#6b7280" : "#ef4444",
                              cursor: deletingDocId === numId ? "not-allowed" : "pointer",
                              fontSize: "11px",
                              padding: "2px 6px",
                              lineHeight: 1,
                              transition: "all 0.15s",
                            }}
                          >
                            {deletingDocId === numId ? "…" : "🗑"}
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Selection summary footer */}
                {!searchAll && selectedDocIds.length > 0 && (
                  <div
                    style={{
                      borderTop: "1px solid rgba(255,255,255,0.08)",
                      paddingTop: "8px",
                      fontSize: "11px",
                      color: "#818cf8",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <span>{selectedDocIds.length} selected</span>
                    <button
                      onClick={() => {
                        setSelectedDocIds([]);
                        setSearchAll(true);
                      }}
                      style={{
                        background: "none",
                        border: "none",
                        color: "#ef4444",
                        fontSize: "11px",
                        cursor: "pointer",
                        padding: "2px 6px",
                        borderRadius: "4px",
                      }}
                    >
                      Clear all
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Chat History ───────────────────────────────────────────────── */}
      <div className="chat-history-container">
        {(isGeneralMode ? generalHistory : documentHistory).length === 0 ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "280px",
              color: "#6b7280",
            }}
          >
            <span style={{ fontSize: "36px", marginBottom: "8px" }}>💬</span>
            <p style={{ fontSize: "14px", fontWeight: "500" }}>
              Start a conversation in{" "}
              {isGeneralMode ? "General AI" : "Document Search"} mode
            </p>
            {!isGeneralMode && allDocs.length === 0 && (
              <p
                style={{
                  fontSize: "12px",
                  color: "#4b5563",
                  marginTop: "6px",
                }}
              >
                Upload a document above to begin asking questions
              </p>
            )}
          </div>
        ) : (
          (isGeneralMode ? generalHistory : documentHistory).map(
            (msg, index) => {
              const isUser = msg.role === "user";
              return (
                <div
                  key={index}
                  className={`chat-message-row ${isUser ? "user" : "assistant"}`}
                >
                  {/* General warning banner */}
                  {!isUser &&
                    msg.mode === "general" &&
                    !isGeneralMode && (
                      <div className="chat-warning-banner">
                        <span>⚠️</span>
                        <span>
                          This question doesn't seem related to your uploaded
                          documents. Here's a general answer:
                        </span>
                      </div>
                    )}

                  {/* Message bubble */}
                  <div className={`chat-message-bubble ${isUser ? "user" : "assistant"}`}>
                    <div style={{ whiteSpace: "pre-wrap" }}>
                      {getDisplayAnswer(msg)}
                    </div>

                    {/* Citation badges */}
                    {!isUser &&
                      msg.mode === "document" &&
                      msg.sources &&
                      msg.sources.length > 0 && (
                        <div className="chat-citations-list">
                          {msg.sources.map((src, sidx) => (
                            <span key={sidx} className="chat-citation-tag">
                              <span>[{src.filename} · chunk {src.chunk_index}]</span>
                            </span>
                          ))}
                        </div>
                      )}

                    {/* Collapsible sources */}
                    {!isUser &&
                      msg.mode === "document" &&
                      msg.sources &&
                      msg.sources.length > 0 && (
                        <div className="chat-sources-details">
                          <details style={{ fontSize: "12px" }}>
                            <summary>View sources ({msg.sources.length})</summary>
                            <div className="chat-sources-list">
                              {msg.sources.map((src, sidx) => (
                                <div key={sidx} className="chat-source-item">
                                  <div
                                    style={{
                                      fontWeight: "600",
                                      color: "#d1d5db",
                                      marginBottom: "4px",
                                    }}
                                  >
                                    {src.filename} (Chunk {src.chunk_index})
                                  </div>
                                  <div
                                    style={{
                                      color: "#9ca3af",
                                      fontStyle: "italic",
                                    }}
                                  >
                                    "{src.excerpt}…"
                                  </div>
                                </div>
                              ))}
                            </div>
                          </details>
                        </div>
                      )}
                  </div>
                </div>
              );
            }
          )
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              color: "#9ca3af",
              fontSize: "14px",
              fontStyle: "italic",
            }}
          >
            <span
              style={{
                animation: "pulse 1.4s ease-in-out infinite",
                display: "inline-block",
              }}
            >
              ●
            </span>
            <span>AI is thinking…</span>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* ── Chat Input ─────────────────────────────────────────────────── */}
      <form onSubmit={handleSendMessage} className="chat-input-form">
        <input
          type="text"
          className="chat-input-field"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder={
            isGeneralMode
              ? "Ask any question…"
              : "Ask questions about uploaded documents…"
          }
        />
        <button
          type="submit"
          disabled={isLoading || !message.trim()}
          className="chat-input-submit"
        >
          Send
        </button>
      </form>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
}
