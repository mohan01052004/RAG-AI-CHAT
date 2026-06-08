import { useState, useRef } from "react";
import { uploadPDF } from "../api";
import "./uploadbox.css";

export default function UploadBox({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState("");
  const [documentId, setDocumentId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleUpload = async () => {
    if (!file) return alert("Please select a PDF file");
    if (!subject) return alert("Please enter a subject name");

    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("subject", subject);

      const res = await uploadPDF(form);
      const docId = res?.data?.document_id ?? null;
      setDocumentId(docId);
      if (onUploaded) {
        onUploaded(docId, subject);
      }
      alert(`✅ Document uploaded and indexed successfully!`);
    } catch (error) {
      alert("❌ Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === "application/pdf") {
        setFile(droppedFile);
      } else {
        alert("Please upload a PDF file only");
      }
    }
  };

  const handleFileClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <div className="upload-container">
      <div
        className={`upload-dropzone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={handleFileClick}
      >
        <div className="upload-icon">📄</div>
        <h3 className="upload-title">Upload PDF Document</h3>
        <p className="upload-text">
          {file ? `Selected: ${file.name}` : 'Click to browse or drag and drop your PDF here'}
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
      </div>

      <div className="upload-controls">
        <input
          className="subject-input"
          type="text"
          placeholder="Document Category / Tag (e.g. Machine Learning, Biology)"
          value={subject}
          onChange={e => setSubject(e.target.value)}
        />
        <button 
          className="upload-button" 
          onClick={handleUpload}
          disabled={!file || !subject || uploading}
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>

      {documentId && (
        <div className="upload-success">
          ✅ Document indexed successfully (ID: {documentId})
        </div>
      )}
    </div>
  );
}
