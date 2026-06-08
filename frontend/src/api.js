import axios from "axios";

// Determine the API base URL:
// 1. Use REACT_APP_API_URL env var if set at build time (Vercel env var or .env.production)
// 2. Fall back to Render URL when running on any non-localhost domain (i.e. deployed on Vercel)
// 3. Fall back to localhost for local development
const getBaseURL = () => {
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  if (typeof window !== "undefined" && !window.location.hostname.includes("localhost")) {
    return "https://rag-ai-chat-st6i.onrender.com";
  }
  return "http://127.0.0.1:8000";
};

const API = axios.create({
  baseURL: getBaseURL()
});

export const uploadPDF = (formData) =>
  API.post("/upload", formData);

export const getDocuments = () =>
  API.get("/documents");

export const getDocumentHierarchy = () =>
  API.get("/documents/hierarchy");

export const askQuestion = (question, options = {}) =>
  API.post("/ask", {
    question,
    subject: options.subject || null,
    document_id: options.document_id || null,
    document_ids: options.document_ids || null
  });

export const getMcqs = (question, options = {}) =>
  API.post("/mcq", {
    question,
    subject: options.subject || null,
    document_id: options.document_id || null,
    document_ids: options.document_ids || null
  });

export const generatePractice = (options = {}) =>
  API.post("/practice/generate", {
    topic: options.topic || null,
    subject: options.subject || null,
    difficulty: options.difficulty || "medium",
    count: options.count || 5,
    question_type: options.question_type || "mcq",
    document_id: options.document_id || null,
    document_ids: options.document_ids || null
  });

export const submitPractice = (submission) =>
  API.post("/practice/submit", submission);

export const getPerformanceStats = (params = {}) =>
  API.get("/practice/stats", { params });

export default API;
