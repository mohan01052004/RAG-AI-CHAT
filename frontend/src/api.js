import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000"
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
