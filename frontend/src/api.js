import axios from 'axios';

const BASE = typeof window !== 'undefined' && (window.location.origin.includes('localhost:5173') || window.location.origin.includes('127.0.0.1:5173'))
  ? 'http://localhost:8000'
  : window.location.origin;

export const api = {
  getStatus:       () => axios.get(`${BASE}/api/status`),
  getSearchResults:() => axios.get(`${BASE}/api/search-results`),
  getSearchLog:    () => axios.get(`${BASE}/api/search-log`),
  getArchitecture: (rank) => axios.get(`${BASE}/api/architecture/${rank}`),
  getFinetuneResults: () => axios.get(`${BASE}/api/finetune-results`),
  runInference:    (formData) => axios.post(`${BASE}/api/inference`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
};
