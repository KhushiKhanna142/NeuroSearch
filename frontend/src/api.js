import axios from 'axios';

const BASE = 'http://localhost:8000';

export const api = {
  getStatus:       () => axios.get(`${BASE}/api/status`),
  getSearchResults:() => axios.get(`${BASE}/api/search-results`),
  getSearchLog:    () => axios.get(`${BASE}/api/search-log`),
  getArchitecture: (rank) => axios.get(`${BASE}/api/architecture/${rank}`),
  runInference:    (formData) => axios.post(`${BASE}/api/inference`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
};
