import axios from 'axios';

export const API_BASE = (import.meta.env.VITE_API_URL || '') + '/api';

const TOKEN_KEY = 'tabular_session_token';

export function getStoredToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setStoredToken(token) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_KEY);
    }
  } catch {
    // ignore storage exceptions
  }
}

export function clearStoredToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore
  }
}

const api = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
  timeout: 120000,
});

api.interceptors.request.use((config) => {
  const token = getStoredToken();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    let responseData = err.response?.data;
    if (typeof Blob !== 'undefined' && responseData instanceof Blob) {
      try {
        responseData = JSON.parse(await responseData.text());
      } catch {
        responseData = null;
      }
    }
    const detail = responseData?.detail;
    const message = typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map(item => item?.msg || String(item)).join('; ')
        : responseData?.message || err.message;
    const normalizedError = new Error(message || 'An unexpected error occurred.');
    normalizedError.response = err.response ? { ...err.response, data: responseData } : undefined;
    if (err.response?.status === 401) {
      const url = err.config?.url || '';
      if (!url.includes('/auth/login') && !url.includes('/auth/register')) {
        window.dispatchEvent(new CustomEvent('auth:unauthorized'));
      }
    }
    return Promise.reject(normalizedError);
  }
);

// ---- Datasets ----
export const uploadDataset = (file, onProgress) => {
  const form = new FormData();
  form.append('file', file);
  return api.post('/datasets/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress ? (progressEvent) => {
      const percentCompleted = progressEvent.total
        ? Math.round((progressEvent.loaded * 100) / progressEvent.total)
        : 0;
      onProgress(percentCompleted);
    } : undefined,
  });
};

export const listDatasets = (params) => api.get('/datasets', { params });
export const renameDataset = (id, name) => api.patch(`/datasets/${id}`, { name });
export const getDatasetDownloadUrl = (id) => {
  const token = getStoredToken();
  return `${API_BASE}/datasets/${id}/download${token ? `?token=${encodeURIComponent(token)}` : ''}`;
};
export const getDatasetPreview = (id, rows = 100) => api.get(`/datasets/${id}/preview?rows=${rows}`);
export const getDatasetProfile = (id) => api.get(`/datasets/${id}/profile`);
export const deleteDataset = (id) => api.delete(`/datasets/${id}`);
export const suggestTask = (id, target) => api.post(`/datasets/${id}/suggest-task`, { target_column: target });
export const getTargetDistribution = (id, target, taskType) =>
  api.get(`/datasets/${id}/target-distribution?target_column=${encodeURIComponent(target)}&task_type=${taskType}`);
export const listSampleDatasets = () => api.get('/datasets/sample/list');
export const loadSampleDataset = (filename) => api.post(`/datasets/sample/load/${filename}`);

// ---- Experiments ----
export const getAvailableModels = () => api.get('/experiments/models/available');
export const createExperiment = (payload) => api.post('/experiments', payload);
export const listExperiments = () => api.get('/experiments');
export const getExperiment = (id) => api.get(`/experiments/${id}`);
export const getExperimentStatus = (id) => api.get(`/experiments/${id}/status`);
export const getExperimentResults = (id) => api.get(`/experiments/${id}/results`);
export const getExplanations = (id) => api.get(`/experiments/${id}/explanations`);
export const deleteExperiment = (id) => api.delete(`/experiments/${id}`);

// ---- Predictions ----
export const predictSingle = (experimentId, features) =>
  api.post(`/models/${experimentId}/predict-single`, { features });
export const getModelSchema = (experimentId) => api.get(`/models/${experimentId}/schema`);
export const predictCsv = (experimentId, file) => {
  const form = new FormData();
  form.append('file', file);
  return api.post(`/models/${experimentId}/predict-csv`, form, {
    responseType: 'blob',
  });
};

// ---- Reports ----
export const getReportHtmlUrl = (id) => {
  const token = getStoredToken();
  return `${API_BASE}/reports/${id}/html${token ? `?token=${encodeURIComponent(token)}` : ''}`;
};
export const getReportJsonUrl = (id) => {
  const token = getStoredToken();
  return `${API_BASE}/reports/${id}/json${token ? `?token=${encodeURIComponent(token)}` : ''}`;
};
export const getCompareUrl = (ids) => {
  const token = getStoredToken();
  const tokenPart = token ? `&token=${encodeURIComponent(token)}` : '';
  return `${API_BASE}/reports/compare/csv?ids=${ids.join(',')}${tokenPart}`;
};
export const getModelDownloadUrl = (id) => {
  const token = getStoredToken();
  return `${API_BASE}/models/${id}/download${token ? `?token=${encodeURIComponent(token)}` : ''}`;
};

// ---- Health ----
export const checkHealth = () => api.get('/health');
export const getCapabilities = (refresh = false) =>
  api.get(`/settings/capabilities${refresh ? '?refresh=true' : ''}`);
export const saveKaggleCredentials = (credentials) => api.post('/settings/kaggle', credentials);
export const testKaggleConnection = () => api.post('/settings/kaggle/test');
export const searchKaggleDatasets = (query) => api.get('/kaggle/search', { params: { q: query } });
export const importKaggleDataset = (payload) => api.post('/kaggle/import', payload);
export const getSynthesisMethods = () => api.get('/datasets/synthesis/methods');
export const synthesizeDataset = (datasetId, payload) => api.post(`/datasets/${datasetId}/synthesize`, payload);
export const getSynthesisStatus = (jobId) => api.get(`/datasets/synthesis/${jobId}/status`);
export default api;
