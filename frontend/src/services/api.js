import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
  timeout: 120000,
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
      window.dispatchEvent(new CustomEvent('auth:unauthorized'));
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
export const getDatasetDownloadUrl = (id) => `/api/datasets/${id}/download`;
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
export const getReportHtmlUrl = (id) => `/api/reports/${id}/html`;
export const getReportJsonUrl = (id) => `/api/reports/${id}/json`;
export const getCompareUrl = (ids) => `/api/reports/compare/csv?ids=${ids.join(',')}`;

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
