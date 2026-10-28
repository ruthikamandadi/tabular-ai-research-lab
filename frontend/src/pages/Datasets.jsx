import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Database, Trash2, CheckCircle, AlertCircle, FileText, Clock, RefreshCw, Search, Sparkles, Download, Edit2 } from 'lucide-react';
import {
  uploadDataset, listDatasets, deleteDataset, loadSampleDataset, listSampleDatasets,
  searchKaggleDatasets, importKaggleDataset, getSynthesisMethods, synthesizeDataset,
  getSynthesisStatus, renameDataset, getDatasetDownloadUrl,
} from '../services/api';
import { useApp } from '../contexts/AppContext';
import { useNavigate } from 'react-router-dom';
import HelpTooltip from '../components/HelpTooltip';

function formatSize(bytes) {
  if (!bytes) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

export default function Datasets() {
  const navigate = useNavigate();
  const { setSelectedDataset } = useApp();
  const [datasets, setDatasets] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadSuccess, setUploadSuccess] = useState('');
  const [uploadError, setUploadError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [datasetError, setDatasetError] = useState('');
  const [samples, setSamples] = useState([]);
  const [loadingSample, setLoadingSample] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [kaggleQuery, setKaggleQuery] = useState('');
  const [kaggleResults, setKaggleResults] = useState([]);
  const [kaggleSearchError, setKaggleSearchError] = useState('');
  const [kaggleBusy, setKaggleBusy] = useState(false);
  const [selectedKaggle, setSelectedKaggle] = useState(null);
  const [selectedCsv, setSelectedCsv] = useState('');
  const [synthesisMethods, setSynthesisMethods] = useState([{ method: 'bootstrap', available: true }]);
  const [maxSyntheticRows, setMaxSyntheticRows] = useState(100000);
  const [synthesisSourceId, setSynthesisSourceId] = useState('');
  const [synthesisMethod, setSynthesisMethod] = useState('bootstrap');
  const [synthesisRows, setSynthesisRows] = useState(1000);
  const [synthesisEpochs, setSynthesisEpochs] = useState(100);
  const [synthesisTarget, setSynthesisTarget] = useState('');
  const [synthesisJob, setSynthesisJob] = useState(null);
  const [synthesisError, setSynthesisError] = useState('');
  const [registrySearch, setRegistrySearch] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const PAGE_SIZE = 10;

  const refresh = useCallback(() => {
    setLoading(true);
    setDatasetError('');
    listDatasets().then(r => setDatasets(r.data)).catch(error => setDatasetError(error.message)).finally(() => setLoading(false));
  }, []);

  const handleRename = async (ds) => {
    const newName = window.prompt('Enter new dataset name:', ds.name);
    if (!newName || !newName.trim() || newName.trim() === ds.name) return;
    try {
      await renameDataset(ds.id, newName.trim());
      await refresh();
    } catch (e) {
      alert('Rename failed: ' + e.message);
    }
  };

  const filteredDatasets = useMemo(() => {
    if (!registrySearch.trim()) return datasets;
    const q = registrySearch.toLowerCase().trim();
    return datasets.filter(d => d.name?.toLowerCase().includes(q));
  }, [datasets, registrySearch]);

  const totalPages = Math.max(1, Math.ceil(filteredDatasets.length / PAGE_SIZE));
  const paginatedDatasets = useMemo(() => {
    const start = (currentPage - 1) * PAGE_SIZE;
    return filteredDatasets.slice(start, start + PAGE_SIZE);
  }, [filteredDatasets, currentPage]);

  useEffect(() => {
    refresh();
    listSampleDatasets().then(r => setSamples(r.data)).catch(error => setUploadError(`Could not load sample datasets: ${error.message}`));
    getSynthesisMethods().then(r => {
      setSynthesisMethods(r.data.methods);
      setMaxSyntheticRows(r.data.max_rows);
    }).catch(error => setSynthesisError(`Could not load synthesis methods: ${error.message}`));
  }, [refresh]);

  useEffect(() => {
    if (!synthesisJob || !['pending', 'running'].includes(synthesisJob.status)) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const response = await getSynthesisStatus(synthesisJob.job_id);
        setSynthesisJob(response.data);
        if (response.data.status === 'completed') {
          await refresh();
        }
      } catch (error) {
        setSynthesisError(error.message);
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [synthesisJob, refresh]);

  const onDrop = useCallback(async (accepted, rejected) => {
    setUploadError(null);
    setUploadSuccess('');
    setUploadProgress(0);
    if (rejected.length > 0) {
      setUploadError('Invalid file. Only CSV files are accepted (max 200MB).');
      return;
    }
    if (accepted.length === 0) return;
    const file = accepted[0];
    if (file.size > 200 * 1024 * 1024) {
      setUploadError('File exceeds 200MB limit.');
      return;
    }
    setUploading(true);
    try {
      const r = await uploadDataset(file, (progress) => {
        setUploadProgress(progress);
      });
      setSelectedDataset(r.data);
      setUploadSuccess(`Dataset "${file.name}" uploaded successfully!`);
      await refresh();
    } catch (e) {
      setUploadError(e.message);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [refresh, setSelectedDataset]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxSize: 200 * 1024 * 1024,
    multiple: false,
    disabled: uploading,
  });

  const handleDelete = async (id) => {
    if (!confirm('Delete this dataset and all its data?')) return;
    try {
      await deleteDataset(id);
      await refresh();
    } catch (e) {
      alert('Delete failed: ' + e.message);
    }
  };

  const handleSelect = (ds) => {
    setSelectedDataset(ds);
    navigate('/explorer');
  };

  const handleLoadSample = async (filename) => {
    setLoadingSample(filename);
    setUploadError(null);
    try {
      const r = await loadSampleDataset(filename);
      setSelectedDataset(r.data);
      await refresh();
    } catch (e) {
      setUploadError(e.message);
    } finally {
      setLoadingSample(null);
    }
  };

  const handleKaggleSearch = async (event) => {
    event.preventDefault();
    setKaggleSearchError('');
    setKaggleResults([]);
    setSelectedKaggle(null);
    if (!kaggleQuery.trim()) {
      setKaggleSearchError('Enter a search term.');
      return;
    }
    setKaggleBusy(true);
    try {
      const response = await searchKaggleDatasets(kaggleQuery.trim());
      setKaggleResults(response.data);
      if (!response.data.length) setKaggleSearchError('No datasets found. Try another search.');
    } catch (error) {
      setKaggleSearchError(error.message);
    } finally {
      setKaggleBusy(false);
    }
  };

  const finishKaggleImport = async (payload) => {
    setKaggleBusy(true);
    setKaggleSearchError('');
    try {
      const response = await importKaggleDataset(payload);
      if (response.data.selection_required) {
        setSelectedKaggle({ ref: payload.ref, csvFiles: response.data.csv_files });
        setSelectedCsv(response.data.csv_files[0] || '');
        return;
      }
      setSelectedDataset(response.data.dataset);
      setSelectedKaggle(null);
      setSelectedCsv('');
      await refresh();
    } catch (error) {
      setKaggleSearchError(error.message);
    } finally {
      setKaggleBusy(false);
    }
  };

  const handleSynthesize = async (event) => {
    event.preventDefault();
    setSynthesisError('');
    setSynthesisJob(null);
    if (!synthesisSourceId) {
      setSynthesisError('Choose a source dataset.');
      return;
    }
    try {
      const payload = {
        method: synthesisMethod,
        num_rows: Number(synthesisRows),
        epochs: Number(synthesisEpochs),
        target_column: synthesisMethod === 'smote' ? synthesisTarget : null,
      };
      const response = await synthesizeDataset(synthesisSourceId, payload);
      setSynthesisJob(response.data);
    } catch (error) {
      setSynthesisError(error.message);
    }
  };

  const sourceDataset = datasets.find(dataset => dataset.id === synthesisSourceId);
  const activeSynthesisMethod = synthesisMethods.find(item => item.method === synthesisMethod);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2"><h1 className="page-title">Datasets</h1><HelpTooltip text="Upload, browse, import, and synthesize datasets for analysis." /></div>
          <p className="page-sub">Upload, import, or generate tabular datasets to begin analysis.</p>
        </div>
        <button onClick={refresh} className="btn-secondary">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>
      {datasetError && (
        <div className="error-box mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle size={14} className="shrink-0" />
            <span>{datasetError}</span>
          </div>
          {(datasetError.toLowerCase().includes('auth') || datasetError.toLowerCase().includes('session')) && (
            <button onClick={() => navigate('/login')} className="btn-secondary text-xs px-2.5 py-1">
              Sign In
            </button>
          )}
        </div>
      )}

      <div className="mb-5 flex flex-wrap gap-2 border-b border-slate-200">
        {[
          ['upload', 'Upload / Samples'],
          ['kaggle', 'Import from Kaggle'],
          ['synthetic', 'Synthetic Data'],
        ].map(([tab, label]) => (
          <button key={tab} onClick={() => setActiveTab(tab)}
            className={`border-b-2 px-4 py-2 text-sm font-medium ${activeTab === tab ? 'border-blue-600 text-blue-700' : 'border-transparent text-slate-500 hover:text-slate-700'}`}>
            {label}
          </button>
        ))}
      </div>

      {activeTab === 'kaggle' && (
        <div className="card mb-6">
          <h2 className="section-title flex items-center gap-2"><Search size={16} className="text-blue-500" />Import from Kaggle</h2>
          <p className="mb-4 text-sm text-slate-500">Search public datasets. Configure your Kaggle credentials in Settings before importing.</p>
          <form onSubmit={handleKaggleSearch} className="mb-4 flex gap-2">
            <input className="input-field flex-1" value={kaggleQuery} onChange={event => setKaggleQuery(event.target.value)} placeholder="Search datasets" />
            <button className="btn-primary" disabled={kaggleBusy}><Search size={14} />{kaggleBusy ? 'Searching…' : 'Search'}</button>
          </form>
          {kaggleSearchError && <div className="error-box mb-3"><AlertCircle size={14} />{kaggleSearchError}</div>}
          {kaggleResults.length > 0 && (
            <div className="space-y-3">
              {kaggleResults.map(result => (
                <div key={result.ref} className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 p-3">
                  <div>
                    <div className="font-medium text-slate-800">{result.title}</div>
                    <div className="text-xs text-slate-500">{result.ref} · {formatSize(result.size)} · {(result.download_count || 0).toLocaleString()} downloads</div>
                  </div>
                  <button onClick={() => finishKaggleImport({ ref: result.ref })} disabled={kaggleBusy} className="btn-secondary">
                    <Download size={14} />{kaggleBusy ? 'Importing…' : 'Import'}
                  </button>
                </div>
              ))}
            </div>
          )}
          {selectedKaggle && (
            <div className="mt-4 rounded-lg border border-blue-100 bg-blue-50 p-4">
              <label className="label">Choose a CSV file from {selectedKaggle.ref}</label>
              <div className="flex flex-wrap gap-2">
                <select className="input-field min-w-0 flex-1" value={selectedCsv} onChange={event => setSelectedCsv(event.target.value)}>
                  {selectedKaggle.csvFiles.map(name => <option key={name} value={name}>{name}</option>)}
                </select>
                <button onClick={() => finishKaggleImport({ ref: selectedKaggle.ref, csv_filename: selectedCsv })} disabled={kaggleBusy || !selectedCsv} className="btn-primary">
                  {kaggleBusy ? 'Importing…' : 'Import selected CSV'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'synthetic' && (
        <div className="card mb-6">
          <h2 className="section-title flex items-center gap-2"><Sparkles size={16} className="text-purple-500" />Synthetic Data</h2>
          <p className="mb-4 text-sm text-slate-500">Generate a new dataset using bootstrap, SMOTE, Gaussian Copula, or CTGAN.</p>
          <form onSubmit={handleSynthesize} className="grid gap-4 md:grid-cols-2">
            <div>
              <label className="label">Source dataset</label>
              <select className="input-field" value={synthesisSourceId} onChange={event => setSynthesisSourceId(event.target.value)}>
                <option value="">Choose a dataset</option>
                {datasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}
              </select>
            </div>
            <div>
              <label className="label">Generation method</label>
              <select className="input-field" value={synthesisMethod} onChange={event => setSynthesisMethod(event.target.value)}>
                {synthesisMethods.map(method => (
                  <option key={method.method} value={method.method} disabled={!method.available}
                    title={!method.available ? `Install ${method.install} to enable` : undefined}>
                    {method.method === 'gaussian_copula' ? 'Gaussian Copula' : method.method.toUpperCase()}
                    {!method.available ? ` (Install ${method.install} to enable)` : ''}
                  </option>
                ))}
              </select>
              {!activeSynthesisMethod?.available && <p className="mt-1 text-xs text-amber-700">Install {activeSynthesisMethod?.install} to enable this method.</p>}
            </div>
            <div>
              <label className="label">Number of rows</label>
              <input className="input-field" type="number" min="1" max={maxSyntheticRows} value={synthesisRows} onChange={event => setSynthesisRows(event.target.value)} />
            </div>
            {synthesisMethod === 'ctgan' && (
              <div>
                <label className="label">Training epochs</label>
                <input className="input-field" type="number" min="1" max="1000" value={synthesisEpochs} onChange={event => setSynthesisEpochs(event.target.value)} />
              </div>
            )}
            {synthesisMethod === 'smote' && sourceDataset && (
              <div>
                <label className="label">Class target</label>
                <select className="input-field" value={synthesisTarget} onChange={event => setSynthesisTarget(event.target.value)}>
                  <option value="">Choose target column</option>
                  {(sourceDataset.columns || []).map(column => <option key={column} value={column}>{column}</option>)}
                </select>
              </div>
            )}
            <div className="md:col-span-2">
              <button className="btn-primary" disabled={!sourceDataset || !activeSynthesisMethod?.available || (synthesisMethod === 'smote' && !synthesisTarget) || !!synthesisJob && ['pending', 'running'].includes(synthesisJob.status)}>
                <Sparkles size={14} />Generate synthetic dataset
              </button>
            </div>
          </form>
          {synthesisError && <div className="error-box mt-4"><AlertCircle size={14} />{synthesisError}</div>}
          {synthesisJob && (
            <div className="mt-5 rounded-lg border border-slate-200 p-4">
              <div className="flex justify-between text-sm">
                <span className="font-medium capitalize">{synthesisJob.status}: {synthesisJob.message}</span>
                <span>{synthesisJob.progress || 0}%</span>
              </div>
              <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full bg-blue-600 transition-all" style={{ width: `${synthesisJob.progress || 0}%` }} />
              </div>
              {synthesisJob.error && <p className="mt-3 text-sm text-red-600">{synthesisJob.error}</p>}
              {synthesisJob.quality_report && (
                <div className="mt-4">
                  <h3 className="mb-2 text-sm font-semibold">Quality report</h3>
                  <div className="mb-3 grid gap-3 sm:grid-cols-3">
                    <div className="rounded bg-slate-50 p-3 text-sm">Mean correlation difference <strong className="block">{synthesisJob.quality_report.mean_absolute_correlation_difference}</strong></div>
                    <div className="rounded bg-slate-50 p-3 text-sm">Numeric columns <strong className="block">{Object.keys(synthesisJob.quality_report.numeric_columns || {}).length}</strong></div>
                    <div className="rounded bg-slate-50 p-3 text-sm">Categorical columns <strong className="block">{Object.keys(synthesisJob.quality_report.categorical_columns || {}).length}</strong></div>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead><tr><th className="table-th">Column</th><th className="table-th">Metric</th><th className="table-th">Distance</th><th className="table-th">Similarity</th></tr></thead>
                      <tbody>
                        {Object.entries(synthesisJob.quality_report.numeric_columns || {}).map(([column, metrics]) => <tr key={column}><td className="table-td">{column}</td><td className="table-td">KS statistic</td><td className="table-td">{metrics.ks_statistic}</td><td className="table-td">{metrics.similarity}</td></tr>)}
                        {Object.entries(synthesisJob.quality_report.categorical_columns || {}).map(([column, metrics]) => <tr key={column}><td className="table-td">{column}</td><td className="table-td">Total variation</td><td className="table-td">{metrics.total_variation_distance}</td><td className="table-td">{metrics.similarity}</td></tr>)}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-5 grid gap-4 lg:grid-cols-2">
                    {Object.entries({
                      ...(synthesisJob.quality_report.numeric_columns || {}),
                      ...(synthesisJob.quality_report.categorical_columns || {}),
                    }).map(([column, metrics]) => metrics.distribution && (
                      <div key={column} className="rounded-lg border border-slate-100 p-3">
                        <div className="mb-2 text-xs font-semibold text-slate-700">{column}: real vs synthetic</div>
                        <div className="space-y-1.5">
                          {metrics.distribution.labels.map((label, index) => (
                            <div key={`${column}-${label}`} className="grid grid-cols-[minmax(3.5rem,auto)_1fr] items-center gap-2 text-[10px] text-slate-500">
                              <span className="truncate" title={label}>{label}</span>
                              <div className="space-y-0.5">
                                <div className="h-1.5 rounded bg-blue-500" style={{ width: `${Math.max(1, metrics.distribution.real[index] * 100)}%` }} />
                                <div className="h-1.5 rounded bg-purple-500" style={{ width: `${Math.max(1, metrics.distribution.synthetic[index] * 100)}%` }} />
                              </div>
                            </div>
                          ))}
                        </div>
                        <div className="mt-2 flex gap-3 text-[10px] text-slate-400"><span className="text-blue-600">Real</span><span className="text-purple-600">Synthetic</span></div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Upload zone */}
      {activeTab === 'upload' && <div className="card mb-6">
        <h2 className="section-title">Upload CSV Dataset</h2>
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all
            ${isDragActive ? 'border-blue-400 bg-blue-50' : 'border-slate-300 hover:border-blue-300 hover:bg-slate-50'}
            ${uploading ? 'opacity-60 cursor-not-allowed' : ''}`}
        >
          <input {...getInputProps()} />
          <Upload size={32} className={`mx-auto mb-3 ${isDragActive ? 'text-blue-500' : 'text-slate-400'}`} />
          {uploading ? (
            <div className="text-sm text-blue-600 font-medium">Uploading and parsing…</div>
          ) : isDragActive ? (
            <div className="text-sm text-blue-600 font-medium">Drop the CSV file here</div>
          ) : (
            <>
              <div className="text-sm font-medium text-slate-700 mb-1">Drag & drop a CSV file, or click to browse</div>
              <div className="text-xs text-slate-400">UTF-8 or common encodings • Up to 200MB • CSV only</div>
            </>
          )}
        </div>
        {uploading && (
          <div className="mt-4">
            <div className="flex justify-between text-xs text-blue-600 font-medium mb-1.5">
              <span>Uploading & parsing dataset...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-600 rounded-full transition-all duration-300"
                style={{ width: `${Math.max(uploadProgress, 5)}%` }}
              />
            </div>
          </div>
        )}
        {uploadSuccess && (
          <div className="flex items-center gap-2 p-3 mt-3 bg-green-50 border border-green-200 rounded-lg text-green-700 text-sm">
            <CheckCircle size={15} className="shrink-0" />
            <span>{uploadSuccess}</span>
          </div>
        )}
        {uploadError && (
          <div className="error-box mt-3">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <span>{uploadError}</span>
          </div>
        )}
      </div>}

      {/* Sample datasets */}
      {activeTab === 'upload' && samples.length > 0 && (
        <div className="card mb-6">
          <h2 className="section-title">Built-in Sample Datasets</h2>
          <p className="text-sm text-slate-500 mb-4">Load a pre-built dataset to explore the full pipeline immediately.</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {samples.map(s => (
              <div key={s.filename} className="border border-slate-200 rounded-lg p-3 hover:border-blue-300 transition-colors">
                <div className="font-medium text-sm text-slate-800 mb-0.5">{s.name}</div>
                <div className="text-xs text-slate-400 mb-2">{s.columns?.length} cols · {formatSize(s.size_bytes)}</div>
                <button onClick={() => handleLoadSample(s.filename)} disabled={!!loadingSample}
                  className="btn-primary text-xs px-2.5 py-1 w-full justify-center">
                  {loadingSample === s.filename ? 'Loading…' : 'Load'}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dataset list */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
          <h2 className="section-title mb-0">Dataset Registry ({datasets.length})</h2>
          {datasets.length > 0 && (
            <div className="relative w-full sm:w-64">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                className="input-field pl-9 py-1.5 text-xs w-full"
                placeholder="Search datasets..."
                value={registrySearch}
                onChange={e => {
                  setRegistrySearch(e.target.value);
                  setCurrentPage(1);
                }}
              />
            </div>
          )}
        </div>

        {loading ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="skeleton h-14 rounded-lg" />
            ))}
          </div>
        ) : filteredDatasets.length === 0 ? (
          <div className="text-center py-12 text-slate-400">
            <Database size={44} className="mx-auto mb-3 opacity-30 text-blue-500" />
            <div className="text-base font-medium text-slate-700 mb-1">
              {datasets.length === 0 ? 'No datasets in your registry' : 'No matching datasets found'}
            </div>
            <div className="text-xs text-slate-400 max-w-sm mx-auto">
              {datasets.length === 0
                ? 'Upload a CSV file (up to 200MB) above or load a sample dataset to start analyzing and training models.'
                : 'Try adjusting your search query to find the dataset you are looking for.'}
            </div>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="table-th">Name</th>
                    <th className="table-th">Rows</th>
                    <th className="table-th">Columns</th>
                    <th className="table-th">Size</th>
                    <th className="table-th">Status</th>
                    <th className="table-th">Uploaded</th>
                    <th className="table-th text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {paginatedDatasets.map(ds => (
                    <tr key={ds.id} className="hover:bg-slate-50 cursor-pointer" onClick={() => handleSelect(ds)}>
                      <td className="table-td font-medium text-blue-600">
                        <div className="flex items-center gap-1.5">
                          <span>{ds.name}</span>
                          {ds.is_synthetic && <span className="badge badge-purple text-[10px]">Synthetic</span>}
                        </div>
                      </td>
                      <td className="table-td">{ds.row_count?.toLocaleString()}</td>
                      <td className="table-td">{ds.col_count}</td>
                      <td className="table-td">{formatSize(ds.size_bytes)}</td>
                      <td className="table-td">
                        <span className={`badge ${ds.status === 'ready' ? 'badge-green' : 'badge-red'}`}>
                          {ds.status}
                        </span>
                      </td>
                      <td className="table-td text-slate-400">{formatDate(ds.created_at)}</td>
                      <td className="table-td text-right" onClick={e => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1">
                          <a
                            href={getDatasetDownloadUrl(ds.id)}
                            download={ds.name}
                            className="text-slate-400 hover:text-blue-600 transition-colors p-1"
                            title="Download dataset CSV"
                          >
                            <Download size={14} />
                          </a>
                          <button
                            onClick={() => handleRename(ds)}
                            className="text-slate-400 hover:text-slate-700 transition-colors p-1"
                            title="Rename dataset"
                          >
                            <Edit2 size={14} />
                          </button>
                          <button
                            onClick={() => handleDelete(ds.id)}
                            className="text-slate-400 hover:text-red-600 transition-colors p-1"
                            title="Delete dataset"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <div className="flex items-center justify-between border-t border-slate-100 pt-3 mt-3 text-xs text-slate-500">
                <span>
                  Showing {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filteredDatasets.length)} of {filteredDatasets.length} datasets
                </span>
                <div className="flex items-center gap-1">
                  <button
                    disabled={currentPage === 1}
                    onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                    className="btn-secondary px-2.5 py-1 text-xs disabled:opacity-40"
                  >
                    Previous
                  </button>
                  <span className="px-2">Page {currentPage} of {totalPages}</span>
                  <button
                    disabled={currentPage === totalPages}
                    onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                    className="btn-secondary px-2.5 py-1 text-xs disabled:opacity-40"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
