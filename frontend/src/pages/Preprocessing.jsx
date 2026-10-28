import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { getDatasetProfile, suggestTask, getTargetDistribution, listDatasets } from '../services/api';
import { Settings2, AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import HelpTooltip from '../components/HelpTooltip';

const TASK_LABELS = {
  binary_classification: 'Binary Classification',
  multiclass_classification: 'Multiclass Classification',
  regression: 'Regression',
  time_series: 'Time-Series Forecasting',
};

export default function Preprocessing() {
  const { selectedDataset, setSelectedDataset, targetColumn, setTargetColumn, taskType, setTaskType, preprocessingConfig, setPreprocessingConfig } = useApp();
  const [datasets, setDatasets] = useState([]);
  const [profile, setProfile] = useState(null);
  const [targetSuggestion, setTargetSuggestion] = useState(null);
  const [targetDist, setTargetDist] = useState(null);
  const [loadingSuggest, setLoadingSuggest] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [error, setError] = useState(null);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [profileLoading, setProfileLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    listDatasets().then(r => setDatasets(r.data)).catch(e => setError(e.message)).finally(() => setDatasetsLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedDataset) return;
    setError(null);
    setProfile(null);
    setProfileLoading(true);
    getDatasetProfile(selectedDataset.id).then(r => setProfile(r.data)).catch(e => setError(e.message)).finally(() => setProfileLoading(false));
    setTargetColumn(null); setTaskType(null); setTargetSuggestion(null); setTargetDist(null); setConfirmed(false);
  }, [selectedDataset]);

  const handleTargetChange = async (col) => {
    setTargetColumn(col);
    setConfirmed(false);
    setTargetSuggestion(null);
    setTargetDist(null);
    setError(null);
    if (!col || !selectedDataset) return;
    setLoadingSuggest(true);
    try {
      const r = await suggestTask(selectedDataset.id, col);
      setTargetSuggestion(r.data);
      setTaskType(r.data.suggestion);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoadingSuggest(false);
    }
  };

  const handleConfirm = async () => {
    if (!targetColumn || !taskType) return;
    if (taskType === 'time_series' && !cfg.time_series_date_column) {
      setError('Choose a date column for time-series forecasting.');
      return;
    }
    setConfirming(true);
    try {
      const r = await getTargetDistribution(selectedDataset.id, targetColumn, taskType);
      setTargetDist(r.data);
      setConfirmed(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setConfirming(false);
    }
  };

  const cfg = preprocessingConfig;
  const setCfg = (key, val) => setPreprocessingConfig(prev => ({ ...prev, [key]: val }));

  if (!selectedDataset) {
    return (
      <div>
        <div className="flex items-center gap-2"><h1 className="page-title">Preprocessing</h1><HelpTooltip text="Choose the prediction target, review detected column types, and configure feature preparation." /></div>
        <div className="card">
          <h2 className="section-title">Select a Dataset First</h2>
          {datasetsLoading && <div className="skeleton h-12 rounded-lg" />}
          {error && <div className="error-box mb-3"><AlertCircle size={14} />{error}</div>}
          <div className="space-y-2">
            {datasets.map(d => (
              <div key={d.id} onClick={() => setSelectedDataset(d)}
                className="flex items-center justify-between p-3 rounded-lg border border-slate-200 hover:border-blue-300 cursor-pointer transition-all">
                <div><div className="font-medium text-sm">{d.name}</div><div className="text-xs text-slate-400">{d.row_count?.toLocaleString()} rows</div></div>
                <span className="badge badge-blue">Select</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2"><h1 className="page-title">Preprocessing Configuration</h1><HelpTooltip text="Choose the prediction target, review detected column types, and configure feature preparation." /></div>
      <p className="page-sub">Configure target selection, feature engineering, and data splits.</p>

      <div className="info-box mb-6">
        <Info size={14} className="shrink-0 mt-0.5" />
        <span>All preprocessing transformations (imputation, scaling, encoding) are fit <strong>only on training data</strong> and applied to the test set, preventing data leakage. Time-Series Forecasting uses chronological splits and lag features.</span>
      </div>
      {profileLoading && <div className="card mb-4"><div className="skeleton h-24 rounded-lg" /></div>}
      {error && !profileLoading && <div className="error-box mb-4"><AlertCircle size={14} />{error}</div>}

      {/* Dataset selector */}
      <div className="card mb-4">
        <label className="label">Dataset</label>
        <select className="select-field max-w-sm" value={selectedDataset.id}
          onChange={e => setSelectedDataset(datasets.find(d => d.id === e.target.value))}>
          {datasets.map(d => <option key={d.id} value={d.id}>{d.name} ({d.row_count?.toLocaleString()} rows)</option>)}
        </select>
      </div>

      {/* Target selection */}
      {profile && (
        <div className="card mb-4">
          <h2 className="section-title">1. Target Column Selection</h2>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <label className="label">Target Column</label>
              <select className="select-field" value={targetColumn || ''} onChange={e => handleTargetChange(e.target.value)}>
                <option value="">— Select target column —</option>
                {profile.columns?.map(c => <option key={c.name} value={c.name}>{c.name} ({c.inferred_type})</option>)}
              </select>
            </div>
            <div>
              <label className="label">Task Type</label>
              <select className="select-field" value={taskType || ''} onChange={e => setTaskType(e.target.value)} disabled={!targetColumn}>
                <option value="">— Auto-detected —</option>
                {Object.entries(TASK_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
              </select>
            </div>
          </div>

          {loadingSuggest && <div className="text-sm text-blue-600 mt-3">Analyzing target…</div>}

          {targetSuggestion && !loadingSuggest && (
            <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <div className="text-xs font-semibold text-blue-800 mb-1">Auto-suggestion: {TASK_LABELS[targetSuggestion.suggestion]}</div>
              <div className="text-xs text-blue-700">{targetSuggestion.rationale}</div>
              {targetSuggestion.warnings?.map((w, i) => (
                <div key={i} className="warning-box mt-2"><AlertTriangle size={12} />{w}</div>
              ))}
            </div>
          )}

          {targetColumn && taskType && !confirmed && (
            <button onClick={handleConfirm} disabled={confirming} className="btn-primary mt-4">
              <CheckCircle size={14} /> {confirming ? 'Confirming…' : 'Confirm Target & Task'}
            </button>
          )}
          {confirmed && (
            <div className="success-box mt-4">
              <CheckCircle size={14} />
              <span>Target confirmed: <strong>{targetColumn}</strong> for <strong>{TASK_LABELS[taskType]}</strong></span>
            </div>
          )}
          {taskType === 'time_series' && (
            <div className="mt-4 grid md:grid-cols-3 gap-3 rounded-lg border border-blue-100 bg-blue-50/50 p-3">
              <div>
                <label className="label">Date Column</label>
                <select className="select-field" value={cfg.time_series_date_column || ''}
                  onChange={e => setCfg('time_series_date_column', e.target.value)}>
                  <option value="">— Choose date column —</option>
                  {profile.columns?.map(column => <option key={column.name} value={column.name}>{column.name}</option>)}
                </select>
              </div>
              <div>
                <label className="label">Lags (comma-separated)</label>
                <input className="input-field" defaultValue={(cfg.time_series_lags || [1, 7, 14]).join(', ')}
                  onBlur={e => {
                    const lags = e.target.value.split(',').map(value => Number(value.trim())).filter(value => Number.isInteger(value) && value > 0);
                    setCfg('time_series_lags', lags);
                  }} />
              </div>
              <div>
                <label className="label">Rolling Windows</label>
                <input className="input-field" defaultValue={(cfg.time_series_rolling_windows || [3, 7]).join(', ')}
                  onBlur={e => {
                    const windows = e.target.value.split(',').map(value => Number(value.trim())).filter(value => Number.isInteger(value) && value > 1);
                    setCfg('time_series_rolling_windows', windows);
                  }} />
              </div>
              <p className="md:col-span-3 text-xs text-slate-500">
                Forecasting requires ascending dates. The split and cross-validation preserve chronology; rows missing lag features are removed.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Target distribution */}
      {targetDist && (
        <div className="card mb-4">
          <h2 className="section-title">Target Distribution</h2>
          {targetDist.type === 'categorical' ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={targetDist.counts?.slice(0, 15)}>
                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#3b82f6" radius={[4,4,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div>
              <div className="flex gap-6 text-sm mb-3">
                <span className="text-slate-500">Mean: <strong>{targetDist.mean}</strong></span>
                <span className="text-slate-500">Std: <strong>{targetDist.std}</strong></span>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={targetDist.histogram?.edges?.slice(0,-1).map((e, i) => ({
                  x: e.toFixed(2), count: targetDist.histogram.counts[i]
                }))}>
                  <XAxis dataKey="x" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
                  <YAxis tick={{ fontSize: 9 }} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[2,2,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Preprocessing options */}
      <div className="card mb-4">
        <h2 className="section-title">2. Feature Preprocessing</h2>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="label">Numeric Imputation</label>
            <select className="select-field" value={cfg.numeric_imputer} onChange={e => setCfg('numeric_imputer', e.target.value)}>
              <option value="median">Median (recommended)</option>
              <option value="mean">Mean</option>
            </select>
          </div>
          <div>
            <label className="label">Numeric Scaling</label>
            <select className="select-field" value={cfg.numeric_scaler} onChange={e => setCfg('numeric_scaler', e.target.value)}>
              <option value="standard">Standard Scaler (Z-score)</option>
              <option value="robust">Robust Scaler (IQR)</option>
              <option value="none">None</option>
            </select>
          </div>
          <div>
            <label className="label">Categorical Encoding</label>
            <select className="select-field" value={cfg.categorical_encoding} onChange={e => setCfg('categorical_encoding', e.target.value)}>
              <option value="onehot">One-Hot Encoding</option>
              <option value="ordinal">Ordinal Encoding</option>
            </select>
          </div>
          <div>
            <label className="label">Categorical Imputation</label>
            <select className="select-field" value={cfg.categorical_imputer} onChange={e => setCfg('categorical_imputer', e.target.value)}>
              <option value="most_frequent">Most Frequent</option>
              <option value="constant">Constant (missing)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Split config */}
      <div className="card mb-4">
        <h2 className="section-title">3. Train/Test Split</h2>
        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label className="label">Test Size</label>
            <select className="select-field" value={cfg.test_size} onChange={e => setCfg('test_size', parseFloat(e.target.value))}>
              <option value={0.1}>10%</option>
              <option value={0.15}>15%</option>
              <option value={0.2}>20% (default)</option>
              <option value={0.25}>25%</option>
              <option value={0.3}>30%</option>
            </select>
          </div>
          <div>
            <label className="label">Random Seed</label>
            <input type="number" className="input-field" value={cfg.random_seed}
              onChange={e => setCfg('random_seed', parseInt(e.target.value))} />
          </div>
          <div className="flex flex-col justify-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={cfg.drop_identifiers} onChange={e => setCfg('drop_identifiers', e.target.checked)} className="rounded" />
              <span className="text-sm text-slate-700">Drop likely identifier columns</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer mt-2">
              <input type="checkbox" checked={cfg.handle_constant} onChange={e => setCfg('handle_constant', e.target.checked)} className="rounded" />
              <span className="text-sm text-slate-700">Drop constant columns</span>
            </label>
          </div>
        </div>
      </div>

      {confirmed && (
        <div className="success-box">
          <CheckCircle size={14} />
          <span>Preprocessing configured. Proceed to <strong>Model Lab</strong> to train models.</span>
        </div>
      )}
    </div>
  );
}
