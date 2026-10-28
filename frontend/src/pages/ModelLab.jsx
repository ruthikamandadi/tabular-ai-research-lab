import React, { useState, useEffect, useRef } from 'react';
import { useApp } from '../contexts/AppContext';
import { getAvailableModels, createExperiment, getExperimentStatus, listExperiments } from '../services/api';
import { FlaskConical, Play, CheckCircle, AlertCircle, Clock, ChevronDown, ChevronUp, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useNavigate } from 'react-router-dom';
import HelpTooltip from '../components/HelpTooltip';

const TASK_LABELS = {
  binary_classification: 'Binary Classification',
  multiclass_classification: 'Multiclass Classification',
  regression: 'Regression',
  time_series: 'Time-Series Forecasting',
};

function ModelCard({ model, taskType, selected, onSelect, onConfigChange, config }) {
  const [expanded, setExpanded] = useState(false);
  const compatible = taskType ? model.tasks.includes(taskType) : true;

  return (
    <div className={`border rounded-xl p-4 transition-all ${selected ? 'border-blue-400 bg-blue-50/30 shadow-card-md' : compatible ? 'border-slate-200 hover:border-slate-300' : 'border-slate-100 opacity-50'}`}>
      <div className="flex items-start gap-3">
        <input type="checkbox" checked={selected} onChange={() => compatible && onSelect(model.id)}
          disabled={!compatible} className="mt-0.5 cursor-pointer" id={`model-${model.id}`} />
        <label htmlFor={`model-${model.id}`} className={`flex-1 ${compatible ? 'cursor-pointer' : 'cursor-not-allowed'}`}>
          <div className="font-semibold text-sm text-slate-800">{model.name}</div>
          {model.device && <span className="badge badge-gray mt-1">{model.device}</span>}
          <div className="text-xs text-slate-500 mt-0.5">{model.description}</div>
          <div className="flex flex-wrap gap-1 mt-2">
            {model.tasks.map(t => <span key={t} className={`badge text-[10px] ${taskType === t ? 'badge-blue' : 'badge-gray'}`}>{t.replace('_', ' ')}</span>)}
          </div>
        </label>
        {selected && (
          <button onClick={() => setExpanded(!expanded)} className="text-slate-400 hover:text-slate-600">
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        )}
      </div>

      {selected && expanded && model.params?.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-200 grid grid-cols-2 gap-3">
          {model.params.map(p => (
            <div key={p.name}>
              <label className="label">{p.label}</label>
              {p.type === 'list_int' ? (
                <input className="input-field" defaultValue={Array.isArray(config?.[p.name]) ? config[p.name].join(',') : (p.default || [128, 64]).join(',')}
                  onChange={e => {
                    const v = e.target.value.split(',').map(x => parseInt(x.trim())).filter(n => !isNaN(n));
                    onConfigChange(model.id, p.name, v);
                  }} placeholder="e.g. 128,64,32" />
              ) : p.type === 'string' ? (
                <select className="select-field" defaultValue={config?.[p.name] ?? p.default}
                  onChange={e => onConfigChange(model.id, p.name, e.target.value)}>
                  <option value="uniform">uniform</option>
                  <option value="distance">distance</option>
                </select>
              ) : p.type === 'bool' ? (
                <label className="flex items-center gap-2 mt-2 cursor-pointer">
                  <input type="checkbox" defaultChecked={config?.[p.name] ?? p.default ?? true}
                    onChange={e => onConfigChange(model.id, p.name, e.target.checked)} />
                  <span className="text-xs text-slate-600">Enabled</span>
                </label>
              ) : (
                <input className="input-field" type="number" step={p.type === 'float' ? 0.001 : 1}
                  min={p.min} max={p.max}
                  defaultValue={config?.[p.name] ?? p.default}
                  onChange={e => onConfigChange(model.id, p.name, p.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TrainingStatus({ job, experimentId }) {
  const navigate = useNavigate();
  if (!job) return null;
  const {
    status = 'pending',
    progress = 0,
    message,
    metrics,
    device,
    training_history: history = [],
    model_name,
    error: jobError,
    elapsed_seconds,
    warning,
  } = job;
  const latestEpoch = history.length ? history[history.length - 1] : null;

  return (
    <div
      className={`card mt-4 ${
        status === 'completed'
          ? 'border-green-200 bg-green-50/20'
          : status === 'failed'
          ? 'border-red-200 bg-red-50/20'
          : status === 'cancelled'
          ? 'border-amber-200 bg-amber-50/20'
          : 'border-blue-200 bg-blue-50/20'
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {status === 'completed' ? (
            <CheckCircle size={18} className="text-green-500 shrink-0" />
          ) : status === 'failed' ? (
            <AlertCircle size={18} className="text-red-500 shrink-0" />
          ) : (
            <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin shrink-0" />
          )}
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-slate-800">
                {model_name || `Experiment (${String(experimentId).slice(0, 8)})`}
              </span>
              <span
                className={`badge ${
                  status === 'completed'
                    ? 'badge-green'
                    : status === 'failed'
                    ? 'badge-red'
                    : status === 'cancelled'
                    ? 'badge-yellow'
                    : 'badge-blue'
                } text-[11px] capitalize`}
              >
                {status}
              </span>
            </div>
            {message && <div className="text-xs text-slate-500 mt-0.5">{message}</div>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {elapsed_seconds != null && (
            <span className="text-xs text-slate-400 font-mono">
              {Number(elapsed_seconds).toFixed(1)}s elapsed
            </span>
          )}
          {device && <span className="badge badge-gray">{String(device).toUpperCase()}</span>}
        </div>
      </div>

      {(status === 'running' || status === 'pending') && (
        <>
          <div className="progress-bar mb-2">
            <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
          </div>
          <div className="flex justify-between text-xs text-slate-500 mb-2">
            <span>
              {latestEpoch
                ? `Epoch ${latestEpoch.epoch}: train loss ${Number(latestEpoch.train_loss).toFixed(4)}${
                    latestEpoch.val_loss != null
                      ? ` · validation loss ${Number(latestEpoch.val_loss).toFixed(4)}`
                      : latestEpoch.val_score != null
                      ? ` · validation score ${Number(latestEpoch.val_score).toFixed(4)}`
                      : ''
                  }`
                : status === 'pending'
                ? 'Queued, waiting for execution...'
                : message || `${Math.round(progress)}% completed`}
            </span>
            <span>{Math.round(progress)}%</span>
          </div>
        </>
      )}

      {warning && (
        <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {warning}
        </div>
      )}

      {status === 'completed' && metrics && (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2 mt-3">
          {Object.entries(metrics)
            .filter(([k]) => !['confusion_matrix', 'class_names'].includes(k))
            .map(([k, v]) => (
              <div key={k} className="text-center p-2 bg-white rounded-lg border border-green-100 shadow-sm">
                <div className="text-xs text-slate-500 font-medium">{k.toUpperCase()}</div>
                <div className="text-sm font-bold text-slate-800">
                  {typeof v === 'number' ? v.toFixed(4) : String(v)}
                </div>
              </div>
            ))}
        </div>
      )}

      {status === 'completed' && (
        <button
          onClick={() => navigate(`/experiments?id=${experimentId}`, { state: { highlightId: experimentId } })}
          className="btn-primary mt-3 text-xs"
        >
          <Zap size={14} /> View in Experiments
        </button>
      )}

      {status === 'failed' && (
        <div className="error-box mt-3 text-xs">
          <AlertCircle size={14} className="shrink-0" />
          <div>
            <div className="font-semibold">Training Failed</div>
            <div className="text-slate-700 mt-0.5">{jobError || message || 'An unexpected error occurred during training.'}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ModelLab() {
  const { selectedDataset, targetColumn, taskType, preprocessingConfig } = useApp();
  const [models, setModels] = useState([]);
  const [selectedModels, setSelectedModels] = useState([]);
  const [modelConfigs, setModelConfigs] = useState({});
  const [jobs, setJobs] = useState({});
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const pollingRef = useRef({});
  const inFlightRef = useRef(new Set());
  const retryCountsRef = useRef({});
  const unmountedRef = useRef(false);
  const navigate = useNavigate();

  useEffect(() => {
    unmountedRef.current = false;
    getAvailableModels()
      .then(r => setModels(r.data))
      .catch(e => setError(`Could not load available models: ${e.message}`))
      .finally(() => setModelsLoading(false));

    return () => {
      unmountedRef.current = true;
      Object.values(pollingRef.current).forEach(clearInterval);
      pollingRef.current = {};
      inFlightRef.current.clear();
    };
  }, []);

  const toggleModel = (id) => {
    setSelectedModels(prev => prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]);
  };

  const updateConfig = (modelId, key, val) => {
    setModelConfigs(prev => ({
      ...prev,
      [modelId]: { ...(prev[modelId] || {}), [key]: val },
    }));
  };

  const pollStatus = (expId) => {
    if (pollingRef.current[expId]) {
      clearInterval(pollingRef.current[expId]);
      delete pollingRef.current[expId];
    }

    const checkStatus = async () => {
      if (unmountedRef.current) return;
      if (inFlightRef.current.has(expId)) return; // Prevent overlapping requests

      inFlightRef.current.add(expId);
      try {
        const r = await getExperimentStatus(expId);
        if (unmountedRef.current) return;

        const job = r.data;
        setJobs(prev => ({ ...prev, [expId]: job }));

        // Clear consecutive retry count on success
        retryCountsRef.current[expId] = 0;

        // Clear status refresh error if polling succeeded
        setError(prev => (prev && prev.includes('Could not refresh training status') ? null : prev));

        if (['completed', 'failed', 'cancelled'].includes(job.status)) {
          if (pollingRef.current[expId]) {
            clearInterval(pollingRef.current[expId]);
            delete pollingRef.current[expId];
          }
        }
      } catch (err) {
        if (unmountedRef.current) return;

        const retries = (retryCountsRef.current[expId] || 0) + 1;
        retryCountsRef.current[expId] = retries;

        const maxRetries = 5;
        if (retries >= maxRetries) {
          if (pollingRef.current[expId]) {
            clearInterval(pollingRef.current[expId]);
            delete pollingRef.current[expId];
          }
          const detail = err.response?.data?.detail || err.message;
          setError(`Could not refresh training status: ${detail}`);
        }
      } finally {
        inFlightRef.current.delete(expId);
      }
    };

    // Immediate first check, then recurring interval
    checkStatus();
    pollingRef.current[expId] = setInterval(checkStatus, 1500);
  };

  const handleTrain = async () => {
    if (launching) return;
    if (!selectedDataset) { setError('No dataset selected. Go to Datasets first.'); return; }
    if (!targetColumn) { setError('No target column set. Go to Preprocessing first.'); return; }
    if (!taskType) { setError('No task type confirmed. Go to Preprocessing first.'); return; }
    if (selectedModels.length === 0) { setError('Select at least one model.'); return; }

    // Check if any selected model is already training or queued
    const activeJobs = Object.values(jobs).filter(j => ['running', 'pending'].includes(j.status));
    const activeModelIds = activeJobs.map(j => j.model_id);
    const modelsToLaunch = selectedModels.filter(mId => !activeModelIds.includes(mId));

    if (modelsToLaunch.length === 0) {
      setError('All selected models are already actively training or queued.');
      return;
    }

    setError(null);
    setLaunching(true);

    try {
      for (const modelId of modelsToLaunch) {
        try {
          const r = await createExperiment({
            dataset_id: selectedDataset.id,
            target_column: targetColumn,
            task_type: taskType,
            model_id: modelId,
            model_config: modelConfigs[modelId] || {},
            preprocessing_config: preprocessingConfig,
            random_seed: preprocessingConfig.random_seed || 42,
          });
          const expId = r.data.experiment_id;
          const modelObj = models.find(m => m.id === modelId);
          setJobs(prev => ({
            ...prev,
            [expId]: {
              experiment_id: expId,
              model_id: modelId,
              model_name: modelObj?.name || modelId,
              status: 'pending',
              progress: 0,
              message: 'Queued…',
            },
          }));
          pollStatus(expId);
        } catch (e) {
          const detail = e.response?.data?.detail || e.message;
          setError(`Failed to start ${modelId}: ${detail}`);
        }
      }
    } finally {
      setLaunching(false);
    }
  };

  const ready = selectedDataset && targetColumn && taskType;

  return (
    <div>
      <div className="flex items-center gap-2"><h1 className="page-title">Model Lab</h1><HelpTooltip text="Select models and configure training and tuning options for the chosen dataset." /></div>
      <p className="page-sub">Select and configure machine learning models, then launch training experiments.</p>

      {!ready && (
        <div className="warning-box mb-6">
          <AlertCircle size={14} className="shrink-0" />
          <span>
            {!selectedDataset ? 'No dataset selected. ' : ''}
            {!targetColumn ? 'No target column. ' : ''}
            {!taskType ? 'Task type not confirmed. ' : ''}
            Complete <strong>Preprocessing</strong> setup first.
          </span>
        </div>
      )}

      {ready && (
        <div className="card mb-6 bg-slate-50 border-slate-200">
          <div className="flex flex-wrap gap-6 text-sm">
            <div><span className="text-slate-500">Dataset:</span> <strong>{selectedDataset.name}</strong></div>
            <div><span className="text-slate-500">Target:</span> <strong>{targetColumn}</strong></div>
            <div><span className="text-slate-500">Task:</span> <strong>{TASK_LABELS[taskType]}</strong></div>
            <div><span className="text-slate-500">Test Size:</span> <strong>{(preprocessingConfig.test_size * 100).toFixed(0)}%</strong></div>
          </div>
        </div>
      )}

      {/* Model selection */}
      <div className="card mb-6">
        <h2 className="section-title">Available Models ({models.length})</h2>
        {modelsLoading && <div className="grid md:grid-cols-2 gap-3">{[...Array(2)].map((_, index) => <div key={index} className="skeleton h-28 rounded-xl" />)}</div>}
        {!modelsLoading && models.length === 0 && <div className="text-sm text-slate-500">No models are available. Check the backend configuration and retry.</div>}
        <div className="grid md:grid-cols-2 gap-3">
          {models.map(m => (
            <ModelCard key={m.id} model={m} taskType={taskType}
              selected={selectedModels.includes(m.id)}
              onSelect={toggleModel}
              config={modelConfigs[m.id]}
              onConfigChange={updateConfig}
            />
          ))}
        </div>
      </div>

      {error && <div className="error-box mb-4"><AlertCircle size={14} />{error}</div>}

      <button onClick={handleTrain} disabled={!ready || launching || selectedModels.length === 0}
        className="btn-primary text-base px-6 py-3">
        {launching ? (
          <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Launching…</>
        ) : (
          <><Play size={16} /> Train {selectedModels.length} Model{selectedModels.length !== 1 ? 's' : ''}</>
        )}
      </button>

      {/* Job statuses */}
      {Object.entries(jobs).map(([expId, job]) => (
        <TrainingStatus key={expId} job={job} experimentId={expId} />
      ))}
    </div>
  );
}
