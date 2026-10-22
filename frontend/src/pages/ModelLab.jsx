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
  const { status, progress, message, metrics, device, training_history: history = [] } = job;
  const latestEpoch = history.length ? history[history.length - 1] : null;

  return (
    <div className={`card mt-4 ${status === 'completed' ? 'border-green-200 bg-green-50/20' : status === 'failed' ? 'border-red-200 bg-red-50/20' : 'border-blue-200 bg-blue-50/20'}`}>
      <div className="flex items-center gap-3 mb-3">
        {status === 'completed' ? <CheckCircle size={18} className="text-green-500" /> :
          status === 'failed' ? <AlertCircle size={18} className="text-red-500" /> :
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />}
        <span className="font-semibold text-sm capitalize">{status}</span>
        <span className="text-xs text-slate-500">{message}</span>
        {device && <span className="badge badge-gray ml-auto">{String(device).toUpperCase()}</span>}
      </div>
      {status === 'running' && (
        <>
          <div className="progress-bar mb-2">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="flex justify-between text-xs text-slate-500 mb-2">
            <span>{latestEpoch ? `Epoch ${latestEpoch.epoch}: train loss ${Number(latestEpoch.train_loss).toFixed(4)}${latestEpoch.val_loss != null ? ` · validation loss ${Number(latestEpoch.val_loss).toFixed(4)}` : latestEpoch.val_score != null ? ` · validation score ${Number(latestEpoch.val_score).toFixed(4)}` : ''}` : message}</span>
            {job.elapsed_seconds != null && <span>{Number(job.elapsed_seconds).toFixed(1)}s elapsed</span>}
          </div>
        </>
      )}
      {job.warning && (
        <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          {job.warning}
        </div>
      )}
      {status === 'completed' && metrics && (
        <div className="grid grid-cols-3 gap-2 mt-3">
          {Object.entries(metrics).filter(([k]) => !['confusion_matrix','class_names'].includes(k)).map(([k, v]) => (
            <div key={k} className="text-center p-2 bg-white rounded-lg border border-green-100">
              <div className="text-xs text-slate-500">{k.toUpperCase()}</div>
              <div className="text-sm font-bold text-slate-800">{typeof v === 'number' ? v.toFixed(4) : v}</div>
            </div>
          ))}
        </div>
      )}
      {status === 'completed' && (
        <button onClick={() => navigate('/experiments')} className="btn-primary mt-3">
          <Zap size={14} /> View in Experiments
        </button>
      )}
      {status === 'failed' && job.error && (
        <div className="error-box mt-2"><AlertCircle size={13} /><code className="text-[10px]">{job.message}</code></div>
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
  const navigate = useNavigate();

  useEffect(() => {
    getAvailableModels().then(r => setModels(r.data)).catch(e => setError(`Could not load available models: ${e.message}`)).finally(() => setModelsLoading(false));
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
    const interval = setInterval(async () => {
      try {
        const r = await getExperimentStatus(expId);
        const job = r.data;
        setJobs(prev => ({ ...prev, [expId]: job }));
        if (job.status === 'completed' || job.status === 'failed') {
          clearInterval(interval);
          delete pollingRef.current[expId];
        }
      } catch (error) {
        clearInterval(interval);
        delete pollingRef.current[expId];
        setError(`Could not refresh training status: ${error.message}`);
      }
    }, 1500);
    pollingRef.current[expId] = interval;
  };

  const handleTrain = async () => {
    if (!selectedDataset) { setError('No dataset selected. Go to Datasets first.'); return; }
    if (!targetColumn) { setError('No target column set. Go to Preprocessing first.'); return; }
    if (!taskType) { setError('No task type confirmed. Go to Preprocessing first.'); return; }
    if (selectedModels.length === 0) { setError('Select at least one model.'); return; }
    setError(null);
    setLaunching(true);

    for (const modelId of selectedModels) {
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
        setJobs(prev => ({ ...prev, [expId]: { status: 'pending', progress: 0, message: 'Queued…' } }));
        pollStatus(expId);
      } catch (e) {
        setError(`Failed to start ${modelId}: ${e.message}`);
      }
    }
    setLaunching(false);
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
