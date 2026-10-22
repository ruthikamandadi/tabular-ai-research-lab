import React, { useState, useEffect } from 'react';
import { listExperiments, getExperimentResults, deleteExperiment } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Trash2, RefreshCw, Download, CheckCircle, AlertCircle, Clock, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { LineChart, Line, Legend } from 'recharts';
import { getReportHtmlUrl, getReportJsonUrl, getCompareUrl } from '../services/api';
import HelpTooltip from '../components/HelpTooltip';

const STATUS_COLORS = { completed: 'badge-green', failed: 'badge-red', running: 'badge-blue', pending: 'badge-yellow' };

function ExperimentRow({ exp, onDelete, onDetails }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <tr className="hover:bg-slate-50 cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <td className="table-td">
          <div className="font-mono text-xs text-slate-400">{exp.id.slice(0, 8)}</div>
          <div className="text-sm font-medium text-slate-800">{exp.model_name}</div>
        </td>
        <td className="table-td text-sm">{exp.dataset_name}</td>
        <td className="table-td"><code className="text-xs bg-slate-100 px-1.5 py-0.5 rounded">{exp.target_column}</code></td>
        <td className="table-td text-xs text-slate-500">{exp.task_type?.replace(/_/g, ' ')}</td>
        <td className="table-td"><span className={`badge ${STATUS_COLORS[exp.status] || 'badge-gray'}`}>{exp.status}</span></td>
        <td className="table-td text-xs font-mono">
          {exp.metrics && Object.entries(exp.metrics)
            .filter(([k]) => !['confusion_matrix','class_names'].includes(k))
            .slice(0, 2)
            .map(([k, v]) => <div key={k}>{k}: <strong>{typeof v === 'number' ? v.toFixed(4) : v}</strong></div>)}
        </td>
        <td className="table-td text-xs text-slate-400">
          {exp.train_duration_s ? `${exp.train_duration_s.toFixed(1)}s` : '—'}
        </td>
        <td className="table-td" onClick={e => e.stopPropagation()}>
          <div className="flex gap-2">
            {exp.status === 'completed' && (
              <a href={getReportHtmlUrl(exp.id)} target="_blank" rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-600" title="HTML Report">
                <ExternalLink size={14} />
              </a>
            )}
            <button onClick={() => onDelete(exp.id)} className="text-red-400 hover:text-red-600" title="Delete">
              <Trash2 size={14} />
            </button>
            <button onClick={() => setExpanded(!expanded)} className="text-slate-400 hover:text-slate-600">
              {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          </div>
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={8} className="bg-slate-50 border-t border-slate-100">
            <div className="p-4">
              {exp.status === 'completed' && exp.metrics && (
                <div className="grid grid-cols-2 gap-6">
                  <div>
                    <div className="text-xs font-semibold text-slate-600 mb-2">All Metrics</div>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(exp.metrics).filter(([k]) => !['confusion_matrix','class_names'].includes(k)).map(([k, v]) => (
                        <div key={k} className="text-xs">
                          <span className="text-slate-500">{k}: </span>
                          <span className="font-semibold">{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                        </div>
                      ))}
                    </div>
                    <div className="mt-3 text-xs text-slate-400">
                      Features: {exp.n_features} · Train: {exp.n_train_rows?.toLocaleString()} · Test: {exp.n_test_rows?.toLocaleString()}
                    </div>
                  </div>
                  {exp.training_history?.length > 0 && (
                    <div>
                      <div className="text-xs font-semibold text-slate-600 mb-2">Training History</div>
                      <ResponsiveContainer width="100%" height={140}>
                        <LineChart data={exp.training_history}>
                          <XAxis dataKey="epoch" tick={{ fontSize: 9 }} />
                          <YAxis tick={{ fontSize: 9 }} />
                          <Tooltip />
                          <Legend wrapperStyle={{ fontSize: 10 }} />
                          <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                          <Line
                            type="monotone"
                            dataKey={exp.training_history.some(row => row.val_loss != null) ? 'val_loss' : 'val_score'}
                            stroke="#f59e0b"
                            dot={false}
                            strokeWidth={1.5}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              )}
              {exp.task_type === 'time_series' && exp.forecast_data?.length > 0 && (
                <div className="mt-5">
                  <div className="text-xs font-semibold text-slate-600 mb-2">Forecast vs Actual</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={exp.forecast_data}>
                      <XAxis dataKey="date" tick={{ fontSize: 9 }} minTickGap={24} />
                      <YAxis tick={{ fontSize: 9 }} />
                      <Tooltip />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line type="monotone" dataKey="actual" name="Actual" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
                      <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#f59e0b" dot={false} strokeWidth={1.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              {exp.status === 'failed' && (
                <div className="error-box"><AlertCircle size={13} /><span>{exp.error_message || 'Training failed.'}</span></div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('created_at');
  const [filterStatus, setFilterStatus] = useState('all');
  const [selected, setSelected] = useState([]);
  const [error, setError] = useState('');

  const refresh = () => {
    setLoading(true);
    setError('');
    listExperiments().then(r => setExperiments(r.data)).catch(e => setError(e.message)).finally(() => setLoading(false));
  };

  useEffect(() => { refresh(); }, []);

  const handleDelete = async (id) => {
    if (!confirm('Delete this experiment?')) return;
    try { await deleteExperiment(id); refresh(); } catch (e) { alert(e.message); }
  };

  const filtered = experiments.filter(e => filterStatus === 'all' || e.status === filterStatus);

  // Comparison chart for completed experiments with same task
  const completed = filtered.filter(e => e.status === 'completed');
  const primaryMetric = completed.length > 0 ? (() => {
    const task = completed[0].task_type;
    if (task === 'regression') return 'rmse';
    if (task === 'binary_classification') return 'roc_auc';
    return 'accuracy';
  })() : null;

  const chartData = completed
    .filter(e => e.metrics?.[primaryMetric] != null)
    .map(e => ({ name: e.model_name, value: parseFloat(e.metrics[primaryMetric].toFixed(4)), id: e.id }))
    .sort((a, b) => (primaryMetric === 'rmse' ? a.value - b.value : b.value - a.value));

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2"><h1 className="page-title">Experiments</h1><HelpTooltip text="Review model training runs, inspect metrics and charts, and compare completed experiments." /></div>
          <p className="page-sub">{experiments.length} total experiment runs</p>
        </div>
        <button onClick={refresh} className="btn-secondary"><RefreshCw size={14} /> Refresh</button>
      </div>

      {chartData.length > 1 && (
        <div className="card mb-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="section-title mb-0">Model Comparison — {primaryMetric?.toUpperCase()}</h2>
            {selected.length >= 2 && (
              <a href={getCompareUrl(selected)} target="_blank" rel="noopener noreferrer" className="btn-secondary text-xs">
                <Download size={12} /> Export Comparison CSV
              </a>
            )}
            {error && <div className="error-box mb-4"><AlertCircle size={14} />{error}</div>}
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} layout="vertical">
              <XAxis type="number" tick={{ fontSize: 10 }} domain={primaryMetric === 'rmse' ? ['auto', 'auto'] : [0, 1]} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={120} />
              <Tooltip formatter={v => [v.toFixed(4), primaryMetric]} />
              <Bar dataKey="value" radius={[0,4,4,0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={i === 0 ? '#3b82f6' : i === 1 ? '#8b5cf6' : '#06b6d4'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-400 mt-2">
            {primaryMetric === 'rmse' ? 'Lower RMSE is better.' : `Higher ${primaryMetric} is better.`}
            Metrics computed on held-out test set only.
          </p>
        </div>
      )}

      {/* Filters */}
      <div className="flex gap-3 mb-4">
        {['all', 'completed', 'running', 'failed', 'pending'].map(s => (
          <button key={s} onClick={() => setFilterStatus(s)}
            className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${filterStatus === s ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>
            {s === 'all' ? `All (${experiments.length})` : `${s} (${experiments.filter(e => e.status === s).length})`}
          </button>
        ))}
      </div>

      <div className="card">
        {loading ? (
          <div className="space-y-3">{[...Array(3)].map((_, i) => <div key={i} className="skeleton h-12 rounded-lg" />)}</div>
        ) : filtered.length === 0 ? (
          <div className="text-center py-12 text-slate-400">
            <Activity size={40} className="mx-auto mb-3 opacity-40" />
            <div className="text-sm">No experiments yet. Train models in Model Lab.</div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead><tr>
                <th className="table-th">Model / ID</th>
                <th className="table-th">Dataset</th>
                <th className="table-th">Target</th>
                <th className="table-th">Task</th>
                <th className="table-th">Status</th>
                <th className="table-th">Key Metrics</th>
                <th className="table-th">Duration</th>
                <th className="table-th">Actions</th>
              </tr></thead>
              <tbody>
                {filtered.map(exp => (
                  <ExperimentRow key={exp.id} exp={exp} onDelete={handleDelete} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
