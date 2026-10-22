import React, { useState, useEffect } from 'react';
import { listExperiments, getExplanations } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Lightbulb, AlertCircle, RefreshCw, Info } from 'lucide-react';
import HelpTooltip from '../components/HelpTooltip';

const COLORS = ['#3b82f6','#8b5cf6','#06b6d4','#10b981','#f59e0b','#ef4444','#64748b','#ec4899'];

export default function Explainability() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExpId, setSelectedExpId] = useState('');
  const [importance, setImportance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [experimentsLoading, setExperimentsLoading] = useState(true);

  useEffect(() => {
    listExperiments().then(r => {
      const completed = r.data.filter(e => e.status === 'completed');
      setExperiments(completed);
      if (completed.length > 0) setSelectedExpId(completed[0].id);
    }).catch(e => setError(e.message)).finally(() => setExperimentsLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedExpId) return;
    setLoading(true); setError(null); setImportance(null);
    getExplanations(selectedExpId)
      .then(r => setImportance(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [selectedExpId]);

  const exp = experiments.find(e => e.id === selectedExpId);

  const top20 = importance?.importances?.slice(0, 20) || [];
  const maxImp = top20.length > 0 ? Math.max(...top20.map(f => Math.abs(f.importance))) : 1;

  return (
    <div>
      <div className="flex items-center gap-2"><h1 className="page-title">Explainability</h1><HelpTooltip text="Explore global and per-feature explanations for completed model experiments." /></div>
      <p className="page-sub">Understand which features most influence model predictions.</p>

      <div className="info-box mb-6">
        <Info size={14} className="shrink-0 mt-0.5" />
        <span>
          Feature importance is computed using <strong>SHAP</strong> (when compatible) or <strong>permutation importance</strong> as a fallback.
          High importance means the feature correlates with predictions — not that it causes the outcome.
        </span>
      </div>

      <div className="card mb-6">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="label">Select Completed Experiment</label>
            <select className="select-field" value={selectedExpId} onChange={e => setSelectedExpId(e.target.value)}>
              <option value="">— Select experiment —</option>
              {experiments.map(e => (
                <option key={e.id} value={e.id}>{e.model_name} on {e.dataset_name} (target: {e.target_column})</option>
              ))}
            </select>
          </div>
          {experimentsLoading && <div className="card mb-4"><div className="skeleton h-10 rounded-lg" /></div>}
          <button onClick={() => {
            if (!selectedExpId) return;
            setLoading(true); setError(null); setImportance(null);
            getExplanations(selectedExpId).then(r => setImportance(r.data)).catch(e => setError(e.message)).finally(() => setLoading(false));
          }} className="btn-secondary mt-5"><RefreshCw size={14} /></button>
        </div>
      </div>

      {experiments.length === 0 && (
        <div className="card text-center py-12 text-slate-400">
          <Lightbulb size={40} className="mx-auto mb-3 opacity-40" />
          <div className="text-sm">No completed experiments. Train models in Model Lab first.</div>
        </div>
      )}

      {error && <div className="error-box mb-4"><AlertCircle size={14} />{error}</div>}

      {loading && (
        <div className="card">
          <div className="text-sm text-blue-600 mb-4">Computing feature importance…</div>
          <div className="space-y-2">{[...Array(8)].map((_, i) => <div key={i} className="skeleton h-8 rounded" />)}</div>
        </div>
      )}

      {importance && !loading && (
        <>
          <div className="card mb-6">
            <div className="flex items-center justify-between mb-1">
              <h2 className="section-title mb-0">Feature Importance</h2>
              <span className="badge badge-blue">{importance.method}</span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              {importance.method === 'SHAP' ? 'Mean |SHAP value| across test samples — larger = more impact on prediction.' :
               importance.method === 'permutation' ? 'Drop in accuracy/R² when feature is randomly shuffled on test set.' :
               'Model built-in feature importance (e.g. Gini impurity for trees).'}
              {importance.fallback_used && ' (SHAP unavailable for this model type, using permutation fallback.)'}
            </p>

            <ResponsiveContainer width="100%" height={Math.max(250, top20.length * 28)}>
              <BarChart data={[...top20].reverse()} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={160} />
                <Tooltip formatter={v => [v?.toFixed(6), 'Importance']} />
                <Bar dataKey="importance" radius={[0,4,4,0]}>
                  {[...top20].reverse().map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2 className="section-title">Top Features Table</h2>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="table-th">Rank</th>
                  <th className="table-th">Feature</th>
                  <th className="table-th">Importance</th>
                  <th className="table-th">Relative</th>
                </tr></thead>
                <tbody>
                  {top20.map((f, i) => (
                    <tr key={f.feature} className="hover:bg-slate-50">
                      <td className="table-td text-slate-400">#{i+1}</td>
                      <td className="table-td font-mono text-xs">{f.feature}</td>
                      <td className="table-td font-semibold">{f.importance?.toFixed(6)}</td>
                      <td className="table-td">
                        <div className="flex items-center gap-2">
                          <div className="progress-bar flex-1 max-w-24">
                            <div className="progress-fill" style={{ width: `${(Math.abs(f.importance) / maxImp) * 100}%`, background: COLORS[i % COLORS.length] }} />
                          </div>
                          <span className="text-xs text-slate-400">{((Math.abs(f.importance) / maxImp) * 100).toFixed(1)}%</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
