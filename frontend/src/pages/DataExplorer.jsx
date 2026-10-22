import React, { useState, useEffect } from 'react';
import { useApp } from '../contexts/AppContext';
import { getDatasetProfile, getDatasetPreview, listDatasets } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { AlertCircle, AlertTriangle, Database, Search, RefreshCw } from 'lucide-react';
import HelpTooltip from '../components/HelpTooltip';

function MissingBar({ value, max }) {
  const pct = max > 0 ? (value / max) * 100 : 0;
  return (
    <div className="flex items-center gap-2">
      <div className="progress-bar flex-1">
        <div className="progress-fill bg-amber-400" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-500 w-10 text-right">{pct.toFixed(1)}%</span>
    </div>
  );
}

function TypeBadge({ type }) {
  const map = {
    numeric: 'badge-blue', integer: 'badge-blue', categorical: 'badge-purple',
    categorical_int: 'badge-purple', boolean: 'badge-green', datetime: 'badge-yellow',
    text: 'badge-gray', text_or_id: 'badge-red', other: 'badge-gray',
  };
  return <span className={`badge ${map[type] || 'badge-gray'}`}>{type}</span>;
}

export default function DataExplorer() {
  const { selectedDataset, setSelectedDataset } = useApp();
  const [datasets, setDatasets] = useState([]);
  const [profile, setProfile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    listDatasets().then(r => setDatasets(r.data)).catch(e => setError(e.message)).finally(() => setDatasetsLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedDataset) return;
    setLoading(true);
    setError(null);
    Promise.all([
      getDatasetProfile(selectedDataset.id),
      getDatasetPreview(selectedDataset.id, 100),
    ]).then(([p, prev]) => {
      setProfile(p.data);
      setPreview(prev.data);
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, [selectedDataset]);

  const filteredCols = profile?.columns?.filter(c =>
    c.name.toLowerCase().includes(search.toLowerCase())
  ) || [];

  if (!selectedDataset) {
    return (
      <div>
        <div className="flex items-center gap-2"><h1 className="page-title">Data Explorer</h1><HelpTooltip text="Inspect a selected dataset’s preview, profile, distributions, missing values, and correlations." /></div>
        <p className="page-sub">Select a dataset to begin exploration.</p>
        <div className="card">
          <h2 className="section-title">Select a Dataset</h2>
          {datasetsLoading && <div className="skeleton h-12 rounded-lg" />}
          {error && <div className="error-box mb-3"><AlertCircle size={14} />{error}</div>}
          {datasets.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              <Database size={36} className="mx-auto mb-2 opacity-40" />
              <div className="text-sm">No datasets available. Upload one first.</div>
            </div>
          ) : (
            <div className="space-y-2">
              {datasets.map(ds => (
                <div key={ds.id} onClick={() => setSelectedDataset(ds)}
                  className="flex items-center justify-between p-3 rounded-lg border border-slate-200 hover:border-blue-300 hover:bg-blue-50/30 cursor-pointer transition-all">
                  <div>
                    <div className="font-medium text-sm">{ds.name}</div>
                    <div className="text-xs text-slate-400">{ds.row_count?.toLocaleString()} rows · {ds.col_count} columns</div>
                  </div>
                  <span className="badge badge-blue">Select</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2"><h1 className="page-title">Data Explorer</h1><HelpTooltip text="Inspect a selected dataset’s preview, profile, distributions, missing values, and correlations." /></div>
          <p className="page-sub">Dataset: <strong>{selectedDataset.name}</strong></p>
        </div>
        <div className="flex gap-2">
          <select className="select-field w-48" value={selectedDataset.id}
            onChange={e => setSelectedDataset(datasets.find(d => d.id === e.target.value))}>
            {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
          <button title="Reload dataset profile and preview" onClick={() => {
            setLoading(true);
            setError(null);
            Promise.all([getDatasetProfile(selectedDataset.id), getDatasetPreview(selectedDataset.id, 100)])
              .then(([p, pr]) => { setProfile(p.data); setPreview(pr.data); })
              .catch(e => setError(e.message)).finally(() => setLoading(false));
          }} className="btn-secondary"><RefreshCw size={14} /></button>
        </div>
      </div>

      {loading && (
        <div className="space-y-4">
          <div className="skeleton h-32 rounded-xl" />
          <div className="grid grid-cols-4 gap-4">{[...Array(4)].map((_, i) => <div key={i} className="skeleton h-20 rounded-xl" />)}</div>
        </div>
      )}
      {error && <div className="error-box mb-4"><AlertCircle size={14} />{error}</div>}

      {profile && !loading && (
        <>
          {/* Overview cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="metric-card"><div className="metric-label">Rows</div><div className="metric-value">{profile.n_rows?.toLocaleString()}</div></div>
            <div className="metric-card"><div className="metric-label">Columns</div><div className="metric-value">{profile.n_cols}</div></div>
            <div className="metric-card"><div className="metric-label">Duplicate Rows</div><div className="metric-value">{profile.duplicate_rows}</div></div>
            <div className="metric-card"><div className="metric-label">Numeric Cols</div><div className="metric-value">{profile.numeric_columns?.length}</div></div>
          </div>

          {/* Warnings */}
          {profile.warnings?.length > 0 && (
            <div className="card mb-6">
              <h2 className="section-title flex items-center gap-2"><AlertTriangle size={16} className="text-amber-500" />Data Quality Warnings</h2>
              <div className="space-y-2">
                {profile.warnings.map((w, i) => (
                  <div key={i} className="warning-box">
                    <AlertTriangle size={13} className="shrink-0 mt-0.5" />
                    <span>{w.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tabs */}
          <div className="flex gap-1 mb-6 border-b border-slate-200">
            {['overview', 'columns', 'missing', 'preview'].map(t => (
              <button key={t} onClick={() => setActiveTab(t)}
                className={`px-4 py-2 text-sm font-medium capitalize transition-colors border-b-2 -mb-px
                  ${activeTab === t ? 'border-blue-500 text-blue-600' : 'border-transparent text-slate-500 hover:text-slate-800'}`}>
                {t === 'overview' ? 'Column Overview' : t === 'preview' ? 'Data Preview' : t === 'missing' ? 'Missing Values' : 'Column Details'}
              </button>
            ))}
          </div>

          {activeTab === 'overview' && (
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <Search size={14} className="text-slate-400" />
                <input className="input-field max-w-xs" placeholder="Search columns…" value={search} onChange={e => setSearch(e.target.value)} />
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead><tr>
                    <th className="table-th">Column</th>
                    <th className="table-th">Type</th>
                    <th className="table-th">Inferred</th>
                    <th className="table-th">Missing</th>
                    <th className="table-th">Unique</th>
                    <th className="table-th">Mean</th>
                    <th className="table-th">Min</th>
                    <th className="table-th">Max</th>
                  </tr></thead>
                  <tbody>
                    {filteredCols.map(c => (
                      <tr key={c.name} className={`hover:bg-slate-50 ${c.is_constant ? 'bg-red-50/30' : ''}`}>
                        <td className="table-td font-mono text-xs text-slate-800">{c.name}</td>
                        <td className="table-td text-xs text-slate-500">{c.dtype}</td>
                        <td className="table-td"><TypeBadge type={c.inferred_type} /></td>
                        <td className="table-td">
                          <span className={c.missing_count > 0 ? 'text-amber-600' : 'text-slate-400'}>
                            {c.missing_count} ({c.missing_pct}%)
                          </span>
                        </td>
                        <td className="table-td">{c.unique_count}</td>
                        <td className="table-td text-slate-500">{c.stats?.mean?.toFixed(4) ?? '—'}</td>
                        <td className="table-td text-slate-500">{c.stats?.min?.toFixed(4) ?? '—'}</td>
                        <td className="table-td text-slate-500">{c.stats?.max?.toFixed(4) ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'columns' && (
            <div className="grid md:grid-cols-2 gap-4">
              {filteredCols.filter(c => c.histogram).map(c => (
                <div key={c.name} className="card-sm">
                  <div className="font-medium text-sm mb-2">{c.name} <TypeBadge type={c.inferred_type} /></div>
                  <ResponsiveContainer width="100%" height={120}>
                    <BarChart data={c.histogram?.edges?.slice(0,-1).map((e,i) => ({
                      x: e.toFixed(2), count: c.histogram.counts[i]
                    }))}>
                      <XAxis dataKey="x" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
                      <YAxis tick={{ fontSize: 9 }} />
                      <Tooltip formatter={v => [v, 'Count']} labelFormatter={l => `~${l}`} />
                      <Bar dataKey="count" fill="#3b82f6" radius={[2,2,0,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ))}
              {filteredCols.filter(c => c.top_values).map(c => (
                <div key={c.name} className="card-sm">
                  <div className="font-medium text-sm mb-2">{c.name} <TypeBadge type={c.inferred_type} /></div>
                  <ResponsiveContainer width="100%" height={Math.min(120, c.top_values.length * 22)}>
                    <BarChart data={c.top_values.slice(0,10)} layout="vertical">
                      <XAxis type="number" tick={{ fontSize: 9 }} />
                      <YAxis type="category" dataKey="value" tick={{ fontSize: 9 }} width={80} />
                      <Tooltip />
                      <Bar dataKey="count" fill="#8b5cf6" radius={[0,2,2,0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          )}

          {activeTab === 'missing' && (
            <div className="card">
              <h2 className="section-title">Missing Value Analysis</h2>
              {profile.missing_summary?.length === 0 ? (
                <div className="text-center py-8 text-green-600">
                  <div className="text-2xl mb-2">✓</div>
                  <div className="text-sm font-medium">No missing values found!</div>
                </div>
              ) : (
                <div className="space-y-3">
                  {profile.missing_summary?.map(m => (
                    <div key={m.column}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="font-mono text-slate-700">{m.column}</span>
                        <span className="text-slate-500">{m.missing} missing</span>
                      </div>
                      <MissingBar value={m.missing} max={profile.n_rows} />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'preview' && preview && (
            <div className="card">
              <div className="text-xs text-slate-400 mb-3">Showing {preview.preview_rows} of {preview.total_rows?.toLocaleString()} rows</div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead><tr>
                    {preview.columns?.map(c => <th key={c} className="table-th font-mono">{c}</th>)}
                  </tr></thead>
                  <tbody>
                    {preview.rows?.map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50/50'}>
                        {preview.columns?.map(c => (
                          <td key={c} className="table-td font-mono">
                            {row[c] === null || row[c] === undefined
                              ? <span className="text-red-400 italic">null</span>
                              : String(row[c])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
