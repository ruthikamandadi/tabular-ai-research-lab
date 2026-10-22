import React, { useState, useEffect } from 'react';
import { listExperiments } from '../services/api';
import { getReportHtmlUrl, getReportJsonUrl, getCompareUrl } from '../services/api';
import { FileText, Download, ExternalLink, CheckCircle, AlertCircle } from 'lucide-react';
import HelpTooltip from '../components/HelpTooltip';

export default function Reports() {
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    listExperiments().then(r => setExperiments(r.data.filter(e => e.status === 'completed'))).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  const toggleSelect = (id) => setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);

  return (
    <div>
      <div className="flex items-center gap-2"><h1 className="page-title">Reports</h1><HelpTooltip text="Generate and download HTML, JSON, and comparison reports for completed experiments." /></div>
      <p className="page-sub">Export experiment results, HTML reports, and comparison data.</p>
      {loading && <div className="card mb-4"><div className="skeleton h-12 rounded-lg" /></div>}
      {error && <div className="error-box mb-4"><AlertCircle size={14} />{error}</div>}

      {!loading && !error && (
        experiments.length === 0 ? (
          <div className="card text-center py-12 text-slate-400">
            <FileText size={40} className="mx-auto mb-3 opacity-40" />
            <div className="text-sm">No completed experiments to report on.</div>
          </div>
        ) : (
          <>
          <div className="card mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="section-title mb-0">Completed Experiments</h2>
              {selected.length >= 2 && (
                <a href={getCompareUrl(selected)} target="_blank" rel="noopener noreferrer" className="btn-primary text-sm">
                  <Download size={14} /> Compare Selected ({selected.length}) as CSV
                </a>
              )}
            </div>
            <p className="text-xs text-slate-400 mb-4">Select multiple experiments to compare them side-by-side.</p>
            <div className="space-y-3">
              {experiments.map(exp => (
                <div key={exp.id} className={`flex items-center gap-4 p-4 rounded-xl border transition-all ${selected.includes(exp.id) ? 'border-blue-300 bg-blue-50/20' : 'border-slate-200 hover:border-slate-300'}`}>
                  <input type="checkbox" checked={selected.includes(exp.id)} onChange={() => toggleSelect(exp.id)} className="cursor-pointer" />
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-semibold text-sm">{exp.model_name}</span>
                      <span className="badge badge-green">{exp.status}</span>
                    </div>
                    <div className="text-xs text-slate-500">
                      {exp.dataset_name} · target: {exp.target_column} · {exp.task_type?.replace(/_/g,' ')}
                    </div>
                    {exp.metrics && (
                      <div className="flex flex-wrap gap-3 mt-1">
                        {Object.entries(exp.metrics).filter(([k]) => !['confusion_matrix','class_names'].includes(k)).slice(0,4).map(([k,v]) => (
                          <span key={k} className="text-xs text-slate-600">{k}: <strong>{typeof v === 'number' ? v.toFixed(4) : v}</strong></span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2 shrink-0">
                    <a href={getReportHtmlUrl(exp.id)} target="_blank" rel="noopener noreferrer"
                      className="btn-secondary text-xs px-3 py-1.5">
                      <ExternalLink size={12} /> HTML Report
                    </a>
                    <a href={getReportJsonUrl(exp.id)} target="_blank" rel="noopener noreferrer"
                      className="btn-secondary text-xs px-3 py-1.5">
                      <Download size={12} /> JSON
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card border-blue-100 bg-blue-50/20">
            <h2 className="section-title">Report Contents</h2>
            <ul className="text-sm text-slate-600 space-y-1">
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Dataset details and schema</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Preprocessing configuration</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Model configuration and hyperparameters</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Evaluation metrics on held-out test set</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Training performance (duration, feature count)</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> MLP training history (when applicable)</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Limitations and disclaimer notice</li>
              <li className="flex items-center gap-2"><CheckCircle size={13} className="text-green-500" /> Research paper citation</li>
            </ul>
          </div>
          </>
        )
      )}
    </div>
  );
}
