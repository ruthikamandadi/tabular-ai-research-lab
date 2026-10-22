import React, { useState, useEffect } from 'react';
import { listExperiments, predictSingle, getModelSchema, predictCsv } from '../services/api';
import { Zap, AlertCircle, Upload, CheckCircle, Info } from 'lucide-react';
import HelpTooltip from '../components/HelpTooltip';

export default function Predictions() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExpId, setSelectedExpId] = useState('');
  const [schema, setSchema] = useState(null);
  const [formValues, setFormValues] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState(null);
  const [batchFile, setBatchFile] = useState(null);
  const [experimentsLoading, setExperimentsLoading] = useState(true);

  useEffect(() => {
    listExperiments().then(r => {
      const done = r.data.filter(e => e.status === 'completed');
      setExperiments(done);
      if (done.length > 0) setSelectedExpId(done[0].id);
    }).catch(e => setError(e.message)).finally(() => setExperimentsLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedExpId) return;
    getModelSchema(selectedExpId).then(r => {
      setSchema(r.data);
      // Initialize form with empty values
      const init = {};
      r.data.input_schema?.forEach(f => { init[f.column] = ''; });
      setFormValues(init);
      setPrediction(null);
    }).catch(e => setError(e.message));
  }, [selectedExpId]);

  const handlePredict = async () => {
    setLoading(true); setError(null); setPrediction(null);
    const features = {};
    schema.input_schema?.forEach(f => {
      const v = formValues[f.column];
      if (v !== '' && v !== null && v !== undefined) {
        const num = parseFloat(v);
        features[f.column] = isNaN(num) ? v : num;
      }
    });
    try {
      const r = await predictSingle(selectedExpId, features);
      setPrediction(r.data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async (event) => {
    event.preventDefault();
    setBatchError(null);

    if (!batchFile) {
      setBatchError('Choose a CSV file before starting batch prediction.');
      return;
    }

    setBatchLoading(true);
    try {
      const response = await predictCsv(selectedExpId, batchFile);
      const downloadUrl = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = 'predictions.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 1000);
    } catch (e) {
      setBatchError(`Batch prediction failed: ${e.message}`);
    } finally {
      setBatchLoading(false);
    }
  };

  const exp = experiments.find(e => e.id === selectedExpId);

  return (
    <div>
      <div className="flex items-center gap-2"><h1 className="page-title">Predictions</h1><HelpTooltip text="Run single-record or batch predictions using a completed experiment’s saved model." /></div>
      <p className="page-sub">Run predictions on new data using trained model artifacts.</p>
      {experimentsLoading && <div className="card mb-4"><div className="skeleton h-10 rounded-lg" /></div>}

      {experiments.length === 0 && (
        <div className="card text-center py-12 text-slate-400">
          <Zap size={40} className="mx-auto mb-3 opacity-40" />
          <div className="text-sm">No trained models available. Complete a training experiment first.</div>
        </div>
      )}

      {experiments.length > 0 && (
        <>
          <div className="card mb-6">
            <label className="label">Select Trained Model</label>
            <select className="select-field max-w-lg" value={selectedExpId} onChange={e => setSelectedExpId(e.target.value)}>
              {experiments.map(e => (
                <option key={e.id} value={e.id}>{e.model_name} — {e.dataset_name} (target: {e.target_column})</option>
              ))}
            </select>
            {exp && (
              <div className="mt-3 flex gap-4 text-xs text-slate-500">
                <span>Task: <strong>{exp.task_type?.replace(/_/g,' ')}</strong></span>
                <span>Features: <strong>{exp.n_features}</strong></span>
                {exp.metrics && Object.entries(exp.metrics).filter(([k]) => !['confusion_matrix','class_names'].includes(k)).slice(0,2).map(([k,v]) => (
                  <span key={k}>{k}: <strong>{typeof v === 'number' ? v.toFixed(4) : v}</strong></span>
                ))}
              </div>
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Single prediction form */}
            <div className="card">
              <h2 className="section-title">Single Record Prediction</h2>
              {schema ? (
                <>
                  <div className="grid grid-cols-2 gap-3 mb-4 max-h-80 overflow-y-auto pr-1">
                    {schema.input_schema?.map(f => (
                      <div key={f.column}>
                        <label className="label truncate" title={f.column}>{f.column}</label>
                        <input className="input-field" placeholder={`${f.dtype}`}
                          value={formValues[f.column] || ''}
                          onChange={e => setFormValues(prev => ({ ...prev, [f.column]: e.target.value }))} />
                      </div>
                    ))}
                  </div>
                  <button onClick={handlePredict} disabled={loading} className="btn-primary w-full justify-center">
                    {loading ? <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Predicting…</> : <><Zap size={14} /> Predict</>}
                  </button>

                  {error && <div className="error-box mt-3"><AlertCircle size={13} />{error}</div>}

                  {prediction && (
                    <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-xl">
                      <div className="text-xs font-semibold text-blue-600 mb-2">Prediction Result</div>
                      <div className="text-3xl font-bold text-slate-900 mb-2">{prediction.prediction}</div>
                      {prediction.class_probabilities && (
                        <div>
                          <div className="text-xs text-slate-500 mb-2">Class Probabilities:</div>
                          {Object.entries(prediction.class_probabilities).map(([cls, prob]) => (
                            <div key={cls} className="flex items-center gap-2 mb-1">
                              <span className="text-xs w-24 truncate text-slate-600">{cls}</span>
                              <div className="progress-bar flex-1">
                                <div className="progress-fill" style={{ width: `${prob*100}%` }} />
                              </div>
                              <span className="text-xs text-slate-500 w-12 text-right">{(prob*100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                      {prediction.warnings?.map((w, i) => (
                        <div key={i} className="warning-box mt-2 text-[10px]"><AlertCircle size={11} />{w}</div>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-slate-400 text-sm">Select a model to load input schema.</div>
              )}
            </div>

            {/* Batch CSV prediction */}
            <div className="card">
              <h2 className="section-title">Batch CSV Prediction</h2>
              <div className="info-box mb-4">
                <Info size={13} className="shrink-0 mt-0.5" />
                <span>Upload a CSV with the same columns as the training data (excluding the target column). Predictions are returned as a downloadable CSV.</span>
              </div>
              {schema && (
                <div className="mb-4">
                  <div className="text-xs font-semibold text-slate-600 mb-2">Required columns ({schema.input_schema?.length}):</div>
                  <div className="flex flex-wrap gap-1 max-h-32 overflow-y-auto">
                    {schema.input_schema?.map(f => (
                      <code key={f.column} className="text-[10px] bg-slate-100 px-1.5 py-0.5 rounded">{f.column}</code>
                    ))}
                  </div>
                </div>
              )}
              {selectedExpId && (
                <form onSubmit={handleBatchPredict}>
                  <label className="btn-secondary cursor-pointer inline-flex">
                    <Upload size={14} /> Choose CSV File
                    <input
                      type="file"
                      name="file"
                      accept=".csv"
                      className="hidden"
                      onChange={e => {
                        setBatchFile(e.target.files?.[0] || null);
                        setBatchError(null);
                      }}
                    />
                  </label>
                  {batchFile && <div className="mt-2 text-xs text-slate-500">{batchFile.name}</div>}
                  <button type="submit" disabled={batchLoading || !batchFile} className="btn-primary mt-3 w-full justify-center">
                    {batchLoading
                      ? <><div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" /> Predicting…</>
                      : <><Zap size={14} /> Predict & Download CSV</>}
                  </button>
                  {batchError && <div className="error-box mt-3"><AlertCircle size={13} />{batchError}</div>}
                </form>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
