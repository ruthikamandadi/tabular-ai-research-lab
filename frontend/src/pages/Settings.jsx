import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Info, BookOpen, ExternalLink, RefreshCw, CheckCircle, AlertTriangle, AlertCircle, KeyRound, Database } from 'lucide-react';
import { getCapabilities, saveKaggleCredentials, testKaggleConnection, getStorageStats } from '../services/api';
import HelpTooltip from '../components/HelpTooltip';

export default function SettingsPage() {
  const [capabilities, setCapabilities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [waitingForBackend, setWaitingForBackend] = useState(false);
  const [error, setError] = useState('');
  const loadRequestId = useRef(0);
  const [kaggleUsername, setKaggleUsername] = useState('');
  const [kaggleKey, setKaggleKey] = useState('');
  const [kaggleMessage, setKaggleMessage] = useState('');
  const [kaggleError, setKaggleError] = useState('');
  const [kaggleBusy, setKaggleBusy] = useState(false);
  const [kaggleTesting, setKaggleTesting] = useState(false);
  const [storage, setStorage] = useState(null);
  const [storageLoading, setStorageLoading] = useState(true);

  const loadStorage = useCallback(async () => {
    try {
      setStorageLoading(true);
      const res = await getStorageStats();
      setStorage(res.data);
    } catch {
      // ignore
    } finally {
      setStorageLoading(false);
    }
  }, []);

  useEffect(() => {
    loadStorage();
  }, [loadStorage]);

  const loadCapabilities = useCallback(async (refresh = false) => {
    const requestId = ++loadRequestId.current;
    setLoading(true);
    setWaitingForBackend(false);
    setError('');
    for (let retry = 0; retry <= 20; retry += 1) {
      try {
        const response = await getCapabilities(refresh && retry === 0);
        if (requestId !== loadRequestId.current) return;
        setCapabilities(response.data);
        setWaitingForBackend(false);
        setLoading(false);
        return;
      } catch (requestError) {
        if (requestId !== loadRequestId.current) return;
        const status = requestError.response?.status;
        const retryable = !requestError.response || status >= 500 ||
          /network error/i.test(requestError.message || '');
        if (!retryable || retry === 20) {
          const fallback = 'Backend not reachable. Start it with: python -m uvicorn app.main:app --reload (in the backend folder)';
          const detail = requestError.message;
          setError(detail && !/status code (502|504)|network error/i.test(detail)
            ? `${fallback}. Details: ${detail}`
            : fallback);
          setLoading(false);
          setWaitingForBackend(false);
          return;
        }
        setWaitingForBackend(true);
        await new Promise(resolve => setTimeout(resolve, 3000));
        if (requestId !== loadRequestId.current) return;
      }
    }
  }, []);

  useEffect(() => {
    loadCapabilities(false);
    return () => {
      loadRequestId.current += 1;
    };
  }, [loadCapabilities]);

  const unavailableCapabilities = capabilities.filter(capability => !capability.available);
  const credentialsPayload = () => ({ username: kaggleUsername.trim(), key: kaggleKey });

  const handleSaveKaggle = async () => {
    setKaggleBusy(true);
    setKaggleError('');
    setKaggleMessage('');
    try {
      const response = await saveKaggleCredentials(credentialsPayload());
      setKaggleMessage(`${response.data.message} Key: ${response.data.masked_key}`);
      setKaggleKey('');
    } catch (requestError) {
      setKaggleError(requestError.message);
    } finally {
      setKaggleBusy(false);
    }
  };

  const handleTestKaggle = async () => {
    setKaggleTesting(true);
    setKaggleError('');
    setKaggleMessage('');
    try {
      const response = await testKaggleConnection();
      const message = response.data.message || (response.data.ok ? 'Kaggle connection successful.' : 'Kaggle connection test failed.');
      if (response.data.ok) {
        setKaggleMessage(message);
      } else {
        setKaggleError(message);
      }
    } catch (requestError) {
      const data = requestError.response?.data;
      setKaggleError(data?.detail || data?.message || requestError.message || 'Kaggle connection test failed.');
    } finally {
      setKaggleTesting(false);
    }
  };

  return (
    <div>
      <div className="flex items-center gap-2"><h1 className="page-title">Settings & About</h1><HelpTooltip text="Check feature availability, configure optional integrations, and view application information." /></div>
      <p className="page-sub">Application configuration, research references, and system information.</p>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="section-title flex items-center gap-2"><Info size={16} className="text-blue-500" />System Information</h2>
          <div className="space-y-3 text-sm">
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Backend</span>
              <span className="font-medium">FastAPI + Uvicorn</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">ML Framework</span>
              <span className="font-medium">scikit-learn + optional PyTorch models</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Explainability</span>
              <span className="font-medium">SHAP + Permutation</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Frontend</span>
              <span className="font-medium">React + Vite + Tailwind</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span className="text-slate-500">Storage</span>
              <span className="font-medium">{storage?.backend || 'SQLite + Local Files'}</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-slate-500">Boost Library</span>
              <span className="font-medium">XGBoost 3.4</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between gap-3 mb-3">
            <h2 className="section-title flex items-center gap-2 mb-0">
              <Database size={16} className="text-blue-500" />
              Storage & Quota
            </h2>
            <button onClick={loadStorage} disabled={storageLoading} className="btn-secondary text-xs">
              <RefreshCw size={12} className={storageLoading ? 'animate-spin' : ''} /> Refresh
            </button>
          </div>
          {storageLoading && !storage ? (
            <div className="text-sm text-slate-500 py-4 flex items-center gap-2">
              <RefreshCw size={14} className="animate-spin" /> Loading storage metrics...
            </div>
          ) : storage ? (
            <div className="space-y-3 text-sm">
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Database Backend</span>
                <span className="font-semibold text-slate-800">{storage.backend}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Storage Used</span>
                <span className="font-semibold text-slate-800">
                  {storage.used_mb} MB / {storage.limit_mb} MB ({storage.percent_used}%)
                </span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2 my-1 overflow-hidden">
                <div
                  className={`h-2 rounded-full ${
                    storage.percent_used > 85 ? 'bg-red-500' : storage.percent_used > 60 ? 'bg-amber-500' : 'bg-blue-500'
                  }`}
                  style={{ width: `${Math.min(100, Math.max(0, storage.percent_used))}%` }}
                />
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500">Datasets Stored</span>
                <span className="font-medium">{storage.datasets_count}</span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="text-slate-500">Experiments Stored</span>
                <span className="font-medium">{storage.experiments_count}</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Storage quota automatically cleans up oldest records if usage exceeds the limit.
              </p>
            </div>
          ) : (
            <div className="text-sm text-slate-500">Storage stats currently unavailable.</div>
          )}
        </div>

        <div className="card">
          <h2 className="section-title flex items-center gap-2"><BookOpen size={16} className="text-purple-500" />Research Citation</h2>
          <div className="p-4 bg-slate-50 rounded-lg border border-slate-200 text-sm font-mono leading-relaxed mb-4">
            Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2024).
            <em>Deep Neural Networks and Tabular Data: A Survey.</em>
            IEEE Transactions on Neural Networks and Learning Systems.
          </div>
          <div className="space-y-2">
            <a href="https://github.com/kathrinse/TabSurvey" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-2 text-blue-600 hover:underline text-sm">
              <ExternalLink size={14} /> TabSurvey GitHub Repository
            </a>
            <a href="https://ieeexplore.ieee.org/document/10097160" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-2 text-blue-600 hover:underline text-sm">
              <ExternalLink size={14} /> IEEE Paper (DOI: 10.1109/TNNLS.2022.3213414)
            </a>
          </div>
        </div>

        <div className="card">
          <h2 className="section-title">Application Limits</h2>
          <div className="space-y-2 text-sm text-slate-600">
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span>Max upload size</span><span className="font-medium">200 MB</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span>Max training rows</span><span className="font-medium">100,000</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span>Max categories (OHE)</span><span className="font-medium">50</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span>SHAP background samples</span><span className="font-medium">200</span>
            </div>
            <div className="flex justify-between py-2 border-b border-slate-100">
              <span>Optional deep models</span><span className="font-medium">TabNet, FT-Transformer</span>
            </div>
            <div className="flex justify-between py-2">
              <span>GPU support</span><span className="font-medium">Auto-detect (CPU fallback)</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between gap-3">
            <h2 className="section-title mb-0">Capabilities and Limitations</h2>
            <button onClick={() => loadCapabilities(true)} disabled={loading} className="btn-secondary text-xs">
              <RefreshCw size={14} className={loading ? 'animate-spin' : ''} /> Refresh
            </button>
          </div>
          {loading ? (
            <div className="mt-4 flex items-center gap-2 text-sm text-slate-500" role="status">
              <RefreshCw size={14} className="animate-spin" />
              Waiting for the backend to start...
            </div>
          ) : error ? (
            <div className="mt-4 space-y-3">
              <div className="error-box"><AlertCircle size={14} />{error}</div>
              <button onClick={() => loadCapabilities(false)} className="btn-secondary text-xs">Retry</button>
            </div>
          ) : (
            <>
              <ul className="mt-4 space-y-3 text-sm">
                {capabilities.map(capability => (
                  <li key={capability.key} className="rounded-lg border border-slate-100 p-3">
                    <div className="flex items-start gap-2">
                      {capability.available
                        ? <CheckCircle size={16} className="mt-0.5 shrink-0 text-green-600" />
                        : <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-500" />}
                      <div className="min-w-0">
                        <div className={`font-medium ${capability.available ? 'text-green-700' : 'text-amber-700'}`}>
                          {capability.available
                            ? `${capability.label}: Available`
                            : `${capability.label}: Not available${capability.install ? ` - run: pip install ${capability.install}` : ''}`}
                        </div>
                        <p className="mt-1 text-xs leading-relaxed text-slate-500">{capability.note}</p>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
              <div className="mt-5 border-t border-slate-100 pt-4">
                <h3 className="mb-2 text-sm font-semibold text-slate-800">Known Limitations</h3>
                {unavailableCapabilities.length === 0 ? (
                  <p className="text-sm text-slate-500">No known limitations for the current feature set.</p>
                ) : (
                  <ul className="space-y-2 text-sm text-slate-600">
                    {unavailableCapabilities.map(capability => (
                      <li key={capability.key} className="flex gap-2">
                        <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" />
                        <span>{capability.label}: {capability.note}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </div>

        <div className="card">
          <h2 className="section-title flex items-center gap-2"><KeyRound size={16} className="text-amber-500" />Kaggle Credentials</h2>
          <p className="mb-4 text-xs text-slate-500">Credentials are stored locally in backend/storage/kaggle.json. The API key is never returned.</p>
          <div className="space-y-3">
            <div>
              <label className="label">Kaggle Username</label>
              <input className="input-field" value={kaggleUsername} onChange={event => setKaggleUsername(event.target.value)} autoComplete="username" />
            </div>
            <div>
              <label className="label">Kaggle API Key</label>
              <input className="input-field" type="password" value={kaggleKey} onChange={event => setKaggleKey(event.target.value)} autoComplete="new-password" />
            </div>
            {kaggleError && <div className="error-box"><AlertCircle size={14} />{kaggleError}</div>}
            {kaggleMessage && <div className="success-box"><CheckCircle size={14} />{kaggleMessage}</div>}
            <div className="flex flex-wrap gap-2">
              <button onClick={handleSaveKaggle} disabled={kaggleBusy || kaggleTesting || !kaggleUsername.trim() || !kaggleKey} className="btn-primary">
                {kaggleBusy ? 'Working…' : 'Save Credentials'}
              </button>
              <button onClick={handleTestKaggle} disabled={kaggleBusy || kaggleTesting} className="btn-secondary">
                {kaggleTesting ? 'Testing…' : 'Test Connection'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
