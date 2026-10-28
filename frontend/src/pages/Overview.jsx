import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Database, FlaskConical, ArrowRight, CheckCircle, Activity, BookOpen, Play, AlertCircle } from 'lucide-react';
import {
  listDatasets,
  listExperiments,
  checkHealth,
  listSampleDatasets,
  loadSampleDataset,
  uploadDataset,
  createExperiment,
  getExperimentStatus,
} from '../services/api';
import { useApp } from '../contexts/AppContext';
import HelpTooltip from '../components/HelpTooltip';

const DEMO_MODEL_CONFIGS = [
  { model_id: 'logistic_regression', model_config: { max_iter: 300 } },
  { model_id: 'random_forest', model_config: { n_estimators: 40, max_depth: 6 } },
];
const DEMO_PREPROCESSING = {
  test_size: 0.2,
  random_seed: 42,
  numeric_imputer: 'median',
  numeric_scaler: 'standard',
  categorical_imputer: 'most_frequent',
  categorical_encoding: 'onehot',
  drop_identifiers: true,
};

function createDemoChurnFile() {
  const rows = Array.from({ length: 120 }, (_, index) => {
    const tenure = (index * 11) % 72 + 1;
    const monthlyCharges = 35 + ((index * 7) % 20) * 5;
    const contract = index % 5 < 3 ? 'month-to-month' : index % 5 === 3 ? 'one-year' : 'two-year';
    const supportCalls = (index * 7) % 8;
    const internet = ['fiber', 'dsl', 'no'][index % 3];
    const churn = (contract === 'month-to-month' && supportCalls >= 3) ||
      (tenure < 10 && monthlyCharges > 80) ? 'Yes' : 'No';
    return [
      tenure,
      monthlyCharges.toFixed(2),
      (tenure * monthlyCharges).toFixed(2),
      contract,
      internet,
      ['electronic', 'credit_card', 'bank_transfer'][index % 3],
      supportCalls,
      index % 2 === 0 ? 'yes' : 'no',
      index % 4 === 0 ? 1 : 0,
      churn,
    ];
  });
  const header = [
    'tenure_months', 'monthly_charges', 'total_charges', 'contract_type',
    'internet_service', 'payment_method', 'num_support_calls', 'tech_support',
    'senior_citizen', 'churn',
  ];
  const csv = [header, ...rows].map(row => row.join(',')).join('\n');
  return new File([csv], 'demo_churn_120.csv', { type: 'text/csv' });
}

const wait = (milliseconds) => new Promise(resolve => setTimeout(resolve, milliseconds));

function StatCard({ icon: Icon, label, value, color = 'blue' }) {
  const colors = { blue: 'text-blue-600 bg-blue-100', green: 'text-green-600 bg-green-100', purple: 'text-purple-600 bg-purple-100', amber: 'text-amber-600 bg-amber-100' };
  return (
    <div className="metric-card">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colors[color]} mb-3`}>
        <Icon size={20} />
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

function StepCard({ num, title, desc, done, onClick, btnText }) {
  return (
    <div className={`card flex items-start gap-4 ${done ? 'border-green-200 bg-green-50/30' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold shrink-0 ${done ? 'bg-green-500 text-white' : 'bg-blue-100 text-blue-700'}`}>
        {done ? <CheckCircle size={16} /> : num}
      </div>
      <div className="flex-1">
        <div className="font-semibold text-slate-800 mb-1">{title}</div>
        <div className="text-sm text-slate-500 mb-3">{desc}</div>
        <button onClick={onClick} className="btn-primary text-xs px-3 py-1.5">
          {btnText} <ArrowRight size={12} />
        </button>
      </div>
    </div>
  );
}

export default function Overview() {
  const navigate = useNavigate();
  const { selectedDataset, setSelectedDataset, setTargetColumn, setTaskType, setPreprocessingConfig } = useApp();
  const [stats, setStats] = useState({ datasets: 0, experiments: 0, completed: 0, bestScore: null });
  const [backendOk, setBackendOk] = useState(null);
  const [sampleDatasets, setSampleDatasets] = useState([]);
  const [loadingSample, setLoadingSample] = useState(null);
  const [demoRunning, setDemoRunning] = useState(false);
  const [demoProgress, setDemoProgress] = useState(0);
  const [demoMessage, setDemoMessage] = useState('');
  const [demoError, setDemoError] = useState('');
  const [demoComplete, setDemoComplete] = useState(false);

  const refreshStats = async () => {
    const [datasetsResult, experimentsResult] = await Promise.all([listDatasets(), listExperiments()]);
    const completed = experimentsResult.data.filter(experiment => experiment.status === 'completed');
    const accuracyValues = completed
      .map(experiment => experiment.metrics?.accuracy)
      .filter(value => typeof value === 'number' && Number.isFinite(value));
    setStats({
      datasets: datasetsResult.data.length,
      experiments: experimentsResult.data.length,
      completed: completed.length,
      bestScore: accuracyValues.length ? Math.max(...accuracyValues) : null,
    });
  };

  const pollHealth = useCallback(async () => {
    setBackendOk(null);
    for (let retry = 0; retry <= 20; retry += 1) {
      try {
        await checkHealth();
        setBackendOk(true);
        refreshStats().catch(() => {});
        listSampleDatasets().then(r => setSampleDatasets(r.data)).catch(() => {});
        return;
      } catch {
        if (retry === 20) {
          setBackendOk(false);
          return;
        }
        await wait(3500);
      }
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    pollHealth();
    return () => {
      cancelled = true;
    };
  }, [pollHealth]);

  const runDemoMode = async () => {
    setDemoRunning(true);
    setDemoComplete(false);
    setDemoError('');
    setDemoProgress(2);
    setDemoMessage('Creating and loading the 120-row churn sample...');
    try {
      const uploaded = await uploadDataset(createDemoChurnFile());
      const dataset = uploaded.data;
      setSelectedDataset(dataset);
      setTargetColumn('churn');
      setTaskType('binary_classification');
      setDemoProgress(12);

      for (let index = 0; index < DEMO_MODEL_CONFIGS.length; index += 1) {
        const model = DEMO_MODEL_CONFIGS[index];
        const percentBase = 12 + index * 42;
        setDemoMessage(`Preprocessing and training ${model.model_id === 'logistic_regression' ? 'Logistic Regression' : 'Random Forest'}...`);
        const created = await createExperiment({
          dataset_id: dataset.id,
          target_column: 'churn',
          task_type: 'binary_classification',
          model_id: model.model_id,
          model_config: model.model_config,
          preprocessing_config: DEMO_PREPROCESSING,
          random_seed: 42,
        });
        let status;
        const deadline = Date.now() + 180000;
        do {
          await wait(500);
          const response = await getExperimentStatus(created.data.experiment_id);
          status = response.data;
          const trainingProgress = Number(status.progress || 0);
          setDemoProgress(Math.min(94, percentBase + Math.round(trainingProgress * 0.4)));
          if (status.status === 'failed') {
            throw new Error(status.message || `${model.model_id} training failed.`);
          }
          if (Date.now() > deadline) {
            throw new Error(`${model.model_id} training did not finish within 3 minutes.`);
          }
        } while (status.status !== 'completed');
      }

      await refreshStats().catch(() => {});
      setDemoProgress(100);
      setDemoMessage('Both demo models are trained and ready to compare.');
      setDemoComplete(true);
    } catch (error) {
      setDemoError(error.message || 'The demo workflow failed. Check that the backend is running and try again.');
      setDemoMessage('');
    } finally {
      setDemoRunning(false);
    }
  };

  const handleLoadSample = async (filename) => {
    setLoadingSample(filename);
    try {
      const r = await loadSampleDataset(filename);
      setSelectedDataset(r.data);
      navigate('/datasets');
    } catch (e) {
      setDemoError(`Failed to load sample: ${e.message}`);
    } finally {
      setLoadingSample(null);
    }
  };

  const handleDemoTimeSeries = async (filename) => {
    setLoadingSample(filename);
    try {
      const r = await loadSampleDataset(filename);
      setSelectedDataset(r.data);
      setTargetColumn('sales');
      setTaskType('time_series');
      setPreprocessingConfig(prev => ({
        ...prev,
        time_series_date_column: 'date',
        time_series_lags: [1, 7, 14],
        time_series_rolling_windows: [3, 7],
      }));
      navigate('/model-lab');
    } catch (e) {
      setDemoError(`Failed to load time-series demo: ${e.message}`);
    } finally {
      setLoadingSample(null);
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="page-title">Tabular AI Research Lab</h1>
          <p className="page-sub">A local research workspace to prepare tabular data, train models, compare results, explain predictions, and generate reports.</p>
        </div>
        <HelpTooltip text="Overview summarizes project activity and provides a guided demo of the end-to-end churn modeling workflow." />
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${backendOk === true ? 'bg-green-100 text-green-700' : backendOk === false ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>
            <div className={`w-2 h-2 rounded-full ${backendOk === true ? 'bg-green-500' : backendOk === false ? 'bg-red-500' : 'bg-amber-400'} animate-pulse`} />
            {backendOk === true ? 'Backend Online' : backendOk === false ? 'Backend Offline' : 'Waiting for the backend to start...'}
          </div>
          {backendOk === false && (
            <button
              onClick={() => pollHealth()}
              className="btn-secondary text-xs px-2.5 py-1"
            >
              Retry
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard icon={Database} label="Datasets" value={stats.datasets} color="blue" />
        <StatCard icon={FlaskConical} label="Experiments" value={stats.experiments} color="purple" />
        <StatCard icon={CheckCircle} label="Completed Runs" value={stats.completed} color="green" />
        <StatCard icon={Activity} label="Best Model Score (accuracy)" value={stats.bestScore == null ? '—' : `${(stats.bestScore * 100).toFixed(1)}%`} color="amber" />
      </div>

      <div className="card mb-8 border-blue-100 bg-blue-50/30">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="section-title mb-1">Try the complete workflow</h2>
            <p className="text-sm text-slate-600">Load a deterministic 120-row churn dataset, preprocess it, and train two baseline models automatically.</p>
          </div>
          <button onClick={runDemoMode} disabled={demoRunning} className="btn-primary shrink-0">
            <Play size={15} /> {demoRunning ? 'Demo running...' : 'Demo mode'}
          </button>
        </div>
        {(demoRunning || demoComplete) && (
          <div className="mt-4">
            <div className="progress-bar mb-2"><div className="progress-fill" style={{ width: `${demoProgress}%` }} /></div>
            <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-600">
              <span>{demoMessage}</span>
              {demoComplete && <button onClick={() => navigate('/experiments')} className="btn-secondary text-xs">View Experiments <ArrowRight size={13} /></button>}
            </div>
          </div>
        )}
        {demoError && <div className="error-box mt-4"><AlertCircle size={14} />{demoError}</div>}
      </div>

      <div className="card mb-8">
        <h2 className="section-title">Workflow</h2>
        <div className="flex flex-wrap items-center gap-2">
          {['Upload', 'Explore', 'Preprocess', 'Train', 'Compare', 'Explain', 'Predict', 'Report'].map((step, index) => (
            <React.Fragment key={step}>
              <span className="rounded-lg border border-blue-100 bg-blue-50 px-3 py-2 text-sm font-medium text-blue-800">{step}</span>
              {index < 7 && <ArrowRight size={15} className="text-slate-400" />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Workflow steps */}
      <div className="mb-8">
        <h2 className="section-title">🚀 Get Started — Workflow</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <StepCard num="1" title="Upload or Load Dataset"
            desc="Upload a CSV file or load one of the built-in sample datasets to get started immediately."
            done={!!selectedDataset} btnText="Go to Datasets" onClick={() => navigate('/datasets')} />
          <StepCard num="2" title="Explore & Profile Data"
            desc="Inspect columns, distributions, missing values, correlations, and data quality warnings."
            done={false} btnText="Open Data Explorer" onClick={() => navigate('/explorer')} />
          <StepCard num="3" title="Configure Preprocessing"
            desc="Set imputation strategy, scaling, encoding, feature exclusion, and train/test split."
            done={false} btnText="Configure Preprocessing" onClick={() => navigate('/preprocessing')} />
          <StepCard num="4" title="Train & Compare Models"
            desc="Select ML models, configure hyperparameters, and compare results side-by-side."
            done={false} btnText="Open Model Lab" onClick={() => navigate('/model-lab')} />
        </div>
      </div>

      {/* Sample datasets */}
      {sampleDatasets.length > 0 && (
        <div className="mb-8">
          <h2 className="section-title">📂 Sample Datasets</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {sampleDatasets.map(s => (
              <div key={s.filename} className="card-sm hover:shadow-card-md transition-shadow">
                <div className="font-medium text-sm text-slate-800 mb-1">{s.name}</div>
                <div className="text-xs text-slate-400 mb-3">{s.columns?.length} columns · {(s.size_bytes/1024).toFixed(1)} KB</div>
                {s.filename === 'daily_sales.csv' ? (
                  <div className="flex gap-2">
                    <button onClick={() => handleLoadSample(s.filename)} disabled={loadingSample === s.filename}
                      className="btn-secondary text-xs px-2 py-1.5 flex-1 justify-center">
                      {loadingSample === s.filename ? 'Loading...' : 'Load'}
                    </button>
                    <button onClick={() => handleDemoTimeSeries(s.filename)} disabled={loadingSample === s.filename}
                      className="btn-primary text-xs px-2 py-1.5 flex-1 justify-center bg-amber-600 hover:bg-amber-700 text-white">
                      Demo
                    </button>
                  </div>
                ) : (
                  <button onClick={() => handleLoadSample(s.filename)} disabled={loadingSample === s.filename}
                    className="btn-primary text-xs px-3 py-1.5 w-full justify-center">
                    {loadingSample === s.filename ? 'Loading...' : 'Load Dataset'}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Research context */}
      <div className="card border-blue-100 bg-blue-50/30">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center shrink-0">
            <BookOpen size={16} className="text-blue-600" />
          </div>
          <div>
            <div className="font-semibold text-slate-800 mb-1">Research Background</div>
            <p className="text-sm text-slate-600 leading-relaxed">
              This application is inspired by <strong>Borisov et al. (2024)</strong>, "Deep Neural Networks and Tabular Data: A Survey,"
              published in <em>IEEE Transactions on Neural Networks and Learning Systems</em>.
              The survey benchmarks dozens of deep learning architectures against gradient boosting on tabular datasets,
              finding that tree-based methods remain strong competitors while specialized architectures (TabNet, FT-Transformer)
              show promising results on certain datasets.
            </p>
            <a href="https://github.com/kathrinse/TabSurvey" target="_blank" rel="noopener noreferrer"
               className="text-blue-600 text-xs mt-2 inline-flex items-center gap-1 hover:underline">
              Reference Implementation Repository <ArrowRight size={12} />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
