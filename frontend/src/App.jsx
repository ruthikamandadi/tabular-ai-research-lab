import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Overview from './pages/Overview';
import Datasets from './pages/Datasets';
import DataExplorer from './pages/DataExplorer';
import Preprocessing from './pages/Preprocessing';
import ModelLab from './pages/ModelLab';
import Experiments from './pages/Experiments';
import Explainability from './pages/Explainability';
import Predictions from './pages/Predictions';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import { AppProvider } from './contexts/AppContext';

function Layout({ children }) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-surface overflow-y-auto">
        <div className="p-8 max-w-7xl mx-auto fade-in">
          {children}
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/explorer" element={<DataExplorer />} />
            <Route path="/preprocessing" element={<Preprocessing />} />
            <Route path="/model-lab" element={<ModelLab />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Layout>
      </AppProvider>
    </BrowserRouter>
  );
}
