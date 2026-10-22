import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, Database, Table2, Settings2, FlaskConical,
  BarChart3, Lightbulb, Zap, FileText, Settings, ChevronRight, Activity,
  Atom
} from 'lucide-react';

const navItems = [
  { path: '/', label: 'Overview', icon: LayoutDashboard },
  { path: '/datasets', label: 'Datasets', icon: Database },
  { path: '/explorer', label: 'Data Explorer', icon: Table2 },
  { path: '/preprocessing', label: 'Preprocessing', icon: Settings2 },
  { path: '/model-lab', label: 'Model Lab', icon: FlaskConical },
  { path: '/experiments', label: 'Experiments', icon: Activity },
  { path: '/explainability', label: 'Explainability', icon: Lightbulb },
  { path: '/predictions', label: 'Predictions', icon: Zap },
  { path: '/reports', label: 'Reports', icon: FileText },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <aside className="w-60 min-h-screen bg-[#0f172a] flex flex-col shrink-0">
      {/* Logo */}
      <div className="p-5 border-b border-slate-700/50">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
            <Atom size={18} className="text-white" />
          </div>
          <div>
            <div className="text-white font-bold text-sm leading-tight">Tabular AI</div>
            <div className="text-slate-400 text-[10px] leading-tight">Research Lab</div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {navItems.map(({ path, label, icon: Icon }) => {
          const isActive = path === '/'
            ? location.pathname === '/'
            : location.pathname.startsWith(path);
          return (
            <NavLink
              key={path}
              to={path}
              className={`nav-item ${isActive ? 'active' : ''}`}
            >
              <Icon size={16} className={isActive ? 'text-white' : 'text-slate-500'} />
              <span>{label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700/50">
        <div className="text-[10px] text-slate-500 leading-relaxed">
          Based on:<br />
          <span className="text-slate-400">Borisov et al., 2024</span><br />
          IEEE Trans. NNLS
        </div>
      </div>
    </aside>
  );
}
