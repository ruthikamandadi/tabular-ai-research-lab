import React from 'react';
import { CircleHelp } from 'lucide-react';

export default function HelpTooltip({ text }) {
  return (
    <button
      type="button"
      title={text}
      aria-label={`Help: ${text}`}
      className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-slate-500 hover:bg-slate-100 hover:text-slate-700"
    >
      <CircleHelp size={14} />
      Help
    </button>
  );
}
