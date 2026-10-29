import React, { useState, useEffect, useCallback, useRef } from "react";
import { AlertCircle } from "lucide-react";
import { checkHealth } from "../services/api";

export default function ServerWakeNotice() {
  const [status, setStatus] = useState("waking"); // 'waking' | 'ready' | 'unreachable'
  const isMountedRef = useRef(true);
  const startTimeRef = useRef(Date.now());
  const timerRef = useRef(null);
  const timeoutRef = useRef(null);

  const check = useCallback(async () => {
    try {
      await checkHealth();
      if (isMountedRef.current) {
        setStatus("ready");
      }
    } catch {
      if (!isMountedRef.current) return;
      const elapsed = Date.now() - startTimeRef.current;
      if (elapsed >= 120000) {
        setStatus("unreachable");
      } else {
        timerRef.current = setTimeout(check, 3000);
      }
    }
  }, []);

  const startCheck = useCallback(() => {
    startTimeRef.current = Date.now();
    setStatus("waking");
    clearTimeout(timerRef.current);
    clearTimeout(timeoutRef.current);

    timeoutRef.current = setTimeout(() => {
      if (isMountedRef.current) {
        setStatus((prev) => (prev === "ready" ? "ready" : "unreachable"));
      }
    }, 120000);

    check();
  }, [check]);

  useEffect(() => {
    isMountedRef.current = true;
    startCheck();
    return () => {
      isMountedRef.current = false;
      clearTimeout(timerRef.current);
      clearTimeout(timeoutRef.current);
    };
  }, [startCheck]);

  if (status === "ready") {
    return null;
  }

  if (status === "unreachable") {
    return (
      <div className="flex items-center justify-between gap-2 p-3 mb-4 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-300 text-xs">
        <div className="flex items-center gap-2">
          <AlertCircle size={14} className="shrink-0" />
          <span>The server is not reachable. Please try again in a minute.</span>
        </div>
        <button
          type="button"
          onClick={startCheck}
          className="px-2.5 py-1 bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/40 rounded text-xs transition-all font-medium shrink-0"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2.5 p-3 mb-4 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-300 text-xs">
      <div className="w-3.5 h-3.5 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin shrink-0" />
      <span>Waking up the server. This can take up to a minute the first time.</span>
    </div>
  );
}
