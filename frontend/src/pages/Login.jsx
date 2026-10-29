import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";
import { Atom, AlertCircle, Eye, EyeOff } from "lucide-react";
import ServerWakeNotice from "../components/ServerWakeNotice";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState("");

  const isNetworkOrTimeoutError = (err) => {
    if (!err) return false;
    if (err.code === "ECONNABORTED" || err.code === "ERR_NETWORK") return true;
    const msg = (err.message || "").toLowerCase();
    if (msg.includes("timeout") || msg.includes("network") || msg.includes("starting")) return true;
    if (!err.response) return true;
    const status = err.response?.status;
    return status === 502 || status === 503 || status === 504;
  };

  const sanitizeError = (err) => {
    const msg = err?.message || "Login failed.";
    if (msg.toLowerCase().includes("timeout")) {
      return "Server request timed out. Please try again.";
    }
    return msg;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading) return;
    setError("");
    setStatusMessage("");
    setLoading(true);

    try {
      await login(email, password);
      navigate("/", { replace: true });
    } catch (err) {
      if (isNetworkOrTimeoutError(err)) {
        setStatusMessage("Server is starting, please wait...");
        await new Promise((resolve) => setTimeout(resolve, 5000));
        try {
          await login(email, password);
          navigate("/", { replace: true });
          return;
        } catch (retryErr) {
          setStatusMessage("");
          setError(sanitizeError(retryErr));
        } finally {
          setLoading(false);
        }
      } else {
        setError(sanitizeError(err));
        setLoading(false);
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-blue-600 rounded-2xl mb-4 shadow-lg shadow-blue-900/40">
            <Atom size={28} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Tabular AI Research Lab</h1>
          <p className="text-slate-400 text-sm mt-1">Sign in to your research workspace</p>
        </div>

        <div className="bg-white/5 backdrop-blur border border-white/10 rounded-2xl p-8 shadow-xl">
          <h2 className="text-lg font-semibold text-white mb-6">Sign In</h2>
          <ServerWakeNotice />
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Email address</label>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2.5 bg-white/10 border border-white/20 rounded-lg text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">Password</label>
              <div className="relative">
                <input
                  type={showPw ? "text" : "password"}
                  required
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-3 py-2.5 pr-10 bg-white/10 border border-white/20 rounded-lg text-white placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="Your password"
                />
                <button type="button" onClick={() => setShowPw(!showPw)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200">
                  {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </div>
            </div>

            {statusMessage && (
              <div className="flex items-center gap-2 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-300 text-sm">
                <div className="w-3.5 h-3.5 border-2 border-blue-400/30 border-t-blue-400 rounded-full animate-spin shrink-0" />
                <span>{statusMessage}</span>
              </div>
            )}

            {error && (
              <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
                <AlertCircle size={14} className="shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button type="submit" disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-2.5 rounded-lg transition-all text-sm mt-2 flex items-center justify-center gap-2">
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  {statusMessage ? "Waiting for server..." : "Signing in..."}
                </>
              ) : "Sign In"}
            </button>
          </form>

          <p className="text-center text-sm text-slate-400 mt-6">
            Don&apos;t have an account?{" "}
            <Link to="/register" className="text-blue-400 hover:text-blue-300 font-medium">Create one</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
