import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import axios from "axios";

const AuthContext = createContext(null);

const authApi = axios.create({ baseURL: "/api", withCredentials: true, timeout: 30000 });

authApi.interceptors.response.use(
  (res) => res,
  (err) => {
    const detail = err.response?.data?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
        ? detail.map((d) => d?.msg || String(d)).join("; ")
        : err.response?.data?.message || err.message;
    const normalizedError = new Error(message || "An unexpected error occurred.");
    normalizedError.response = err.response;
    return Promise.reject(normalizedError);
  }
);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(undefined);
  const [authLoading, setAuthLoading] = useState(true);

  const checkSession = useCallback(async () => {
    try {
      const res = await authApi.get("/auth/me");
      setUser(res.data.user);
    } catch {
      setUser(null);
    } finally {
      setAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    checkSession();
  }, [checkSession]);

  useEffect(() => {
    const handleUnauthorized = () => {
      setUser(null);
    };
    window.addEventListener("auth:unauthorized", handleUnauthorized);
    return () => window.removeEventListener("auth:unauthorized", handleUnauthorized);
  }, []);

  const login = useCallback(async (email, password) => {
    const res = await authApi.post("/auth/login", { email, password });
    setUser(res.data.user);
    return res.data.user;
  }, []);

  const register = useCallback(async (name, email, password, confirm_password) => {
    const res = await authApi.post("/auth/register", { name, email, password, confirm_password });
    setUser(res.data.user);
    return res.data.user;
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.post("/auth/logout");
    } catch {
      // ignore
    }
    setUser(null);
  }, []);

  const value = { user, authLoading, login, register, logout, checkSession };
  return React.createElement(AuthContext.Provider, { value }, children);
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
