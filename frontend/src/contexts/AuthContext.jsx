import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import { getStoredToken, setStoredToken, clearStoredToken } from "../services/api";

const AuthContext = createContext(null);

const API_BASE = (import.meta.env.VITE_API_URL || "") + "/api";

const authApi = axios.create({ baseURL: API_BASE, withCredentials: true, timeout: 30000 });

authApi.interceptors.request.use((config) => {
  const token = getStoredToken();
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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
  const isLoggingInRef = useRef(false);

  const checkSession = useCallback(async () => {
    const token = getStoredToken();
    if (!token) {
      setUser(null);
      setAuthLoading(false);
      return;
    }
    try {
      const res = await authApi.get("/auth/me");
      setUser(res.data.user);
    } catch (err) {
      if (err.response?.status === 401) {
        clearStoredToken();
      }
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
      // Do not log out on a 401 from any other request while the login is still in progress
      if (isLoggingInRef.current) {
        return;
      }
      clearStoredToken();
      setUser(null);
    };
    window.addEventListener("auth:unauthorized", handleUnauthorized);
    return () => window.removeEventListener("auth:unauthorized", handleUnauthorized);
  }, []);

  const login = useCallback(async (email, password) => {
    isLoggingInRef.current = true;
    try {
      const res = await authApi.post("/auth/login", { email, password });
      if (res.data?.token) {
        setStoredToken(res.data.token);
      }
      setUser(res.data.user);
      return res.data.user;
    } finally {
      isLoggingInRef.current = false;
    }
  }, []);

  const register = useCallback(async (name, email, password, confirm_password) => {
    isLoggingInRef.current = true;
    try {
      const res = await authApi.post("/auth/register", { name, email, password, confirm_password });
      if (res.data?.token) {
        setStoredToken(res.data.token);
      }
      setUser(res.data.user);
      return res.data.user;
    } finally {
      isLoggingInRef.current = false;
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.post("/auth/logout");
    } catch {
      // ignore
    } finally {
      clearStoredToken();
      setUser(null);
    }
  }, []);

  const value = { user, authLoading, login, register, logout, checkSession };
  return React.createElement(AuthContext.Provider, { value }, children);
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
