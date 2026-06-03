"use client";

import * as React from "react";
import { authApi, type AuthUser } from "@/lib/api";
import { clearToken, getToken, onTokenChange, setToken } from "./token";

type Status = "loading" | "authenticated" | "anonymous";

interface AuthContextValue {
  status: Status;
  user: AuthUser | null;
  /** True once the deployment's provider config is known. */
  ready: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, name?: string) => Promise<void>;
  logout: () => void;
  /** Adopt a token obtained out-of-band (OAuth callback / password reset). */
  adoptToken: (token: string) => Promise<void>;
  refresh: () => Promise<void>;
}

const AuthContext = React.createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = React.useState<Status>("loading");
  const [user, setUser] = React.useState<AuthUser | null>(null);
  const [ready, setReady] = React.useState(false);

  const loadMe = React.useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setStatus("anonymous");
      return;
    }
    try {
      const res = await authApi.me();
      setUser(res.user);
      setStatus("authenticated");
    } catch {
      clearToken();
      setUser(null);
      setStatus("anonymous");
    }
  }, []);

  React.useEffect(() => {
    void loadMe().finally(() => setReady(true));
    // Cross-tab / programmatic token clears (e.g. a 401) flip us to anonymous.
    return onTokenChange((t) => {
      if (!t) {
        setUser(null);
        setStatus("anonymous");
      }
    });
  }, [loadMe]);

  const login = React.useCallback(async (email: string, password: string) => {
    const res = await authApi.login({ email, password });
    setToken(res.access_token);
    setUser(res.user);
    setStatus("authenticated");
  }, []);

  const signup = React.useCallback(
    async (email: string, password: string, name?: string) => {
      const res = await authApi.signup({ email, password, name });
      setToken(res.access_token);
      setUser(res.user);
      setStatus("authenticated");
    },
    []
  );

  const adoptToken = React.useCallback(async (token: string) => {
    setToken(token);
    await loadMe();
  }, [loadMe]);

  const logout = React.useCallback(() => {
    void authApi.logout().catch(() => {});
    clearToken();
    setUser(null);
    setStatus("anonymous");
  }, []);

  const value: AuthContextValue = {
    status, user, ready, login, signup, logout, adoptToken, refresh: loadMe,
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = React.useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within <AuthProvider>");
  return ctx;
}
