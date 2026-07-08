"use client";

import type { Session, SupabaseClient, User } from "@supabase/supabase-js";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { createClient } from "@/lib/supabase/client";

type AuthStatus = "loading" | "authenticated" | "anonymous";

type AuthContextValue = {
  supabase: SupabaseClient;
  session: Session | null;
  user: User | null;
  accessToken: string | null;
  status: AuthStatus;
  signInWithEmail: (email: string, next?: string) => Promise<{ error?: string }>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const supabase = useMemo(() => createClient(), []);
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<AuthStatus>("loading");

  useEffect(() => {
    let mounted = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session);
      setStatus(data.session?.access_token ? "authenticated" : "anonymous");
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      setStatus(nextSession?.access_token ? "authenticated" : "anonymous");
    });
    return () => {
      mounted = false;
      listener.subscription.unsubscribe();
    };
  }, [supabase]);

  const signInWithEmail = useCallback(
    async (email: string, next = "/app") => {
      const redirect = new URL("/auth/callback", window.location.origin);
      redirect.searchParams.set("next", next);
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: { emailRedirectTo: redirect.toString() }
      });
      return { error: error?.message };
    },
    [supabase]
  );

  const signOut = useCallback(async () => {
    await supabase.auth.signOut();
    setSession(null);
    setStatus("anonymous");
  }, [supabase]);

  const value = useMemo(
    () => ({
      supabase,
      session,
      user: session?.user ?? null,
      accessToken: session?.access_token ?? null,
      status,
      signInWithEmail,
      signOut
    }),
    [session, signInWithEmail, signOut, status, supabase]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return context;
}
