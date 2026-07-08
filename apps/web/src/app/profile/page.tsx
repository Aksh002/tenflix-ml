"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useAuth } from "@/components/auth-provider";
import { Nav } from "@/components/nav";
import { useToast } from "@/components/toast-provider";
import { apiFetch } from "@/lib/api";
import { isSupabaseConfigured } from "@/lib/env";

type Rating = {
  movie_id: number;
  title: string;
  rating: number;
  rated_at: string;
  source: string;
};

export default function ProfilePage() {
  const { accessToken, status, supabase, user, signInWithEmail } = useAuth();
  const { notify } = useToast();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const me = useQuery({
    queryKey: ["me", user?.id ?? "anonymous"],
    queryFn: () => apiFetch<Record<string, unknown>>(supabase, "/v1/me", {}, accessToken),
    enabled: status === "authenticated" && Boolean(accessToken)
  });
  const ratings = useQuery({
    queryKey: ["ratings-me", user?.id ?? "anonymous"],
    queryFn: () => apiFetch<{ ratings: Rating[] }>(supabase, "/v1/ratings/me", {}, accessToken),
    enabled: status === "authenticated" && Boolean(accessToken)
  });

  async function signIn() {
    if (!isSupabaseConfigured) {
      const detail = "Add apps/web/.env.local and restart Next.js.";
      setMessage(detail);
      notify({ tone: "error", title: "Supabase frontend config missing", detail });
      return;
    }
    const { error } = await signInWithEmail(email, "/profile");
    if (error) {
      setMessage(error);
      notify({ tone: "error", title: "Could not send login link", detail: error });
      return;
    }
    setMessage("Check your email for the TenFlix magic link.");
    notify({ tone: "success", title: "Magic link sent", detail: "Return here after opening it." });
  }

  return (
    <main className="shell">
      <Nav />
      <p className="section-kicker">Your signal</p>
      <h1 className="display" style={{ fontSize: "clamp(72px, 12vw, 176px)" }}>
        A public ledger of taste.
      </h1>
      {status === "loading" ? <p style={{ color: "var(--color-sage)" }}>Checking session…</p> : null}
      {status === "anonymous" ? (
        <section className="auth-panel">
          <div>
            <p className="section-kicker">Login required</p>
            <h2>Claim your taste ledger.</h2>
            <p>
              Enter your email and TenFlix will send a Supabase magic link. After login, your
              ratings and recommendations will appear here.
            </p>
          </div>
          <div className="auth-form">
            <input
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="you@example.com"
              type="email"
            />
            <button className="voltage-button" disabled={!email} onClick={signIn} type="button">
              Send link →
            </button>
            {message ? <p>{message}</p> : null}
          </div>
        </section>
      ) : null}
      {status === "authenticated" ? (
        <>
      <section className="profile-strip">
        <span>Signed in</span>
        <strong>{String(me.data?.email ?? user?.email ?? "")}</strong>
        <span>Region: {String(me.data?.provider_region ?? "IN")}</span>
      </section>
      <section style={{ marginTop: 60 }}>
        {ratings.isLoading ? <p>Loading your ratings…</p> : null}
        {ratings.isError ? (
          <p style={{ color: "var(--color-sage)" }}>
            Could not load ratings. Check that the backend is running and your Supabase token is
            valid.
          </p>
        ) : null}
        {ratings.data?.ratings.map((rating) => (
          <article
            key={rating.movie_id}
            style={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: 20,
              borderTop: "1px solid var(--color-mist)",
              padding: "18px 0"
            }}
          >
            <span>{rating.title}</span>
            <strong>{rating.rating.toFixed(1)}</strong>
          </article>
        ))}
        {ratings.data?.ratings.length === 0 ? (
          <p style={{ color: "var(--color-sage)" }}>
            No ratings yet. Rate a few films from Discover to build your recommendation profile.
          </p>
        ) : null}
      </section>
        </>
      ) : null}
    </main>
  );
}
