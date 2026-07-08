"use client";

import Link from "next/link";
import { useAuth } from "@/components/auth-provider";

export function AuthGate({
  children,
  label = "This room is private",
  detail = "Sign in to load your ratings, recommendations and catalog state."
}: {
  children: React.ReactNode;
  label?: string;
  detail?: string;
}) {
  const { accessToken, status } = useAuth();

  if (status === "loading") {
    return (
      <section className="state-panel">
        <p className="section-kicker">Session check</p>
        <h2>Reading the ticket.</h2>
        <p>TenFlix is checking your Supabase session before loading private data.</p>
      </section>
    );
  }

  if (status === "anonymous") {
    return (
      <section className="state-panel">
        <p className="section-kicker">Login required</p>
        <h2>{label}</h2>
        <p>{detail}</p>
        <Link className="voltage-button" href="/login">
          Sign in →
        </Link>
      </section>
    );
  }

  if (!accessToken) {
    return (
      <section className="state-panel" data-tone="error">
        <p className="section-kicker">Session incomplete</p>
        <h2>Signed in, but no bearer token is available.</h2>
        <p>
          Clear site data for this localhost origin, sign in again, and make sure the magic link
          opens on the same host you use for the app.
        </p>
      </section>
    );
  }

  return children;
}
