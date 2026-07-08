"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { useAuth } from "@/components/auth-provider";
import { useToast } from "@/components/toast-provider";
import { isSupabaseConfigured } from "@/lib/env";
import { Nav } from "@/components/nav";

export default function LoginPage() {
  return (
    <Suspense fallback={<LoginShell callbackError={null} />}>
      <LoginContent />
    </Suspense>
  );
}

function LoginContent() {
  const searchParams = useSearchParams();
  return <LoginShell callbackError={searchParams.get("auth_error")} />;
}

function LoginShell({ callbackError }: { callbackError: string | null }) {
  const { signInWithEmail, status, user } = useAuth();
  const { notify } = useToast();
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");

  async function signIn() {
    if (!isSupabaseConfigured) {
      const detail =
        "Create apps/web/.env.local with NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY, then restart Next.js.";
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
    notify({
      tone: "success",
      title: "Magic link sent",
      detail: "Open the email on this browser to finish Supabase login."
    });
  }

  return (
    <main className="shell">
      <Nav />
      <section style={{ maxWidth: 760 }}>
        <p className="section-kicker">Supabase Auth</p>
        <h1 className="display" style={{ fontSize: "clamp(72px, 14vw, 180px)" }}>
          Enter the screening room.
        </h1>
        {status === "authenticated" ? (
          <p className="auth-note">
            Signed in as {user?.email ?? "a Supabase user"}. Use the Profile link to inspect your
            account and ratings.
          </p>
        ) : null}
        {callbackError ? <p className="auth-error">Auth callback failed: {callbackError}</p> : null}
        <div style={{ display: "flex", gap: 12, marginTop: 32 }}>
          <input
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="you@example.com"
            style={{
              flex: 1,
              border: "1px solid var(--color-obsidian-ink)",
              borderRadius: 10,
              background: "transparent",
              padding: "0 18px",
              minHeight: 54
            }}
          />
          <button className="voltage-button" disabled={!email} onClick={signIn} type="button">
            Send link →
          </button>
        </div>
        {message ? <p className="auth-note">{message}</p> : null}
      </section>
    </main>
  );
}
