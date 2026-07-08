"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/auth-provider";
import { useToast } from "@/components/toast-provider";

export function Nav() {
  const router = useRouter();
  const { status, signOut } = useAuth();
  const { notify } = useToast();

  async function logout() {
    await signOut();
    notify({ tone: "info", title: "Signed out", detail: "Your local session has been cleared." });
    router.push("/login");
    router.refresh();
  }

  return (
    <nav className="nav">
      <Link className="brand" href="/">
        Ten<span>Flix</span>
      </Link>
      <div style={{ display: "flex", gap: 22, alignItems: "center" }}>
        <Link href="/app">Discover</Link>
        <Link href="/app/recommendations">Recommendations</Link>
        <Link href="/profile">Profile</Link>
        {status === "authenticated" ? (
          <button className="nav-action" onClick={logout} type="button">
            Logout
          </button>
        ) : (
          <Link className="nav-action" href="/login">
            Login
          </Link>
        )}
        <span style={{ color: "var(--color-voltage)", fontWeight: 550 }}>||</span>
      </div>
    </nav>
  );
}
