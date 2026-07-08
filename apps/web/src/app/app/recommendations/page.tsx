"use client";

import { useQuery } from "@tanstack/react-query";
import { AuthGate } from "@/components/auth-gate";
import { useAuth } from "@/components/auth-provider";
import { MovieCard } from "@/components/movie-card";
import { Nav } from "@/components/nav";
import { apiFetch, Recommendation } from "@/lib/api";

export default function RecommendationsPage() {
  const { accessToken, status, supabase, user } = useAuth();
  const recommendations = useQuery({
    queryKey: ["recommendations", user?.id ?? "anonymous"],
    queryFn: () =>
      apiFetch<{ recommendations: Recommendation[] }>(
        supabase,
        "/v1/recommendations/me?top_k=10",
        {},
        accessToken
      ),
    enabled: status === "authenticated" && Boolean(accessToken)
  });

  return (
    <main className="shell">
      <Nav />
      <p className="section-kicker">Assembled from your ratings</p>
      <h1 className="display" style={{ fontSize: "clamp(72px, 12vw, 176px)" }}>
        Your top ten.
      </h1>
      <AuthGate label="Recommendations need your signed-in ratings">
      {recommendations.isLoading ? (
        <section className="state-panel">
          <p className="section-kicker">Taste assembly</p>
          <h2>Building the board.</h2>
          <p>TenFlix is folding in your live ratings and ranking candidates.</p>
        </section>
      ) : null}
      {recommendations.error ? (
        <section className="state-panel" data-tone="error">
          <p className="section-kicker">Recommendations unavailable</p>
          <h2>The model did not respond cleanly.</h2>
          <p>
            {recommendations.error instanceof Error
              ? recommendations.error.message
              : "Rate more films or check the API."}
          </p>
        </section>
      ) : null}
      {recommendations.data?.recommendations.length === 0 ? (
        <section className="state-panel">
          <p className="section-kicker">No results</p>
          <h2>The board is empty.</h2>
          <p>Rate at least a few films in Discover, then regenerate recommendations.</p>
        </section>
      ) : null}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 22,
          marginTop: 60
        }}
      >
        {recommendations.data?.recommendations.map((item) => (
          <MovieCard key={item.movie_id} item={item} />
        ))}
      </div>
      </AuthGate>
    </main>
  );
}
