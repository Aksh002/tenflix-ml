"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { AuthGate } from "@/components/auth-gate";
import { useAuth } from "@/components/auth-provider";
import { MovieCard } from "@/components/movie-card";
import { Nav } from "@/components/nav";
import { RatingDock } from "@/components/rating-dock";
import { useToast } from "@/components/toast-provider";
import { apiFetch, CatalogItem, CatalogRow, recordRating } from "@/lib/api";

export default function DiscoverPage() {
  const { accessToken, status, supabase, user } = useAuth();
  const { notify } = useToast();
  const queryClient = useQueryClient();
  const [activeItem, setActiveItem] = useState<CatalogItem | null>(null);
  const rows = useQuery({
    queryKey: ["catalog-rows", user?.id ?? "anonymous"],
    queryFn: () =>
      apiFetch<{ rows: CatalogRow[] }>(supabase, "/v1/catalog/rows", {}, accessToken),
    enabled: status === "authenticated" && Boolean(accessToken)
  });
  const rate = useMutation({
    mutationFn: (rating: number) => {
      if (!activeItem) throw new Error("No active movie");
      return recordRating(supabase, activeItem.movie_id, rating, accessToken);
    },
    onSuccess: () => {
      setActiveItem(null);
      queryClient.invalidateQueries({ queryKey: ["catalog-rows", user?.id ?? "anonymous"] });
      notify({ tone: "success", title: "Rating recorded", detail: "Your profile updated instantly." });
    },
    onError: (error) => {
      notify({
        tone: "error",
        title: "Could not save rating",
        detail: error instanceof Error ? error.message : "Try again after checking the API."
      });
    }
  });

  return (
    <main className="shell">
      <Nav />
      <section>
        <p className="section-kicker">Discovery board</p>
        <h1 className="display" style={{ fontSize: "clamp(76px, 13vw, 196px)" }}>
          TenFlix learns in public.
        </h1>
      </section>
      <AuthGate label="Discover needs a signed-in profile">
      {rows.isLoading ? <section className="state-panel"><p className="section-kicker">Catalog</p><h2>Loading the shelves.</h2><p>Fetching curated rows from the TenFlix API.</p></section> : null}
      {rows.error ? (
        <section className="state-panel" data-tone="error">
          <p className="section-kicker">Catalog unavailable</p>
          <h2>The shelves did not load.</h2>
          <p>{rows.error instanceof Error ? rows.error.message : "Check that the TenFlix API is running."}</p>
        </section>
      ) : null}
      {rows.data?.rows.length === 0 ? (
        <section className="state-panel">
          <p className="section-kicker">Empty catalog</p>
          <h2>No films are in the database yet.</h2>
          <p>Run `tenflix-v4 enrich-catalog --movies movies.csv --links links.csv` and refresh.</p>
        </section>
      ) : null}
      {rows.data?.rows.map((row) => (
        <section key={row.slug} style={{ marginTop: 90 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "end" }}>
            <div>
              <p className="section-kicker">{row.subtitle}</p>
              <h2 style={{ margin: 0, fontSize: 38 }}>{row.title}</h2>
            </div>
            <span style={{ color: "var(--color-sage)" }}>Drag a card ↓</span>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))",
              gap: 20,
              marginTop: 22
            }}
          >
            {row.items.map((item) => (
              <MovieCard
                key={item.movie_id}
                item={item}
                draggable
                onDragStart={setActiveItem}
                onDragEnd={() => undefined}
              />
            ))}
          </div>
        </section>
      ))}
      <RatingDock activeItem={activeItem} onRate={(value) => rate.mutate(value)} />
      </AuthGate>
    </main>
  );
}
