"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { AuthGate } from "@/components/auth-gate";
import { useAuth } from "@/components/auth-provider";
import { Nav } from "@/components/nav";
import { useToast } from "@/components/toast-provider";
import { WatchActions } from "@/components/watch-actions";
import { apiFetch, CatalogItem, recordRating } from "@/lib/api";

export default function MovieDetailPage() {
  const params = useParams<{ id: string }>();
  const { accessToken, status, supabase, user } = useAuth();
  const { notify } = useToast();
  const queryClient = useQueryClient();
  const movie = useQuery({
    queryKey: ["movie", params.id, user?.id ?? "anonymous"],
    queryFn: () => apiFetch<CatalogItem>(supabase, `/v1/catalog/${params.id}`, {}, accessToken),
    enabled: status === "authenticated" && Boolean(accessToken)
  });
  const rate = useMutation({
    mutationFn: (rating: number) =>
      recordRating(supabase, Number(params.id), rating, accessToken),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["movie", params.id, user?.id ?? "anonymous"] });
      notify({ tone: "success", title: "Rating recorded", detail: "This title now informs your profile." });
    },
    onError: (error) => {
      notify({
        tone: "error",
        title: "Could not save rating",
        detail: error instanceof Error ? error.message : "Try again after checking the API."
      });
    }
  });

  const item = movie.data;
  return (
    <main className="shell">
      <Nav />
      <AuthGate label="Movie details need a signed-in profile">
      {item ? (
        <section className="grid-asym" style={{ alignItems: "start" }}>
          <div className="poster-tile" style={{ gridColumn: "1 / span 5", minHeight: 620 }}>
            {item.poster_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={item.poster_url} alt="" />
            ) : null}
            <div className="poster-copy">
              <p>{item.genres.join(" / ")}</p>
            </div>
          </div>
          <div style={{ gridColumn: "7 / span 6" }}>
            <p className="section-kicker">{item.release_year ?? "Year unknown"}</p>
            <h1 className="display" style={{ fontSize: "clamp(64px, 8vw, 138px)" }}>
              {item.title}
            </h1>
            <p style={{ fontSize: 18, lineHeight: 1.45, maxWidth: 620 }}>
              {item.overview ?? "No overview yet. Rate it if it belongs in your taste profile."}
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 30 }}>
              {[1, 2, 3, 4, 5].map((value) => (
                <button className="voltage-button" key={value} onClick={() => rate.mutate(value)}>
                  {value}.0
                </button>
              ))}
            </div>
            {item.user_rating ? (
              <p style={{ color: "var(--color-sage)" }}>Your rating: {item.user_rating}</p>
            ) : null}
            <WatchActions providers={item.watch_providers} actions={item.watch_actions} />
          </div>
        </section>
      ) : movie.isLoading ? (
        <section className="state-panel">
          <p className="section-kicker">Film detail</p>
          <h2>Loading the title card.</h2>
          <p>Fetching metadata, your rating and watch actions.</p>
        </section>
      ) : (
        <section className="state-panel" data-tone="error">
          <p className="section-kicker">Movie unavailable</p>
          <h2>This title did not load.</h2>
          <p>{movie.error instanceof Error ? movie.error.message : "Check the API and catalog data."}</p>
        </section>
      )}
      </AuthGate>
    </main>
  );
}
