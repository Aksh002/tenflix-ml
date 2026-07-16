import type { SupabaseClient } from "@supabase/supabase-js";

export type CatalogItem = {
  movie_id: number;
  title: string;
  genres: string[];
  release_year: number | null;
  media_type: string;
  poster_url?: string | null;
  backdrop_url?: string | null;
  overview?: string | null;
  user_rating?: number | null;
  watch_providers?: WatchProvider[];
  watch_actions?: WatchAction[];
};

export type WatchProvider = {
  provider_name: string;
  provider_type: string;
  region: string;
  provider_logo_url?: string | null;
  deep_link?: string | null;
};

export type WatchAction = {
  action_type: string;
  label: string;
  url: string;
  region?: string | null;
};

export type CatalogRow = {
  slug: string;
  title: string;
  subtitle: string;
  items: CatalogItem[];
};

export type Recommendation = {
  movie_id: number;
  title: string;
  genres: string[];
  release_year: number | null;
  score: number;
  rank: number;
  reason: string;
  score_contributors: Record<string, number>;
};

const API_URL = process.env.NEXT_PUBLIC_TENFLIX_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function readableApiError(status: number, body: string) {
  if (body) {
    try {
      const parsed = JSON.parse(body) as { detail?: unknown };
      if (typeof parsed.detail === "string") return parsed.detail;
      if (Array.isArray(parsed.detail)) return "The request was rejected by validation.";
    } catch {
      return body;
    }
  }
  if (status === 401) return "Your session is missing or expired. Sign in again.";
  if (status === 404) return "The requested TenFlix resource was not found.";
  if (status >= 500) return "The TenFlix API had an internal error. Check the backend logs.";
  return `Request failed with status ${status}.`;
}

export async function apiFetch<T>(
  supabase: SupabaseClient,
  path: string,
  init: RequestInit = {},
  accessToken?: string | null
): Promise<T> {
  const fallback = accessToken === undefined ? await supabase.auth.getSession() : null;
  const token = accessToken === undefined ? fallback?.data.session?.access_token : accessToken;
  if (!token) {
    throw new ApiError(401, "No Supabase session is available. Sign in again.");
  }
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...init.headers
    }
  });
  if (!response.ok) {
    const body = await response.text();
    throw new ApiError(response.status, readableApiError(response.status, body));
  }
  return response.json() as Promise<T>;
}

export async function recordRating(
  supabase: SupabaseClient,
  movieId: number,
  rating: number,
  accessToken?: string | null
) {
  return apiFetch(supabase, "/v1/ratings", {
    method: "POST",
    body: JSON.stringify({
      movie_id: movieId,
      rating,
      rated_at: new Date().toISOString(),
      source: "organic"
    })
  }, accessToken);
}
