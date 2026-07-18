# TenFlix Web App Upgrade Runbook

This layer turns the promoted V4 model into a full-stack web app:

- FastAPI product API with Supabase Auth/Postgres repositories.
- Next.js App Router frontend in `apps/web`.
- TMDb enrichment for posters, IMDb/TMDb IDs, watch providers and Stremio actions.

## Backend setup

1. Install Python dependencies with the web runtime extra:

   ```powershell
   python -m pip install -e ".[dev]"
   ```

   If you only need to run the web API without development tools:

   ```powershell
   python -m pip install -e ".[web]"
   ```

2. Copy environment files:

   ```powershell
   Copy-Item .env.example .env
   Copy-Item apps\web\.env.example apps\web\.env.local
   ```

   For local-only backend secrets, root `.env.local` is also supported and
   overrides values loaded from root `.env`. Next.js still reads frontend values
   from `apps/web/.env.local`.

3. Fill these values:

   ```text
   DATABASE_URL=
   SUPABASE_URL=
   SUPABASE_ANON_KEY=
   SUPABASE_JWT_SECRET=
   TMDB_API_TOKEN=
   TENFLIX_V4_RUN=artifacts/movielens-v4-tuned-eval-v4-production
   ```

4. Apply and verify the web-app database schema:

   ```powershell
   tenflix-v4 migrate-web-db
   tenflix-v4 check-web-db
   ```

   By default this applies every sorted SQL file in `db/migrations`. It creates
   the required `profiles`, `catalog_items`, `rating_events`, `current_ratings`,
   `watch_providers` and `watch_actions` tables in the database pointed to by
   `DATABASE_URL`, then enables RLS and revokes direct anon/authenticated table
   access. The browser talks to these tables through FastAPI, not through
   Supabase PostgREST.

5. Ingest catalog without TMDb:

   ```powershell
   tenflix-v4 enrich-catalog --movies movies.csv
   ```

   If `links.csv` is available:

   ```powershell
   tenflix-v4 enrich-catalog --movies movies.csv --links links.csv
   ```

6. Enrich with TMDb when `TMDB_API_TOKEN` is configured:

   ```powershell
   tenflix-v4 enrich-catalog --movies movies.csv --links links.csv --with-tmdb --region IN
   ```

7. Run the product API:

   ```powershell
   tenflix-v4 serve-web --run artifacts/movielens-v4-tuned-eval-v4-production
   ```

For local UI work without Supabase JWTs, set:

```text
TENFLIX_DEV_AUTH_USER_ID=<a uuid>
TENFLIX_DEV_AUTH_EMAIL=dev@tenflix.local
```

## Frontend setup

Install dependencies:

```powershell
npm install
```

Run the app:

```powershell
npm run web:dev
```

Open:

```text
http://localhost:3000
```

## Current UX

- Brutalist/editorial landing page based on `skills/referoUI_v1/DESIGN.md`.
- Supabase magic-link login.
- Movie discovery rows from `/v1/catalog/rows`.
- Drag-to-rate cards with a rising five-bin rating dock.
- Recommendation board from `/v1/recommendations/me`.
- Movie detail page with rating buttons, provider buttons and Stremio/IMDb/TMDb actions when enriched.

## Notes

- The V4 model artifact remains immutable.
- Live ratings are stored in Supabase/Postgres.
- Stremio is an external launch action only; TenFlix does not provide streams.
- Series support is schema-ready through `media_type`, but V1 imports movies first.
