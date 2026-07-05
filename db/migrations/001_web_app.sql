create extension if not exists pgcrypto;

create table if not exists profiles (
  id bigserial primary key,
  auth_user_id uuid unique,
  email text,
  display_name text,
  provider_region text not null default 'IN',
  onboarding_complete boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists catalog_items (
  movie_id integer primary key,
  media_type text not null default 'movie',
  title text not null,
  normalized_title text,
  genres text[] not null default '{}',
  release_year integer,
  imdb_id text,
  tmdb_id integer,
  poster_url text,
  backdrop_url text,
  overview text,
  runtime_minutes integer,
  enrichment_status text not null default 'pending',
  enrichment_confidence numeric,
  enrichment_error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_catalog_items_title on catalog_items using gin (to_tsvector('english', title));
create index if not exists idx_catalog_items_genres on catalog_items using gin (genres);
create index if not exists idx_catalog_items_tmdb on catalog_items (tmdb_id);
create index if not exists idx_catalog_items_imdb on catalog_items (imdb_id);

create table if not exists rating_events (
  id bigserial primary key,
  app_user_id bigint not null references profiles(id) on delete cascade,
  movie_id integer not null references catalog_items(movie_id) on delete cascade,
  rating numeric(2,1) not null check (rating >= 0.5 and rating <= 5.0),
  rated_at timestamptz not null,
  watched_at timestamptz,
  source text not null default 'organic',
  created_at timestamptz not null default now()
);

create index if not exists idx_rating_events_user_time on rating_events (app_user_id, rated_at desc);
create index if not exists idx_rating_events_movie on rating_events (movie_id);

create table if not exists current_ratings (
  app_user_id bigint not null references profiles(id) on delete cascade,
  movie_id integer not null references catalog_items(movie_id) on delete cascade,
  rating numeric(2,1) not null check (rating >= 0.5 and rating <= 5.0),
  rated_at timestamptz not null,
  watched_at timestamptz,
  source text not null default 'organic',
  revision bigint not null default 1,
  updated_at timestamptz not null default now(),
  primary key (app_user_id, movie_id)
);

create table if not exists watch_providers (
  movie_id integer not null references catalog_items(movie_id) on delete cascade,
  region text not null,
  provider_id integer,
  provider_name text not null,
  provider_logo_url text,
  provider_type text not null,
  display_priority integer,
  deep_link text,
  updated_at timestamptz not null default now(),
  primary key (movie_id, region, provider_name, provider_type)
);

create table if not exists watch_actions (
  movie_id integer not null references catalog_items(movie_id) on delete cascade,
  action_type text not null,
  label text not null,
  url text not null,
  region text,
  created_at timestamptz not null default now()
);

create unique index if not exists idx_watch_actions_unique
on watch_actions (movie_id, action_type, coalesce(region, 'GLOBAL'));

create or replace function set_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_profiles_updated_at on profiles;
create trigger trg_profiles_updated_at before update on profiles
for each row execute function set_updated_at();

drop trigger if exists trg_catalog_items_updated_at on catalog_items;
create trigger trg_catalog_items_updated_at before update on catalog_items
for each row execute function set_updated_at();
