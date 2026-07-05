-- Query and foreign-key support indexes for the TenFlix web repository.
-- Postgres automatically indexes primary keys and unique constraints, but not
-- every foreign-key or common multi-column access pattern.

create index if not exists idx_current_ratings_movie
on current_ratings (movie_id);

create index if not exists idx_current_ratings_user_rated_at
on current_ratings (app_user_id, rated_at desc);

create index if not exists idx_catalog_items_media_release_title
on catalog_items (media_type, release_year desc nulls last, title);

create index if not exists idx_catalog_items_release_year
on catalog_items (release_year);

create index if not exists idx_watch_providers_lookup
on watch_providers (movie_id, region, display_priority, provider_name);

create index if not exists idx_watch_actions_lookup
on watch_actions (movie_id, region, action_type);
