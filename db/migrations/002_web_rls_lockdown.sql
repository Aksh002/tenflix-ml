-- TenFlix web tables are accessed through the FastAPI service, not directly
-- from the browser via Supabase PostgREST. Enable RLS on the exposed public
-- schema tables without adding browser-facing policies. This gives defense in
-- depth if the publishable/anon key is used against the generated API.

alter table if exists profiles enable row level security;
alter table if exists catalog_items enable row level security;
alter table if exists rating_events enable row level security;
alter table if exists current_ratings enable row level security;
alter table if exists watch_providers enable row level security;
alter table if exists watch_actions enable row level security;

revoke all on table profiles from anon, authenticated;
revoke all on table catalog_items from anon, authenticated;
revoke all on table rating_events from anon, authenticated;
revoke all on table current_ratings from anon, authenticated;
revoke all on table watch_providers from anon, authenticated;
revoke all on table watch_actions from anon, authenticated;

revoke all on sequence profiles_id_seq from anon, authenticated;
revoke all on sequence rating_events_id_seq from anon, authenticated;
