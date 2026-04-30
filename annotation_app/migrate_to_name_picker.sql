-- =========================================================================
-- Migration: switch from GitHub OAuth (UUID user_id) to simple name picker
-- (text user_id like "heo", "swetha", etc.)
--
-- Run this in the Supabase SQL Editor ONCE.
-- =========================================================================

-- 1. Drop ALL RLS policies FIRST (they reference user_id columns)

drop policy if exists "event_chunks_select" on public.event_chunks;
drop policy if exists "event_chunks_insert" on public.event_chunks;

drop policy if exists "event_annotations_select" on public.event_annotations;
drop policy if exists "event_annotations_insert" on public.event_annotations;
drop policy if exists "event_annotations_update" on public.event_annotations;
drop policy if exists "event_annotations_delete" on public.event_annotations;

drop policy if exists "event_activity_select" on public.event_activity;
drop policy if exists "event_activity_insert" on public.event_activity;


-- 2. Now alter column types (safe after policies are gone)

-- event_annotations: drop FK + unique constraint, change column type
alter table public.event_annotations
  drop constraint if exists event_annotations_user_id_fkey;

alter table public.event_annotations
  drop constraint if exists event_annotations_chunk_key_user_id_key;

alter table public.event_annotations
  alter column user_id type text using user_id::text;

-- Re-add the unique constraint with the new text type
alter table public.event_annotations
  add constraint event_annotations_chunk_key_user_id_key unique (chunk_key, user_id);

-- event_activity: drop FK, change column type
alter table public.event_activity
  drop constraint if exists event_activity_user_id_fkey;

alter table public.event_activity
  alter column user_id type text using user_id::text;

-- event_chunks: drop FK on uploaded_by, change column type
alter table public.event_chunks
  drop constraint if exists event_chunks_uploaded_by_fkey;

alter table public.event_chunks
  alter column uploaded_by type text using uploaded_by::text;


-- 3. Create new open RLS policies (anon key can read/write)

create policy "event_chunks_select_open" on public.event_chunks
  for select using (true);

create policy "event_chunks_insert_open" on public.event_chunks
  for insert with check (true);

create policy "event_annotations_select_open" on public.event_annotations
  for select using (true);

create policy "event_annotations_insert_open" on public.event_annotations
  for insert with check (true);

create policy "event_annotations_update_open" on public.event_annotations
  for update using (true) with check (true);

create policy "event_activity_select_open" on public.event_activity
  for select using (true);

create policy "event_activity_insert_open" on public.event_activity
  for insert with check (true);
