-- =========================================================================
-- Supabase schema for collaborative Event Graph annotation
-- Run this in the Supabase SQL editor (alongside the existing schema.sql
-- if you also use the Fact/Non-Fact dashboard).
-- =========================================================================

-- 1. Chunks table — stores the chunk metadata so every annotator sees
--    the same canonical list without needing local JSONL files.
create table if not exists public.event_chunks (
  id            uuid primary key default gen_random_uuid(),
  source_file   text not null,              -- e.g. "chunk_event_graphs_claude-opus-4-1.jsonl"
  doc_title     text not null,
  chunk_id      text not null,
  chunk_key     text not null unique,       -- "doc_title::chunk_id"
  chunk_theme   text,
  chunk_text    text not null,
  events_json   jsonb default '[]'::jsonb,  -- extracted events (read-only reference)
  entities_json jsonb default '[]'::jsonb,  -- extracted entities
  metadata      jsonb default '{}'::jsonb,  -- start/end sentence ids, etc.
  chunk_index   integer not null default 0, -- ordering within file
  uploaded_by   uuid references auth.users(id),
  created_at    timestamptz default now()
);

create index if not exists idx_event_chunks_source_file on public.event_chunks(source_file);
create index if not exists idx_event_chunks_chunk_key   on public.event_chunks(chunk_key);

-- 2. Annotations table — one row per chunk per annotator.
--    Multiple annotators can annotate the same chunk (inter-annotator agreement).
create table if not exists public.event_annotations (
  id                     uuid primary key default gen_random_uuid(),
  chunk_key              text not null references public.event_chunks(chunk_key) on delete cascade,
  user_id                uuid not null references auth.users(id),
  human_summary          text,
  alignment_label        text check (alignment_label in ('aligned','partially_aligned','misaligned')),
  annotation_reasoning   text,
  error_tags             text[] default '{}',
  annotation_completed   boolean default false,
  created_at             timestamptz default now(),
  updated_at             timestamptz default now(),
  unique (chunk_key, user_id)   -- one annotation per user per chunk
);

create index if not exists idx_event_annotations_chunk on public.event_annotations(chunk_key);
create index if not exists idx_event_annotations_user  on public.event_annotations(user_id);

-- Auto-update updated_at
create or replace function public.set_event_annotation_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger event_annotations_set_updated_at
before update on public.event_annotations
for each row execute procedure public.set_event_annotation_updated_at();

-- 3. Activity log — lightweight feed so teammates see who did what.
create table if not exists public.event_activity (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users(id),
  chunk_key   text not null,
  action      text not null default 'annotated',  -- "annotated", "updated"
  created_at  timestamptz default now()
);

create index if not exists idx_event_activity_time on public.event_activity(created_at desc);

-- =========================================================================
-- Row-Level Security
-- =========================================================================
alter table public.event_chunks      enable row level security;
alter table public.event_annotations enable row level security;
alter table public.event_activity    enable row level security;

-- Chunks: anyone authenticated can read; only authenticated can insert (upload)
create policy "event_chunks_select" on public.event_chunks
  for select using (auth.role() = 'authenticated');

create policy "event_chunks_insert" on public.event_chunks
  for insert with check (auth.role() = 'authenticated');

-- Annotations: anyone authenticated can read; users own their rows
create policy "event_annotations_select" on public.event_annotations
  for select using (auth.role() = 'authenticated');

create policy "event_annotations_insert" on public.event_annotations
  for insert with check (auth.uid() = user_id);

create policy "event_annotations_update" on public.event_annotations
  for update using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "event_annotations_delete" on public.event_annotations
  for delete using (auth.uid() = user_id);

-- Activity: anyone authenticated can read/insert
create policy "event_activity_select" on public.event_activity
  for select using (auth.role() = 'authenticated');

create policy "event_activity_insert" on public.event_activity
  for insert with check (auth.role() = 'authenticated');
