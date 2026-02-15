-- Migration for existing Supabase deployments to preserve original input rows.
-- Run this in Supabase SQL Editor.

create table if not exists public.source_records (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid references public.datasets(id) on delete cascade,
  source_row integer not null,
  record jsonb not null,
  created_at timestamptz default now(),
  unique (dataset_id, source_row)
);

alter table public.source_records enable row level security;

drop policy if exists "Source records readable" on public.source_records;
create policy "Source records readable"
  on public.source_records for select
  using (auth.role() = 'authenticated');

drop policy if exists "Source records insert" on public.source_records;
create policy "Source records insert"
  on public.source_records for insert
  with check (auth.role() = 'authenticated');

drop policy if exists "Source records update" on public.source_records;
create policy "Source records update"
  on public.source_records for update
  using (auth.role() = 'authenticated')
  with check (auth.role() = 'authenticated');
