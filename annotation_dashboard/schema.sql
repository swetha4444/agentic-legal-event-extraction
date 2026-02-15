-- Supabase schema for shared annotation dashboard

create extension if not exists "pgcrypto";

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  created_at timestamptz default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do nothing;
  return new;
end;
$$;

create trigger on_auth_user_created
after insert on auth.users
for each row execute procedure public.handle_new_user();

create table if not exists public.datasets (
  id uuid primary key default gen_random_uuid(),
  hash text unique not null,
  file_name text,
  text_column text,
  sentence_mode boolean default true,
  created_by uuid references auth.users(id),
  created_at timestamptz default now()
);

create table if not exists public.sentences (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid references public.datasets(id) on delete cascade,
  item_index integer not null,
  source_row integer,
  sentence_index integer,
  sentence_count integer,
  text text not null,
  created_at timestamptz default now(),
  unique (dataset_id, item_index)
);

create table if not exists public.annotations (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid references public.datasets(id) on delete cascade,
  item_index integer not null,
  label text not null,
  user_id uuid references auth.users(id),
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique (dataset_id, item_index, user_id)
);

create table if not exists public.source_records (
  id uuid primary key default gen_random_uuid(),
  dataset_id uuid references public.datasets(id) on delete cascade,
  source_row integer not null,
  record jsonb not null,
  created_at timestamptz default now(),
  unique (dataset_id, source_row)
);

create table if not exists public.admin_users (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz default now()
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger annotations_set_updated_at
before update on public.annotations
for each row execute procedure public.set_updated_at();

alter table public.profiles enable row level security;
alter table public.datasets enable row level security;
alter table public.sentences enable row level security;
alter table public.annotations enable row level security;
alter table public.source_records enable row level security;
alter table public.admin_users enable row level security;

create policy "Profiles are readable by authenticated users"
  on public.profiles for select
  using (auth.role() = 'authenticated');

create policy "Users can insert their profile"
  on public.profiles for insert
  with check (auth.uid() = id);

create policy "Datasets readable"
  on public.datasets for select
  using (auth.role() = 'authenticated');

create policy "Datasets insert"
  on public.datasets for insert
  with check (auth.role() = 'authenticated');

create policy "Datasets delete admin only"
  on public.datasets for delete
  using (
    exists (
      select 1
      from public.admin_users admins
      where admins.user_id = auth.uid()
    )
  );

create policy "Sentences readable"
  on public.sentences for select
  using (auth.role() = 'authenticated');

create policy "Sentences insert"
  on public.sentences for insert
  with check (auth.role() = 'authenticated');

create policy "Annotations readable"
  on public.annotations for select
  using (auth.role() = 'authenticated');

create policy "Annotations insert"
  on public.annotations for insert
  with check (auth.uid() = user_id);

create policy "Annotations update own"
  on public.annotations for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "Annotations delete own"
  on public.annotations for delete
  using (auth.uid() = user_id);

create policy "Source records readable"
  on public.source_records for select
  using (auth.role() = 'authenticated');

create policy "Source records insert"
  on public.source_records for insert
  with check (auth.role() = 'authenticated');

create policy "Source records update"
  on public.source_records for update
  using (auth.role() = 'authenticated')
  with check (auth.role() = 'authenticated');

create policy "Admin users read own row"
  on public.admin_users for select
  using (auth.uid() = user_id);
