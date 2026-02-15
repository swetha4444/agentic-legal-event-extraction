-- Admin-only dataset deletion policies for existing Supabase deployments.
-- Run this in Supabase SQL Editor after running schema.sql.

create table if not exists public.admin_users (
  user_id uuid primary key references auth.users(id) on delete cascade,
  created_at timestamptz default now()
);

alter table public.admin_users enable row level security;

drop policy if exists "Admin users read own row" on public.admin_users;
create policy "Admin users read own row"
  on public.admin_users for select
  using (auth.uid() = user_id);

drop policy if exists "Datasets delete admin only" on public.datasets;
create policy "Datasets delete admin only"
  on public.datasets for delete
  using (
    exists (
      select 1
      from public.admin_users admins
      where admins.user_id = auth.uid()
    )
  );

-- Example: make a user admin by email
-- insert into public.admin_users (user_id)
-- select id from auth.users where email = 'you@example.com'
-- on conflict (user_id) do nothing;
