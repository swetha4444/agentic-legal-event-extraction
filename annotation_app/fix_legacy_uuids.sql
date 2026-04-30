-- =====================================================================
-- Fix legacy UUID user_ids → readable names
-- Run this in Supabase SQL Editor (safe to run multiple times).
-- =====================================================================

UPDATE event_annotations
SET user_id = 'heo'
WHERE user_id = '218b4166-e5d5-4f64-86b6-2adf3c14ee70';

UPDATE event_activity
SET user_id = 'heo'
WHERE user_id = '218b4166-e5d5-4f64-86b6-2adf3c14ee70';
