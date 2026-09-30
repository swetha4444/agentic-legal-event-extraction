-- Create framenet_annotations table (mirrors event_annotations)
CREATE TABLE IF NOT EXISTS framenet_annotations (
    id BIGSERIAL PRIMARY KEY,
    chunk_key TEXT NOT NULL,
    user_id TEXT NOT NULL,
    human_summary TEXT DEFAULT '',
    alignment_label TEXT DEFAULT '',
    annotation_reasoning TEXT DEFAULT '',
    error_tags JSONB DEFAULT '[]',
    annotation_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_key, user_id)
);

-- Create framenet_activity table (mirrors event_activity)
CREATE TABLE IF NOT EXISTS framenet_activity (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    chunk_key TEXT NOT NULL,
    action TEXT DEFAULT 'annotated',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS (Row Level Security) — same as event tables
ALTER TABLE framenet_annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE framenet_activity ENABLE ROW LEVEL SECURITY;

-- Allow anon access (same policy as event tables)
CREATE POLICY "Allow all for anon" ON framenet_annotations FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all for anon" ON framenet_activity FOR ALL USING (true) WITH CHECK (true);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_framenet_annotations_chunk_key ON framenet_annotations(chunk_key);
CREATE INDEX IF NOT EXISTS idx_framenet_annotations_user_id ON framenet_annotations(user_id);
CREATE INDEX IF NOT EXISTS idx_framenet_activity_created_at ON framenet_activity(created_at DESC);
