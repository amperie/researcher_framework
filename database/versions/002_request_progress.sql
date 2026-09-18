ALTER TABLE researcher.platform_requests
    ADD COLUMN progress jsonb NOT NULL DEFAULT '[]'::jsonb
    CHECK (jsonb_typeof(progress) = 'array');
