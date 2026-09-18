REVOKE ALL ON SCHEMA researcher FROM PUBLIC;
GRANT USAGE ON SCHEMA researcher TO qc_researcher;

CREATE TABLE researcher.platform_requests (
    tenant_id text NOT NULL CHECK (tenant_id <> ''),
    request_id text NOT NULL, session_id text NOT NULL, input_hash text NOT NULL,
    status text NOT NULL CHECK (status IN ('running','succeeded','failed','stopped','interrupted')),
    stage text NOT NULL, started_at timestamptz NOT NULL, expires_at timestamptz NOT NULL,
    finished_at timestamptz, result jsonb, error jsonb,
    PRIMARY KEY (tenant_id, request_id)
);
CREATE INDEX active_requests ON researcher.platform_requests (tenant_id, expires_at) WHERE status='running';

CREATE TABLE researcher.llm_usage (
    tenant_id text NOT NULL CHECK (tenant_id <> ''), call_id text NOT NULL,
    request_id text NOT NULL, started_at timestamptz NOT NULL, event jsonb NOT NULL,
    PRIMARY KEY (tenant_id, call_id),
    CHECK (event->>'tenantId' IS NOT DISTINCT FROM tenant_id),
    CHECK (event->>'requestId' IS NOT DISTINCT FROM request_id),
    CHECK (event->>'callId' IS NOT DISTINCT FROM call_id)
);
CREATE INDEX usage_request ON researcher.llm_usage (tenant_id, request_id);
CREATE INDEX usage_time ON researcher.llm_usage (tenant_id, started_at, call_id);

ALTER TABLE researcher.platform_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE researcher.platform_requests FORCE ROW LEVEL SECURITY;
ALTER TABLE researcher.llm_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE researcher.llm_usage FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_requests ON researcher.platform_requests
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), ''))
    WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), ''));
CREATE POLICY tenant_usage ON researcher.llm_usage
    USING (tenant_id = nullif(current_setting('app.tenant_id', true), ''))
    WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), ''));
GRANT SELECT, INSERT, UPDATE ON researcher.platform_requests, researcher.llm_usage TO qc_researcher;

-- Only the migration owner can read all receipts. Runtime callers receive a count,
-- never another tenant's identifiers or contents. Claims serialize before counting.
CREATE POLICY owner_capacity ON researcher.platform_requests FOR SELECT TO qc_researcher_owner USING (true);
CREATE FUNCTION researcher.active_request_count() RETURNS bigint
    LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog AS $$
    SELECT count(*) FROM researcher.platform_requests
    WHERE status='running' AND expires_at > clock_timestamp()
$$;
REVOKE ALL ON FUNCTION researcher.active_request_count() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION researcher.active_request_count() TO qc_researcher;
