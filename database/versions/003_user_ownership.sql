-- Users are references to QC identities, not a second login/password database.
CREATE TABLE researcher.users (
    tenant_id text NOT NULL,
    user_id text NOT NULL CHECK (length(user_id) BETWEEN 1 AND 160),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id,user_id)
);
ALTER TABLE researcher.platform_requests ADD COLUMN user_id text;
ALTER TABLE researcher.llm_usage ADD COLUMN user_id text;

-- Transactional owner-only backfill; restored before any runtime access can resume.
ALTER TABLE researcher.platform_requests NO FORCE ROW LEVEL SECURITY;
ALTER TABLE researcher.llm_usage NO FORCE ROW LEVEL SECURITY;
UPDATE researcher.platform_requests SET user_id=current_setting('app.legacy_user_id');
UPDATE researcher.llm_usage SET user_id=current_setting('app.legacy_user_id'),
    event=event || jsonb_build_object('userId',current_setting('app.legacy_user_id'));
INSERT INTO researcher.users(tenant_id,user_id)
    SELECT tenant_id,user_id FROM researcher.platform_requests
    UNION SELECT tenant_id,user_id FROM researcher.llm_usage;

-- Receipts contain proposals, validation and cached metering; their owner is the request owner.
UPDATE researcher.platform_requests SET result=result || jsonb_build_object('userId',user_id)
    || CASE WHEN result->'proposal' IS NOT NULL AND result->'proposal'<>'null'::jsonb
        THEN jsonb_build_object('proposal',result->'proposal' || jsonb_build_object('userId',user_id)) ELSE '{}'::jsonb END
    || CASE WHEN result->'validation' IS NOT NULL AND result->'validation'<>'null'::jsonb
        THEN jsonb_build_object('validation',result->'validation' || jsonb_build_object('userId',user_id)) ELSE '{}'::jsonb END
    || CASE WHEN jsonb_typeof(result->'usage'->'steps')='array' THEN
        jsonb_build_object('usage',result->'usage' || jsonb_build_object('steps',
            (SELECT coalesce(jsonb_agg(e || jsonb_build_object('userId',user_id)),'[]'::jsonb)
             FROM jsonb_array_elements(result->'usage'->'steps') e))) ELSE '{}'::jsonb END
    WHERE result IS NOT NULL;

ALTER TABLE researcher.platform_requests ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE researcher.llm_usage ALTER COLUMN user_id SET NOT NULL;
ALTER TABLE researcher.platform_requests ADD FOREIGN KEY (tenant_id,user_id) REFERENCES researcher.users(tenant_id,user_id);
ALTER TABLE researcher.llm_usage ADD FOREIGN KEY (tenant_id,user_id) REFERENCES researcher.users(tenant_id,user_id);
ALTER TABLE researcher.llm_usage ADD CHECK (event->>'userId' IS NOT DISTINCT FROM user_id);
CREATE INDEX requests_owner ON researcher.platform_requests(tenant_id,user_id,started_at);
CREATE INDEX usage_owner ON researcher.llm_usage(tenant_id,user_id,started_at);

ALTER TABLE researcher.platform_requests FORCE ROW LEVEL SECURITY;
ALTER TABLE researcher.llm_usage FORCE ROW LEVEL SECURITY;
ALTER POLICY tenant_requests ON researcher.platform_requests
    USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
    WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true));
ALTER POLICY tenant_usage ON researcher.llm_usage
    USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
    WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true));
ALTER TABLE researcher.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE researcher.users FORCE ROW LEVEL SECURITY;
CREATE POLICY user_identity ON researcher.users
    USING (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true))
    WITH CHECK (tenant_id=current_setting('app.tenant_id',true) AND user_id=current_setting('app.user_id',true));
GRANT SELECT,INSERT ON researcher.users TO qc_researcher;

-- Preserve tenant-wide admission limits even though runtime reads are user-scoped.
CREATE FUNCTION researcher.active_tenant_request_count() RETURNS bigint
    LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog AS $$
    SELECT count(*) FROM researcher.platform_requests WHERE tenant_id=current_setting('app.tenant_id',true)
      AND status='running' AND expires_at>clock_timestamp()
$$;
REVOKE ALL ON FUNCTION researcher.active_tenant_request_count() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION researcher.active_tenant_request_count() TO qc_researcher;
