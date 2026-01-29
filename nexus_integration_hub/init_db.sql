-- NEXUS Integration Hub - Database Schema
-- =========================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- SOCIAL POSTS (From CHROMADON)
-- =============================================================================

CREATE TABLE IF NOT EXISTS social_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    platform VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    hook_used VARCHAR(255) NOT NULL,
    hook_category VARCHAR(100) NOT NULL,
    visual_prompt TEXT,
    scriptwriter_session_id VARCHAR(100),
    
    -- Engagement metrics
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    shares INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,
    
    -- Attribution
    leads_generated TEXT[] DEFAULT '{}',
    conversion_count INTEGER DEFAULT 0,
    quality_rating VARCHAR(50),
    
    -- Timestamps
    posted_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_posts_platform ON social_posts(platform);
CREATE INDEX idx_posts_posted_at ON social_posts(posted_at DESC);
CREATE INDEX idx_posts_hook_category ON social_posts(hook_category);
CREATE INDEX idx_posts_quality ON social_posts(quality_rating) WHERE quality_rating IS NOT NULL;

-- =============================================================================
-- LEADS
-- =============================================================================

CREATE TABLE IF NOT EXISTS leads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source VARCHAR(50) NOT NULL,
    platform VARCHAR(50) NOT NULL,
    source_post_id UUID REFERENCES social_posts(id),
    
    -- Contact info
    contact_handle VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    
    -- Journey
    status VARCHAR(50) NOT NULL DEFAULT 'new',
    conversation_thread_id VARCHAR(100) NOT NULL,
    
    -- Qualification
    interest_level INTEGER DEFAULT 0,
    budget_indicator VARCHAR(100),
    pain_points TEXT[] DEFAULT '{}',
    
    -- Conversion
    deal_value DECIMAL(12, 2),
    
    -- Timestamps
    first_contact_at TIMESTAMPTZ NOT NULL,
    last_interaction_at TIMESTAMPTZ NOT NULL,
    converted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_leads_status ON leads(status);
CREATE INDEX idx_leads_platform ON leads(platform);
CREATE INDEX idx_leads_source_post ON leads(source_post_id);
CREATE INDEX idx_leads_first_contact ON leads(first_contact_at DESC);
CREATE INDEX idx_leads_conversion ON leads(status, converted_at) WHERE status = 'converted';

-- =============================================================================
-- CONTENT FEEDBACK (For SCRIPTWRITER-X learning)
-- =============================================================================

CREATE TABLE IF NOT EXISTS content_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES social_posts(id),
    hook_used VARCHAR(255) NOT NULL,
    hook_category VARCHAR(100) NOT NULL,
    quality_rating VARCHAR(50) NOT NULL,
    leads_generated INTEGER DEFAULT 0,
    conversions INTEGER DEFAULT 0,
    engagement_score DECIMAL(5, 2) DEFAULT 0,
    sent_to_scriptwriter BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_hook ON content_feedback(hook_used);
CREATE INDEX idx_feedback_quality ON content_feedback(quality_rating);
CREATE INDEX idx_feedback_pending ON content_feedback(sent_to_scriptwriter) WHERE sent_to_scriptwriter = FALSE;

-- =============================================================================
-- EVENTS LOG (Audit trail)
-- =============================================================================

CREATE TABLE IF NOT EXISTS events_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    source_system VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_events_type ON events_log(event_type);
CREATE INDEX idx_events_created ON events_log(created_at DESC);
CREATE INDEX idx_events_pending ON events_log(processed) WHERE processed = FALSE;

-- Partition by month for scalability (optional)
-- CREATE TABLE events_log_2024_01 PARTITION OF events_log
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- =============================================================================
-- ATTRIBUTION ANALYTICS (Materialized View)
-- =============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_attribution_stats AS
SELECT 
    sp.platform,
    sp.hook_category,
    sp.hook_used,
    COUNT(DISTINCT sp.id) as total_posts,
    COUNT(DISTINCT l.id) as total_leads,
    COUNT(DISTINCT CASE WHEN l.status = 'converted' THEN l.id END) as conversions,
    COALESCE(SUM(l.deal_value), 0) as total_revenue,
    AVG(sp.likes + sp.comments * 2 + sp.shares * 3) as avg_engagement,
    CASE 
        WHEN COUNT(DISTINCT l.id) > 0 
        THEN ROUND(COUNT(DISTINCT CASE WHEN l.status = 'converted' THEN l.id END)::NUMERIC / COUNT(DISTINCT l.id) * 100, 2)
        ELSE 0 
    END as conversion_rate
FROM social_posts sp
LEFT JOIN leads l ON l.source_post_id = sp.id
GROUP BY sp.platform, sp.hook_category, sp.hook_used;

CREATE UNIQUE INDEX idx_mv_attribution ON mv_attribution_stats(platform, hook_category, hook_used);

-- Refresh function (run periodically)
CREATE OR REPLACE FUNCTION refresh_attribution_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_attribution_stats;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_posts_updated
    BEFORE UPDATE ON social_posts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_leads_updated
    BEFORE UPDATE ON leads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Log lead status changes
CREATE OR REPLACE FUNCTION log_lead_status_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO events_log (event_type, source_system, payload)
        VALUES (
            'lead.status_changed',
            'database_trigger',
            jsonb_build_object(
                'lead_id', NEW.id,
                'old_status', OLD.status,
                'new_status', NEW.status,
                'source_post_id', NEW.source_post_id
            )
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_lead_status_changed
    AFTER UPDATE ON leads
    FOR EACH ROW EXECUTE FUNCTION log_lead_status_change();

-- =============================================================================
-- SEED DATA (Hook categories)
-- =============================================================================

CREATE TABLE IF NOT EXISTS hook_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    priority INTEGER DEFAULT 5
);

INSERT INTO hook_categories (name, description, priority) VALUES
    ('pain_point', 'Highlights a specific problem the audience faces', 10),
    ('curiosity', 'Creates intrigue and desire to learn more', 9),
    ('value_prop', 'Communicates unique value proposition', 8),
    ('social_proof', 'Leverages testimonials or results', 8),
    ('urgency', 'Creates time-sensitive motivation', 7),
    ('authority', 'Establishes expertise and credibility', 7),
    ('contrarian', 'Challenges conventional wisdom', 6),
    ('story', 'Opens with a compelling narrative', 6),
    ('question', 'Engages with a thought-provoking question', 5),
    ('statistic', 'Leads with a surprising data point', 5)
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- GRANTS
-- =============================================================================

-- Create application role
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nexus_app') THEN
        CREATE ROLE nexus_app WITH LOGIN PASSWORD 'nexus_app_2024';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO nexus_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO nexus_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO nexus_app;

-- Read-only role for analytics
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'nexus_readonly') THEN
        CREATE ROLE nexus_readonly WITH LOGIN PASSWORD 'nexus_readonly_2024';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO nexus_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO nexus_readonly;
