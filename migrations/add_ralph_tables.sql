-- Ralph System Tables
-- Track iteration history and checkpoints for RAGNAROK v8.0
-- Run with: psql $DATABASE_URL -f migrations/add_ralph_tables.sql

-- ============================================================
-- RALPH CHECKPOINTS TABLE
-- Stores full state for each Ralph loop instance
-- ============================================================
CREATE TABLE IF NOT EXISTS ralph_checkpoints (
    id SERIAL PRIMARY KEY,
    loop_id VARCHAR(50) UNIQUE NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    task TEXT,
    state JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'running',
    best_score FLOAT,
    total_cost FLOAT DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    iteration_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for quick loop_id lookups
CREATE INDEX IF NOT EXISTS idx_ralph_checkpoints_loop_id ON ralph_checkpoints(loop_id);
CREATE INDEX IF NOT EXISTS idx_ralph_checkpoints_agent ON ralph_checkpoints(agent_name);
CREATE INDEX IF NOT EXISTS idx_ralph_checkpoints_status ON ralph_checkpoints(status);
CREATE INDEX IF NOT EXISTS idx_ralph_checkpoints_created ON ralph_checkpoints(created_at DESC);

-- ============================================================
-- RALPH ITERATIONS TABLE
-- Individual iteration records within a Ralph loop
-- ============================================================
CREATE TABLE IF NOT EXISTS ralph_iterations (
    id SERIAL PRIMARY KEY,
    loop_id VARCHAR(50) NOT NULL,
    iteration_number INTEGER NOT NULL,
    score FLOAT,
    tokens_used INTEGER DEFAULT 0,
    cost FLOAT DEFAULT 0,
    duration_ms INTEGER,
    completion_signal BOOLEAN DEFAULT FALSE,
    output_summary TEXT,  -- Truncated output for logging
    errors JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (loop_id) REFERENCES ralph_checkpoints(loop_id) ON DELETE CASCADE
);

-- Index for iteration queries
CREATE INDEX IF NOT EXISTS idx_ralph_iterations_loop_id ON ralph_iterations(loop_id);
CREATE INDEX IF NOT EXISTS idx_ralph_iterations_score ON ralph_iterations(score DESC);

-- ============================================================
-- PIPELINE ITERATIONS TABLE
-- Track full pipeline reruns triggered by Quality Gate
-- ============================================================
CREATE TABLE IF NOT EXISTS pipeline_iterations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    iteration_number INTEGER NOT NULL,
    auteur_score FLOAT,
    technical_qa_passed BOOLEAN DEFAULT TRUE,
    gate_decision VARCHAR(50),
    reason TEXT,
    feedback TEXT,
    phases_rerun JSONB,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    duration_ms INTEGER,
    cost_delta FLOAT DEFAULT 0
);

-- Index for session queries
CREATE INDEX IF NOT EXISTS idx_pipeline_iterations_session ON pipeline_iterations(session_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_iterations_decision ON pipeline_iterations(gate_decision);

-- ============================================================
-- PRODUCTION QUALITY HISTORY TABLE
-- Long-term tracking of production quality metrics
-- ============================================================
CREATE TABLE IF NOT EXISTS production_quality_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    business_name VARCHAR(255),
    industry VARCHAR(100),
    final_auteur_score FLOAT,
    final_technical_score FLOAT,
    total_pipeline_iterations INTEGER DEFAULT 1,
    total_agent_iterations INTEGER DEFAULT 0,
    total_cost FLOAT,
    total_duration_ms INTEGER,
    ralph_enabled BOOLEAN DEFAULT TRUE,
    quality_gate_decisions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for quality analytics
CREATE INDEX IF NOT EXISTS idx_quality_history_session ON production_quality_history(session_id);
CREATE INDEX IF NOT EXISTS idx_quality_history_score ON production_quality_history(final_auteur_score DESC);
CREATE INDEX IF NOT EXISTS idx_quality_history_created ON production_quality_history(created_at DESC);

-- ============================================================
-- HELPER VIEWS
-- ============================================================

-- View: Ralph loop summary
CREATE OR REPLACE VIEW ralph_loop_summary AS
SELECT
    rc.loop_id,
    rc.agent_name,
    rc.status,
    rc.best_score,
    rc.total_cost,
    rc.iteration_count,
    COUNT(ri.id) as actual_iterations,
    MAX(ri.score) as max_score_achieved,
    MIN(ri.score) as min_score_achieved,
    AVG(ri.score) as avg_score,
    SUM(ri.duration_ms) as total_duration_ms,
    rc.created_at
FROM ralph_checkpoints rc
LEFT JOIN ralph_iterations ri ON rc.loop_id = ri.loop_id
GROUP BY rc.loop_id, rc.agent_name, rc.status, rc.best_score,
         rc.total_cost, rc.iteration_count, rc.created_at
ORDER BY rc.created_at DESC;

-- View: Pipeline iteration summary
CREATE OR REPLACE VIEW pipeline_iteration_summary AS
SELECT
    session_id,
    COUNT(*) as total_iterations,
    MAX(auteur_score) as best_auteur_score,
    MIN(auteur_score) as worst_auteur_score,
    SUM(cost_delta) as total_cost,
    array_agg(gate_decision ORDER BY iteration_number) as decision_history
FROM pipeline_iterations
GROUP BY session_id
ORDER BY MAX(started_at) DESC;

-- View: Quality improvement tracking
CREATE OR REPLACE VIEW quality_improvement AS
SELECT
    DATE(created_at) as date,
    COUNT(*) as productions,
    AVG(final_auteur_score) as avg_auteur_score,
    AVG(total_pipeline_iterations) as avg_iterations,
    AVG(total_cost) as avg_cost,
    SUM(CASE WHEN final_auteur_score >= 85 THEN 1 ELSE 0 END)::float / COUNT(*) * 100 as pass_rate
FROM production_quality_history
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- ============================================================
-- COMMENTS
-- ============================================================
COMMENT ON TABLE ralph_checkpoints IS 'Stores Ralph loop state checkpoints for recovery and audit';
COMMENT ON TABLE ralph_iterations IS 'Individual iteration records within Ralph loops';
COMMENT ON TABLE pipeline_iterations IS 'Full pipeline iteration records triggered by Quality Gate';
COMMENT ON TABLE production_quality_history IS 'Long-term quality metrics for analytics';

-- ============================================================
-- GRANTS (adjust role as needed)
-- ============================================================
-- GRANT SELECT, INSERT, UPDATE ON ralph_checkpoints TO genesis_api;
-- GRANT SELECT, INSERT ON ralph_iterations TO genesis_api;
-- GRANT SELECT, INSERT ON pipeline_iterations TO genesis_api;
-- GRANT SELECT, INSERT ON production_quality_history TO genesis_api;
