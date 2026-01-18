"""
================================================================================
NEXUS ORCHESTRATOR - Database Migration
================================================================================
PostgreSQL migration script for customer lifecycle tables.
Run with: python -m alembic upgrade head (or directly)
================================================================================
"""

import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("nexus.migration")

# Raw SQL for direct execution
MIGRATION_SQL = """
-- =============================================================================
-- NEXUS ORCHESTRATOR DATABASE SCHEMA
-- Version: 1.0.0
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =============================================================================
-- ENUM TYPES
-- =============================================================================

DO $$ BEGIN
    CREATE TYPE nexus_phase AS ENUM (
        'acquisition',
        'onboarding', 
        'provisioning',
        'active',
        'expansion',
        'engagement',
        'retention',
        'billing',
        'renewal',
        'dunning',
        'suspension',
        'cancellation',
        'churn'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE customer_status AS ENUM (
        'lead',
        'trial',
        'active',
        'past_due',
        'suspended',
        'cancelled',
        'churned'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE subscription_tier AS ENUM (
        'starter',
        'growth',
        'scale',
        'enterprise',
        'nexus_personal'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE payment_status AS ENUM (
        'pending',
        'succeeded',
        'failed',
        'refunded',
        'disputed'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE notification_type AS ENUM (
        'welcome',
        'onboarding',
        'payment_succeeded',
        'payment_failed',
        'renewal_reminder',
        'dunning_1',
        'dunning_2',
        'dunning_final',
        'suspension_warning',
        'account_suspended',
        'cancellation_confirmed',
        'win_back'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- =============================================================================
-- CUSTOMERS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_customers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- External IDs
    stripe_customer_id VARCHAR(255) UNIQUE NOT NULL,
    clerk_user_id VARCHAR(255) UNIQUE,
    
    -- Profile
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    company VARCHAR(255),
    phone VARCHAR(50),
    
    -- Lifecycle state
    current_phase nexus_phase DEFAULT 'acquisition',
    status customer_status DEFAULT 'lead',
    tier subscription_tier,
    
    -- Subscription tracking
    subscription_id VARCHAR(255),
    subscription_started_at TIMESTAMPTZ,
    subscription_ends_at TIMESTAMPTZ,
    trial_ends_at TIMESTAMPTZ,
    
    -- Billing
    mrr DECIMAL(12, 2) DEFAULT 0.00,
    lifetime_value DECIMAL(12, 2) DEFAULT 0.00,
    last_payment_at TIMESTAMPTZ,
    next_billing_at TIMESTAMPTZ,
    
    -- Dunning state
    dunning_started_at TIMESTAMPTZ,
    dunning_attempts INTEGER DEFAULT 0,
    last_dunning_at TIMESTAMPTZ,
    
    -- Engagement metrics
    last_active_at TIMESTAMPTZ,
    login_count INTEGER DEFAULT 0,
    feature_usage JSONB DEFAULT '{}',
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    
    -- Versioning for MVCC
    version INTEGER DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Customer indexes
CREATE INDEX IF NOT EXISTS idx_customer_stripe_id ON nexus_customers(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_customer_email ON nexus_customers(email);
CREATE INDEX IF NOT EXISTS idx_customer_phase_status ON nexus_customers(current_phase, status);
CREATE INDEX IF NOT EXISTS idx_customer_tier ON nexus_customers(tier);
CREATE INDEX IF NOT EXISTS idx_customer_subscription ON nexus_customers(subscription_id);
CREATE INDEX IF NOT EXISTS idx_customer_next_billing ON nexus_customers(next_billing_at);
CREATE INDEX IF NOT EXISTS idx_customer_dunning ON nexus_customers(dunning_started_at, dunning_attempts);
CREATE INDEX IF NOT EXISTS idx_customer_email_trgm ON nexus_customers USING gin(email gin_trgm_ops);

-- =============================================================================
-- PHASE TRANSITIONS TABLE (Audit Log)
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_phase_transitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES nexus_customers(id) ON DELETE CASCADE,
    
    -- Transition details
    from_phase nexus_phase,
    to_phase nexus_phase NOT NULL,
    from_status customer_status,
    to_status customer_status NOT NULL,
    
    -- Trigger info
    trigger_event VARCHAR(100) NOT NULL,
    trigger_event_id VARCHAR(255),
    trigger_source VARCHAR(50) DEFAULT 'stripe',
    
    -- Handler result
    handler_action VARCHAR(100) NOT NULL,
    handler_success BOOLEAN DEFAULT true,
    handler_error TEXT,
    handler_duration_ms INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Customer version at transition time
    customer_version INTEGER NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Phase transition indexes
CREATE INDEX IF NOT EXISTS idx_transition_customer ON nexus_phase_transitions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transition_customer_time ON nexus_phase_transitions(customer_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_transition_trigger ON nexus_phase_transitions(trigger_event);
CREATE INDEX IF NOT EXISTS idx_transition_created ON nexus_phase_transitions(created_at DESC);

-- =============================================================================
-- PAYMENTS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_payments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES nexus_customers(id) ON DELETE CASCADE,
    
    -- Stripe IDs
    stripe_payment_intent_id VARCHAR(255) UNIQUE,
    stripe_charge_id VARCHAR(255) UNIQUE,
    stripe_invoice_id VARCHAR(255),
    stripe_subscription_id VARCHAR(255),
    
    -- Amount
    amount INTEGER NOT NULL,  -- In cents
    currency VARCHAR(3) DEFAULT 'usd',
    
    -- Status
    status payment_status DEFAULT 'pending',
    failure_code VARCHAR(100),
    failure_message TEXT,
    
    -- Refund tracking
    refunded_amount INTEGER DEFAULT 0,
    refund_reason VARCHAR(255),
    
    -- Metadata
    description VARCHAR(500),
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Payment indexes
CREATE INDEX IF NOT EXISTS idx_payment_customer ON nexus_payments(customer_id);
CREATE INDEX IF NOT EXISTS idx_payment_intent ON nexus_payments(stripe_payment_intent_id);
CREATE INDEX IF NOT EXISTS idx_payment_charge ON nexus_payments(stripe_charge_id);
CREATE INDEX IF NOT EXISTS idx_payment_invoice ON nexus_payments(stripe_invoice_id);
CREATE INDEX IF NOT EXISTS idx_payment_status ON nexus_payments(status);
CREATE INDEX IF NOT EXISTS idx_payment_created ON nexus_payments(created_at DESC);

-- =============================================================================
-- ENTITLEMENTS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_entitlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES nexus_customers(id) ON DELETE CASCADE,
    
    -- Entitlement details
    feature_key VARCHAR(100) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    
    -- Limits
    quota INTEGER,  -- null = unlimited
    used INTEGER DEFAULT 0,
    
    -- Status
    is_active BOOLEAN DEFAULT true,
    granted_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    
    -- Source
    source_subscription_id VARCHAR(255),
    source_product_id VARCHAR(255),
    
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT uq_customer_feature UNIQUE (customer_id, feature_key)
);

-- Entitlement indexes
CREATE INDEX IF NOT EXISTS idx_entitlement_customer ON nexus_entitlements(customer_id);
CREATE INDEX IF NOT EXISTS idx_entitlement_active ON nexus_entitlements(customer_id, is_active);
CREATE INDEX IF NOT EXISTS idx_entitlement_feature ON nexus_entitlements(feature_key);

-- =============================================================================
-- NOTIFICATIONS TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES nexus_customers(id) ON DELETE CASCADE,
    
    -- Notification details
    type notification_type NOT NULL,
    channel VARCHAR(50) DEFAULT 'email',
    
    -- Content
    subject VARCHAR(500),
    template_id VARCHAR(100),
    
    -- Delivery status
    sent_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    clicked_at TIMESTAMPTZ,
    bounced_at TIMESTAMPTZ,
    
    -- External tracking
    external_id VARCHAR(255),
    
    -- Trigger context
    trigger_phase nexus_phase,
    trigger_event VARCHAR(100),
    
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Notification indexes
CREATE INDEX IF NOT EXISTS idx_notification_customer ON nexus_notifications(customer_id);
CREATE INDEX IF NOT EXISTS idx_notification_type_sent ON nexus_notifications(customer_id, type, sent_at);
CREATE INDEX IF NOT EXISTS idx_notification_created ON nexus_notifications(created_at DESC);

-- =============================================================================
-- DUNNING SCHEDULE TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_dunning_schedules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID NOT NULL REFERENCES nexus_customers(id) ON DELETE CASCADE,
    
    -- Invoice being dunned
    stripe_invoice_id VARCHAR(255) NOT NULL,
    amount_due INTEGER NOT NULL,
    
    -- Schedule
    attempt_number INTEGER DEFAULT 1,
    max_attempts INTEGER DEFAULT 4,
    
    -- Timing
    scheduled_at TIMESTAMPTZ NOT NULL,
    executed_at TIMESTAMPTZ,
    
    -- Result
    retry_succeeded BOOLEAN,
    failure_reason TEXT,
    
    -- Actions taken
    notification_sent BOOLEAN DEFAULT false,
    card_update_requested BOOLEAN DEFAULT false,
    
    is_active BOOLEAN DEFAULT true,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dunning indexes
CREATE INDEX IF NOT EXISTS idx_dunning_customer ON nexus_dunning_schedules(customer_id);
CREATE INDEX IF NOT EXISTS idx_dunning_active_scheduled ON nexus_dunning_schedules(is_active, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_dunning_invoice ON nexus_dunning_schedules(stripe_invoice_id);

-- =============================================================================
-- WEBHOOK EVENTS TABLE (Idempotency)
-- =============================================================================

CREATE TABLE IF NOT EXISTS nexus_webhook_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Event details
    stripe_event_id VARCHAR(255) UNIQUE NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    
    -- Processing status
    processed_at TIMESTAMPTZ,
    processing_time_ms INTEGER,
    success BOOLEAN,
    error TEXT,
    
    -- Raw event data
    payload JSONB NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Webhook event indexes
CREATE INDEX IF NOT EXISTS idx_webhook_stripe_id ON nexus_webhook_events(stripe_event_id);
CREATE INDEX IF NOT EXISTS idx_webhook_type ON nexus_webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_created ON nexus_webhook_events(created_at DESC);

-- =============================================================================
-- FUNCTIONS & TRIGGERS
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables
DROP TRIGGER IF EXISTS update_nexus_customers_updated_at ON nexus_customers;
CREATE TRIGGER update_nexus_customers_updated_at
    BEFORE UPDATE ON nexus_customers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_nexus_payments_updated_at ON nexus_payments;
CREATE TRIGGER update_nexus_payments_updated_at
    BEFORE UPDATE ON nexus_payments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_nexus_entitlements_updated_at ON nexus_entitlements;
CREATE TRIGGER update_nexus_entitlements_updated_at
    BEFORE UPDATE ON nexus_entitlements
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_nexus_dunning_schedules_updated_at ON nexus_dunning_schedules;
CREATE TRIGGER update_nexus_dunning_schedules_updated_at
    BEFORE UPDATE ON nexus_dunning_schedules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- VIEWS
-- =============================================================================

-- Customer lifecycle summary view
CREATE OR REPLACE VIEW v_customer_lifecycle_summary AS
SELECT 
    c.id,
    c.email,
    c.name,
    c.current_phase,
    c.status,
    c.tier,
    c.mrr,
    c.lifetime_value,
    c.dunning_attempts,
    c.created_at,
    c.updated_at,
    COUNT(DISTINCT pt.id) as total_transitions,
    COUNT(DISTINCT p.id) as total_payments,
    COUNT(DISTINCT e.id) FILTER (WHERE e.is_active) as active_entitlements,
    MAX(pt.created_at) as last_transition_at,
    MAX(p.created_at) as last_payment_at
FROM nexus_customers c
LEFT JOIN nexus_phase_transitions pt ON c.id = pt.customer_id
LEFT JOIN nexus_payments p ON c.id = p.customer_id
LEFT JOIN nexus_entitlements e ON c.id = e.customer_id
GROUP BY c.id;

-- Dunning queue view
CREATE OR REPLACE VIEW v_dunning_queue AS
SELECT 
    ds.*,
    c.email,
    c.name,
    c.dunning_attempts as total_attempts,
    c.mrr
FROM nexus_dunning_schedules ds
JOIN nexus_customers c ON ds.customer_id = c.id
WHERE ds.is_active = true
  AND ds.executed_at IS NULL
ORDER BY ds.scheduled_at ASC;

-- Revenue by tier view
CREATE OR REPLACE VIEW v_revenue_by_tier AS
SELECT 
    tier,
    status,
    COUNT(*) as customer_count,
    SUM(mrr) as total_mrr,
    AVG(mrr) as avg_mrr,
    SUM(lifetime_value) as total_ltv,
    AVG(lifetime_value) as avg_ltv
FROM nexus_customers
WHERE status IN ('active', 'trial')
GROUP BY tier, status
ORDER BY total_mrr DESC;

-- =============================================================================
-- SAMPLE DATA (Optional - for testing)
-- =============================================================================

-- Uncomment below to insert sample data for testing
/*
INSERT INTO nexus_customers (stripe_customer_id, email, name, current_phase, status, tier, mrr)
VALUES 
    ('cus_test_001', 'test1@example.com', 'Test User 1', 'active', 'active', 'starter', 49.00),
    ('cus_test_002', 'test2@example.com', 'Test User 2', 'dunning', 'past_due', 'growth', 149.00),
    ('cus_test_003', 'test3@example.com', 'Test User 3', 'onboarding', 'trial', 'scale', 0.00);
*/

-- =============================================================================
-- MIGRATION COMPLETE
-- =============================================================================

SELECT 'Nexus Orchestrator database migration complete!' as status;
"""


async def run_migration(connection_string: str):
    """
    Run the database migration.
    
    Args:
        connection_string: PostgreSQL connection string
    """
    import asyncpg
    
    logger.info("Starting Nexus database migration...")
    
    try:
        conn = await asyncpg.connect(connection_string)
        
        # Run migration
        await conn.execute(MIGRATION_SQL)
        
        logger.info("Migration completed successfully!")
        
        # Verify tables
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'nexus_%'
        """)
        
        logger.info(f"Created tables: {[t['table_name'] for t in tables]}")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def get_migration_sql() -> str:
    """Return the raw migration SQL for manual execution."""
    return MIGRATION_SQL


if __name__ == "__main__":
    import sys
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    # Get connection string from env or command line
    connection_string = os.getenv(
        "DATABASE_URL",
        sys.argv[1] if len(sys.argv) > 1 else None
    )
    
    if not connection_string:
        print("Usage: python migration.py <connection_string>")
        print("Or set DATABASE_URL environment variable")
        sys.exit(1)
    
    asyncio.run(run_migration(connection_string))
