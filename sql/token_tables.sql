-- ============================================================================
-- BARRIOS A2I TOKEN SYSTEM - Supabase Migration
-- Run this in Supabase Dashboard > SQL Editor
-- ============================================================================

-- User token balances
CREATE TABLE IF NOT EXISTS user_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,  -- Can reference auth.users(id) if using Supabase Auth
    email TEXT,
    tokens_balance INTEGER DEFAULT 0,
    plan_type TEXT,  -- 'starter', 'creator', 'growth', 'scale', or NULL
    stripe_customer_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Token transaction history (audit log)
CREATE TABLE IF NOT EXISTS token_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    amount INTEGER NOT NULL,  -- positive = added, negative = used
    transaction_type TEXT NOT NULL,  -- 'purchase', 'subscription', 'generation', 'refund'
    description TEXT,
    stripe_payment_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_user_tokens_user_id ON user_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_user_tokens_email ON user_tokens(email);
CREATE INDEX IF NOT EXISTS idx_user_tokens_stripe ON user_tokens(stripe_customer_id);
CREATE INDEX IF NOT EXISTS idx_token_transactions_user_id ON token_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_token_transactions_created ON token_transactions(created_at DESC);

-- ============================================================================
-- ROW LEVEL SECURITY (Optional - enable if using Supabase Auth)
-- ============================================================================

-- Uncomment these if you want RLS:
-- ALTER TABLE user_tokens ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE token_transactions ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Users can view own tokens" ON user_tokens
--     FOR SELECT USING (auth.uid() = user_id);

-- CREATE POLICY "Users can view own transactions" ON token_transactions
--     FOR SELECT USING (auth.uid() = user_id);

-- ============================================================================
-- SERVICE ROLE POLICIES (for backend API access)
-- ============================================================================

-- Allow service role full access (for backend API)
-- This is default behavior in Supabase when RLS is disabled

-- ============================================================================
-- HELPER FUNCTION: Update timestamp on modification
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_tokens_updated_at
    BEFORE UPDATE ON user_tokens
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VERIFICATION QUERY (run after migration)
-- ============================================================================

-- SELECT
--     table_name,
--     column_name,
--     data_type
-- FROM information_schema.columns
-- WHERE table_name IN ('user_tokens', 'token_transactions')
-- ORDER BY table_name, ordinal_position;
