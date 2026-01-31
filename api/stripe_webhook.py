"""
Stripe Webhook Token Handler for Barrios A2I Commercial Lab
Adds tokens based on Stripe checkout/subscription events.

IDEMPOTENCY: Uses Supabase table `processed_stripe_events` to prevent
double-crediting when Stripe retries webhook delivery.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from fastapi import APIRouter, Request, HTTPException, Header
import stripe

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", os.getenv("STRIPE_SECRET_KEY", ""))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

# ============================================================================
# IDEMPOTENCY HELPERS
# ============================================================================

def get_supabase_client():
    """Get Supabase client for idempotency tracking."""
    from supabase import create_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase not configured - idempotency checks disabled")
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def is_event_already_processed(event_id: str) -> bool:
    """
    Check if a Stripe event has already been processed.
    Returns True if event exists in processed_stripe_events table.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False  # Can't check, proceed with processing

        result = supabase.table("processed_stripe_events").select("event_id").eq("event_id", event_id).execute()

        if result.data and len(result.data) > 0:
            logger.info(f"Event {event_id} already processed - skipping to prevent double-credit")
            return True
        return False
    except Exception as e:
        # Log but don't fail - table might not exist yet
        logger.warning(f"Idempotency check failed (table may not exist): {e}")
        return False


def record_processed_event(
    event_id: str,
    event_type: str,
    customer_id: Optional[str] = None,
    amount_tokens: int = 0
) -> bool:
    """
    Record a processed Stripe event to prevent duplicate processing.
    Returns True if recorded successfully.
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return False

        supabase.table("processed_stripe_events").insert({
            "event_id": event_id,
            "event_type": event_type,
            "customer_id": customer_id,
            "amount_tokens": amount_tokens
        }).execute()

        logger.info(f"Recorded processed event: {event_id} ({event_type})")
        return True
    except Exception as e:
        # Don't fail the webhook if we can't record - the credit already happened
        logger.error(f"Failed to record processed event: {e}")
        return False

# Token amounts per product
# V2 "Performance Engine" Pricing - Updated 2026-01-29
PRODUCT_TOKENS = {
    # V2 Subscription tiers (monthly)
    "prototyper": 16,    # $599/mo
    "growth": 40,        # $1,199/mo
    "scale": 96,         # $2,499/mo
    # V2 Entry offer
    "rapid_pilot": 8,    # $299 one-time
    # V2 Token packs (non-subscriber pricing)
    "pack_8": 8,         # $600
    "pack_16": 16,       # $1,200
    "pack_40": 40,       # $3,000
}

# Price ID to product mapping (Stripe price IDs)
# V2 "Performance Engine" Pricing - Updated 2026-01-29
PRICE_TO_PRODUCT = {
    # V2 Subscription tiers (monthly)
    "price_1SuxDoLyFGkLiU4CxxLjgoZq": "prototyper",  # $599/mo - 16 tokens
    "price_1SuxFnLyFGkLiU4CzqWvv9DR": "growth",      # $1,199/mo - 40 tokens
    "price_1SuxGQLyFGkLiU4CiDiAEkOD": "scale",       # $2,499/mo - 96 tokens
    # V2 Entry offer
    "price_1SuxIMLyFGkLiU4CBoIEIfs8": "rapid_pilot", # $299 - 8 tokens
    # V2 Token packs (non-subscriber premium pricing)
    "price_1SuxKTLyFGkLiU4CyMuPCSPL": "pack_8",      # $600 - 8 tokens
    "price_1SuxMCLyFGkLiU4C8Q45qVPJ": "pack_16",     # $1,200 - 16 tokens
    "price_1SuxO2LyFGkLiU4CRR7D14wh": "pack_40",     # $3,000 - 40 tokens
}

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/api/webhooks", tags=["webhooks"])

# ============================================================================
# WEBHOOK ENDPOINT
# ============================================================================

@router.post("/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")
):
    """
    Handle Stripe webhook events for token management.

    Events handled:
    - checkout.session.completed: New purchase/subscription
    - invoice.paid: Subscription renewal
    - customer.subscription.deleted: Subscription cancelled

    IDEMPOTENCY: Checks processed_stripe_events table before crediting.
    """
    # Get raw body
    payload = await request.body()

    # Verify webhook signature - REQUIRED in production
    if STRIPE_WEBHOOK_SECRET:
        if not stripe_signature:
            logger.error("Missing Stripe-Signature header")
            raise HTTPException(status_code=400, detail="Missing signature")
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, STRIPE_WEBHOOK_SECRET
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {e}")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature: {e}")
            raise HTTPException(status_code=400, detail="Invalid signature")
    else:
        # Only allow unverified webhooks in dev mode (no secret configured)
        import json
        event = json.loads(payload)
        logger.warning("STRIPE_WEBHOOK_SECRET not set - processing without signature verification (dev mode only)")

    event_id = event.get("id", "")
    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    logger.info(f"Received Stripe webhook: {event_type} (event_id: {event_id})")

    # ========================================================================
    # IDEMPOTENCY CHECK - Prevent double-crediting on webhook retries
    # ========================================================================
    if is_event_already_processed(event_id):
        logger.info(f"Event {event_id} already processed - returning success without re-processing")
        return {"status": "already_processed", "event_id": event_id, "event_type": event_type}

    try:
        tokens_credited = 0
        customer_id = None

        if event_type == "checkout.session.completed":
            tokens_credited, customer_id = await handle_checkout_completed(event_id, data)
        elif event_type == "invoice.paid":
            tokens_credited, customer_id = await handle_invoice_paid(event_id, data)
        elif event_type == "customer.subscription.deleted":
            tokens_credited, customer_id = await handle_subscription_deleted(event_id, data)
        else:
            logger.debug(f"Unhandled event type: {event_type}")

        # Record successful processing for idempotency
        if tokens_credited > 0 or event_type == "customer.subscription.deleted":
            record_processed_event(event_id, event_type, customer_id, tokens_credited)

        return {"status": "ok", "event_type": event_type, "event_id": event_id, "tokens_credited": tokens_credited}
    except Exception as e:
        logger.error(f"Error processing webhook {event_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EVENT HANDLERS
# ============================================================================

async def handle_checkout_completed(event_id: str, data: Dict[str, Any]) -> Tuple[int, Optional[str]]:
    """
    Handle successful checkout - add tokens to user.
    Returns: (tokens_credited, customer_id)
    """
    from api.tokens import add_tokens, AddTokensRequest

    customer_id = data.get("customer")
    customer_email = data.get("customer_email") or data.get("customer_details", {}).get("email")
    payment_id = data.get("payment_intent") or data.get("id")
    mode = data.get("mode")  # 'payment' or 'subscription'

    # Get line items to determine product
    line_items = data.get("line_items", {}).get("data", [])
    if not line_items:
        # Try to expand line items
        session_id = data.get("id")
        if session_id:
            try:
                session = stripe.checkout.Session.retrieve(
                    session_id,
                    expand=["line_items"]
                )
                line_items = session.line_items.data if session.line_items else []
            except Exception as e:
                logger.error(f"Could not retrieve line items: {e}")

    # Determine tokens to add
    tokens_to_add = 0
    plan_type = None

    for item in line_items:
        price_id = item.get("price", {}).get("id") if isinstance(item.get("price"), dict) else item.get("price")
        product_key = PRICE_TO_PRODUCT.get(price_id)

        # Fallback: try to detect from product name/description
        if not product_key:
            product_name = (item.get("description") or "").lower()
            for key in PRODUCT_TOKENS.keys():
                if key in product_name:
                    product_key = key
                    break

        if product_key:
            tokens_to_add += PRODUCT_TOKENS.get(product_key, 0)
            if product_key in ["prototyper", "growth", "scale"]:
                plan_type = product_key

    # If we couldn't determine tokens, try metadata
    if tokens_to_add == 0:
        metadata = data.get("metadata", {})
        tokens_to_add = int(metadata.get("tokens", 0))
        plan_type = metadata.get("plan_type")

    if tokens_to_add > 0 and customer_id:
        logger.info(f"[{event_id}] Adding {tokens_to_add} tokens to customer {customer_id}")

        await add_tokens(AddTokensRequest(
            user_id=customer_id,
            amount=tokens_to_add,
            transaction_type="subscription" if mode == "subscription" else "purchase",
            description=f"Checkout completed - {plan_type or 'token purchase'}",
            stripe_payment_id=payment_id,
            email=customer_email,
            plan_type=plan_type
        ))
        return (tokens_to_add, customer_id)
    else:
        logger.warning(f"[{event_id}] Could not determine tokens for checkout {data.get('id')}")
        return (0, customer_id)


async def handle_invoice_paid(event_id: str, data: Dict[str, Any]) -> Tuple[int, Optional[str]]:
    """
    Handle subscription renewal - add monthly tokens.
    Returns: (tokens_credited, customer_id)
    """
    from api.tokens import add_tokens, AddTokensRequest

    customer_id = data.get("customer")
    customer_email = data.get("customer_email")
    invoice_id = data.get("id")
    billing_reason = data.get("billing_reason")  # 'subscription_cycle', 'subscription_create', etc.

    # Skip initial subscription (handled by checkout.session.completed)
    if billing_reason == "subscription_create":
        logger.info(f"[{event_id}] Skipping invoice.paid for initial subscription (handled by checkout)")
        return (0, customer_id)

    # Get subscription details
    subscription_id = data.get("subscription")
    if subscription_id:
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            items = subscription.get("items", {}).get("data", [])

            tokens_to_add = 0
            plan_type = None

            for item in items:
                price_id = item.get("price", {}).get("id")
                product_key = PRICE_TO_PRODUCT.get(price_id)

                if product_key:
                    tokens_to_add += PRODUCT_TOKENS.get(product_key, 0)
                    if product_key in ["prototyper", "growth", "scale"]:
                        plan_type = product_key

            if tokens_to_add > 0:
                logger.info(f"[{event_id}] Renewal: Adding {tokens_to_add} tokens to customer {customer_id}")

                await add_tokens(AddTokensRequest(
                    user_id=customer_id,
                    amount=tokens_to_add,
                    transaction_type="subscription",
                    description=f"Subscription renewal - {plan_type}",
                    stripe_payment_id=invoice_id,
                    email=customer_email,
                    plan_type=plan_type
                ))
                return (tokens_to_add, customer_id)
        except Exception as e:
            logger.error(f"[{event_id}] Error processing invoice renewal: {e}")

    return (0, customer_id)


async def handle_subscription_deleted(event_id: str, data: Dict[str, Any]) -> Tuple[int, Optional[str]]:
    """
    Handle subscription cancellation - clear plan type.
    Returns: (0, customer_id) - no tokens credited for cancellation
    """
    customer_id = data.get("customer")

    if not customer_id:
        return (0, None)

    logger.info(f"[{event_id}] Subscription deleted for customer {customer_id}")

    # Update user's plan_type to NULL
    try:
        supabase = get_supabase_client()
        if supabase:
            supabase.table("user_tokens").update({
                "plan_type": None
            }).eq("user_id", customer_id).execute()

            logger.info(f"[{event_id}] Cleared plan_type for customer {customer_id}")
    except Exception as e:
        logger.error(f"[{event_id}] Error clearing plan type: {e}")

    return (0, customer_id)


# ============================================================================
# HELPER: Manual token grant (for admin/testing)
# ============================================================================

@router.post("/stripe/manual-grant")
async def manual_token_grant(
    customer_id: str,
    tokens: int,
    plan_type: Optional[str] = None,
    description: str = "Manual grant"
):
    """Manually grant tokens to a user (admin endpoint)."""
    from api.tokens import add_tokens, AddTokensRequest

    await add_tokens(AddTokensRequest(
        user_id=customer_id,
        amount=tokens,
        transaction_type="purchase",
        description=description,
        plan_type=plan_type
    ))

    return {"status": "ok", "tokens_added": tokens, "customer_id": customer_id}


# ============================================================================
# ADMIN: Create idempotency table
# ============================================================================

@router.post("/stripe/admin/create-idempotency-table")
async def create_idempotency_table():
    """
    Create the processed_stripe_events table for idempotency tracking.
    Run this ONCE to set up the table in Supabase.

    Table schema:
    - event_id: TEXT PRIMARY KEY (Stripe event ID like evt_xxx)
    - event_type: TEXT NOT NULL (checkout.session.completed, invoice.paid, etc.)
    - processed_at: TIMESTAMPTZ DEFAULT NOW()
    - customer_id: TEXT (Stripe customer ID)
    - amount_tokens: INTEGER (tokens credited, 0 for non-credit events)
    """
    try:
        supabase = get_supabase_client()
        if not supabase:
            return {"status": "error", "message": "Supabase not configured"}

        # Try to create a test record to verify table exists
        # If it fails, we'll return instructions for manual table creation
        try:
            # First, try to query the table
            result = supabase.table("processed_stripe_events").select("event_id").limit(1).execute()
            return {
                "status": "ok",
                "message": "Table processed_stripe_events already exists",
                "row_count": len(result.data) if result.data else 0
            }
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
                # Table doesn't exist - return SQL to create it
                return {
                    "status": "table_missing",
                    "message": "Table does not exist. Please create it in Supabase SQL Editor.",
                    "sql": """
-- Run this in Supabase SQL Editor:
CREATE TABLE IF NOT EXISTS processed_stripe_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    customer_id TEXT,
    amount_tokens INTEGER DEFAULT 0
);

-- Add index for faster lookups
CREATE INDEX IF NOT EXISTS idx_processed_stripe_events_type ON processed_stripe_events(event_type);
CREATE INDEX IF NOT EXISTS idx_processed_stripe_events_customer ON processed_stripe_events(customer_id);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE processed_stripe_events ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
CREATE POLICY "Service role access" ON processed_stripe_events
    FOR ALL USING (true) WITH CHECK (true);
"""
                }
            raise

    except Exception as e:
        logger.error(f"Error checking idempotency table: {e}")
        return {"status": "error", "message": str(e)}


@router.get("/stripe/admin/idempotency-status")
async def idempotency_status():
    """Check the status of the idempotency system."""
    try:
        supabase = get_supabase_client()
        if not supabase:
            return {
                "status": "not_configured",
                "supabase_configured": False,
                "message": "Supabase not configured - idempotency checks disabled"
            }

        # Try to count records
        result = supabase.table("processed_stripe_events").select("event_id", count="exact").execute()

        return {
            "status": "ok",
            "supabase_configured": True,
            "table_exists": True,
            "total_events_processed": result.count if hasattr(result, 'count') else len(result.data),
            "message": "Idempotency system is active"
        }
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg.lower():
            return {
                "status": "table_missing",
                "supabase_configured": True,
                "table_exists": False,
                "message": "Table processed_stripe_events does not exist. Call /stripe/admin/create-idempotency-table for setup instructions."
            }
        return {"status": "error", "message": str(e)}
