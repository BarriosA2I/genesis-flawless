"""
Stripe Webhook Token Handler for Barrios A2I Commercial Lab
Adds tokens based on Stripe checkout/subscription events.
"""

import os
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, Header
import stripe

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", os.getenv("STRIPE_SECRET_KEY", ""))

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY

# Token amounts per product
PRODUCT_TOKENS = {
    # Subscription tiers (monthly)
    "starter": 8,
    "creator": 16,
    "growth": 32,
    "scale": 64,
    # One-time token packs
    "pack_8": 8,
    "pack_16": 16,
    "pack_32": 32,
    # Lab test
    "lab_test": 8,
}

# Price ID to product mapping (Stripe price IDs)
PRICE_TO_PRODUCT = {
    # Subscription tiers (monthly)
    "price_1SuDIPLyFGkLiU4CWVBwoBAR": "starter",   # $449/mo - 8 tokens
    "price_1SuDJPLyFGkLiU4Ck2CzcwcX": "creator",   # $899/mo - 16 tokens
    "price_1SuDMRLyFGkLiU4Ci4if35Dv": "growth",    # $1,699/mo - 32 tokens
    "price_1SuDNGLyFGkLiU4CS6eYsq6F": "scale",     # $3,199/mo - 64 tokens
    # One-time token packs
    "price_1SuDP7LyFGkLiU4CPQEhLnal": "pack_8",    # $449 - 8 tokens
    "price_1SuDR8LyFGkLiU4Ci907l5b2": "pack_16",   # $799 - 16 tokens
    "price_1SuDS6LyFGkLiU4CGLuNK8wS": "pack_32",   # $1,499 - 32 tokens
    # Lab test
    "price_1SuDOBLyFGkLiU4Ct7F1xeZo": "lab_test",  # $500 - 8 tokens
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
    """
    # Get raw body
    payload = await request.body()

    # Verify webhook signature
    if STRIPE_WEBHOOK_SECRET and stripe_signature:
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
        # For testing without signature verification
        import json
        event = json.loads(payload)
        logger.warning("Processing webhook without signature verification")

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    logger.info(f"Received Stripe webhook: {event_type}")

    try:
        if event_type == "checkout.session.completed":
            await handle_checkout_completed(data)
        elif event_type == "invoice.paid":
            await handle_invoice_paid(data)
        elif event_type == "customer.subscription.deleted":
            await handle_subscription_deleted(data)
        else:
            logger.debug(f"Unhandled event type: {event_type}")

        return {"status": "ok", "event_type": event_type}
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EVENT HANDLERS
# ============================================================================

async def handle_checkout_completed(data: Dict[str, Any]):
    """Handle successful checkout - add tokens to user."""
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
            if product_key in ["starter", "creator", "growth", "scale"]:
                plan_type = product_key

    # If we couldn't determine tokens, try metadata
    if tokens_to_add == 0:
        metadata = data.get("metadata", {})
        tokens_to_add = int(metadata.get("tokens", 0))
        plan_type = metadata.get("plan_type")

    if tokens_to_add > 0 and customer_id:
        logger.info(f"Adding {tokens_to_add} tokens to customer {customer_id}")

        await add_tokens(AddTokensRequest(
            user_id=customer_id,
            amount=tokens_to_add,
            transaction_type="subscription" if mode == "subscription" else "purchase",
            description=f"Checkout completed - {plan_type or 'token purchase'}",
            stripe_payment_id=payment_id,
            email=customer_email,
            plan_type=plan_type
        ))
    else:
        logger.warning(f"Could not determine tokens for checkout {data.get('id')}")


async def handle_invoice_paid(data: Dict[str, Any]):
    """Handle subscription renewal - add monthly tokens."""
    from api.tokens import add_tokens, AddTokensRequest

    customer_id = data.get("customer")
    customer_email = data.get("customer_email")
    invoice_id = data.get("id")
    billing_reason = data.get("billing_reason")  # 'subscription_cycle', 'subscription_create', etc.

    # Skip initial subscription (handled by checkout.session.completed)
    if billing_reason == "subscription_create":
        logger.info("Skipping invoice.paid for initial subscription (handled by checkout)")
        return

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
                    if product_key in ["starter", "creator", "growth", "scale"]:
                        plan_type = product_key

            if tokens_to_add > 0:
                logger.info(f"Renewal: Adding {tokens_to_add} tokens to customer {customer_id}")

                await add_tokens(AddTokensRequest(
                    user_id=customer_id,
                    amount=tokens_to_add,
                    transaction_type="subscription",
                    description=f"Subscription renewal - {plan_type}",
                    stripe_payment_id=invoice_id,
                    email=customer_email,
                    plan_type=plan_type
                ))
        except Exception as e:
            logger.error(f"Error processing invoice renewal: {e}")


async def handle_subscription_deleted(data: Dict[str, Any]):
    """Handle subscription cancellation - clear plan type."""
    from supabase import create_client

    customer_id = data.get("customer")

    if not customer_id:
        return

    logger.info(f"Subscription deleted for customer {customer_id}")

    # Update user's plan_type to NULL
    try:
        supabase_url = os.getenv("SUPABASE_URL", "")
        supabase_key = os.getenv("SUPABASE_KEY", "")

        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
            supabase.table("user_tokens").update({
                "plan_type": None
            }).eq("user_id", customer_id).execute()

            logger.info(f"Cleared plan_type for customer {customer_id}")
    except Exception as e:
        logger.error(f"Error clearing plan type: {e}")


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
