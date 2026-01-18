"""
================================================================================
STRIPE WEBHOOK HANDLER - Genesis Backend
================================================================================
Receives and processes Stripe payment events with Nexus phase routing.

Endpoint: POST /api/webhooks/stripe
Health:   GET  /api/webhooks/stripe/health

================================================================================
Author: Barrios A2I | January 2026
================================================================================
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None

# Configure logging
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/webhooks", tags=["Webhooks"])

# Stripe webhook secret from environment
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# =============================================================================
# NEXUS PHASE MAPPING
# =============================================================================
# Maps Stripe events to Nexus pipeline phases for downstream processing

EVENT_TO_NEXUS_PHASE: Dict[str, Dict[str, Any]] = {
    # Checkout Events (Phase 1: Lead Conversion)
    "checkout.session.completed": {"phase": 1, "action": "convert_lead"},
    "checkout.session.expired": {"phase": 1, "action": "expire_checkout"},
    "checkout.session.async_payment_succeeded": {"phase": 1, "action": "async_payment_success"},
    "checkout.session.async_payment_failed": {"phase": 1, "action": "async_payment_failed"},

    # Customer Events (Phase 2: Customer Management)
    "customer.created": {"phase": 2, "action": "create_customer_record"},
    "customer.updated": {"phase": 2, "action": "update_customer_record"},
    "customer.deleted": {"phase": 2, "action": "delete_customer_record"},

    # Subscription Events (Phase 3: Service Provisioning)
    "customer.subscription.created": {"phase": 3, "action": "provision_services"},
    "customer.subscription.updated": {"phase": 3, "action": "update_subscription"},
    "customer.subscription.paused": {"phase": 3, "action": "pause_services"},
    "customer.subscription.resumed": {"phase": 3, "action": "resume_services"},
    "customer.subscription.pending_update_applied": {"phase": 3, "action": "apply_pending_update"},
    "customer.subscription.pending_update_expired": {"phase": 3, "action": "expire_pending_update"},
    "customer.subscription.trial_will_end": {"phase": 3, "action": "notify_trial_ending"},

    # Invoice Events (Phase 4: Payment Recording)
    "invoice.paid": {"phase": 4, "action": "record_payment"},
    "invoice.payment_succeeded": {"phase": 4, "action": "record_payment"},
    "invoice.finalized": {"phase": 4, "action": "finalize_invoice"},
    "invoice.created": {"phase": 4, "action": "create_invoice_record"},

    # Payment Intent Events (Phase 5: Payment Processing)
    "payment_intent.succeeded": {"phase": 5, "action": "confirm_payment"},
    "payment_intent.created": {"phase": 5, "action": "track_payment_intent"},

    # Charge Events (Phase 6: Charge Management)
    "charge.succeeded": {"phase": 6, "action": "record_charge"},
    "charge.refunded": {"phase": 6, "action": "process_refund"},

    # Failed Payment Events (Phase 10: Dunning)
    "invoice.payment_failed": {"phase": 10, "action": "initiate_dunning"},

    # Subscription Cancellation (Phase 13: Deprovisioning)
    "customer.subscription.deleted": {"phase": 13, "action": "deprovision_services"},
}


# =============================================================================
# WEBHOOK HANDLER
# =============================================================================

@router.post("/stripe")
async def stripe_webhook(request: Request):
    """
    Handle incoming Stripe webhook events.

    Verifies signature, parses event, and routes to appropriate Nexus phase.
    """
    if not STRIPE_AVAILABLE:
        logger.error("Stripe library not installed")
        raise HTTPException(status_code=500, detail="Stripe not configured")

    if not STRIPE_WEBHOOK_SECRET:
        logger.error("STRIPE_WEBHOOK_SECRET not configured")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    # Get raw body and signature
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        logger.warning("Missing Stripe signature header")
        raise HTTPException(status_code=400, detail="Missing signature")

    # Verify webhook signature
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Extract event details
    event_type = event.get("type")
    event_id = event.get("id")
    event_data = event.get("data", {}).get("object", {})

    # Log received event
    logger.info(
        f"Received Stripe event: {event_type}",
        extra={
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Get Nexus phase mapping
    phase_info = EVENT_TO_NEXUS_PHASE.get(event_type)

    if phase_info:
        nexus_phase = phase_info["phase"]
        action = phase_info["action"]

        logger.info(
            f"Routing to Nexus Phase {nexus_phase}: {action}",
            extra={
                "event_id": event_id,
                "event_type": event_type,
                "nexus_phase": nexus_phase,
                "action": action,
            }
        )

        # Process the event based on phase
        await process_nexus_phase(
            phase=nexus_phase,
            action=action,
            event_type=event_type,
            event_id=event_id,
            event_data=event_data
        )
    else:
        logger.info(
            f"Unhandled event type: {event_type}",
            extra={"event_id": event_id, "event_type": event_type}
        )

    # Return 200 to acknowledge receipt
    return JSONResponse(
        status_code=200,
        content={
            "status": "received",
            "event_id": event_id,
            "event_type": event_type,
            "nexus_phase": phase_info["phase"] if phase_info else None,
            "action": phase_info["action"] if phase_info else None,
        }
    )


async def process_nexus_phase(
    phase: int,
    action: str,
    event_type: str,
    event_id: str,
    event_data: Dict[str, Any]
) -> None:
    """
    Process event based on Nexus phase routing.

    This is a placeholder for actual business logic integration.
    In production, this would trigger appropriate downstream services.
    """
    # Phase 1: Lead Conversion
    if phase == 1:
        if action == "convert_lead":
            customer_email = event_data.get("customer_email")
            amount_total = event_data.get("amount_total", 0)
            logger.info(f"Converting lead: {customer_email}, amount: {amount_total/100:.2f}")

    # Phase 2: Customer Management
    elif phase == 2:
        customer_id = event_data.get("id")
        email = event_data.get("email")
        logger.info(f"Customer event: {action} - {customer_id} ({email})")

    # Phase 3: Service Provisioning
    elif phase == 3:
        subscription_id = event_data.get("id")
        status = event_data.get("status")
        logger.info(f"Subscription event: {action} - {subscription_id} (status: {status})")

    # Phase 4: Payment Recording
    elif phase == 4:
        invoice_id = event_data.get("id")
        amount_paid = event_data.get("amount_paid", 0)
        logger.info(f"Invoice event: {action} - {invoice_id}, amount: {amount_paid/100:.2f}")

    # Phase 5: Payment Processing
    elif phase == 5:
        payment_intent_id = event_data.get("id")
        amount = event_data.get("amount", 0)
        logger.info(f"Payment intent: {action} - {payment_intent_id}, amount: {amount/100:.2f}")

    # Phase 6: Charge Management
    elif phase == 6:
        charge_id = event_data.get("id")
        amount = event_data.get("amount", 0)
        logger.info(f"Charge event: {action} - {charge_id}, amount: {amount/100:.2f}")

    # Phase 10: Dunning (Failed Payments)
    elif phase == 10:
        invoice_id = event_data.get("id")
        attempt_count = event_data.get("attempt_count", 0)
        logger.warning(f"Payment failed: {invoice_id}, attempt: {attempt_count}")

    # Phase 13: Deprovisioning
    elif phase == 13:
        subscription_id = event_data.get("id")
        logger.info(f"Deprovisioning services for subscription: {subscription_id}")


# =============================================================================
# HEALTH CHECK
# =============================================================================

@router.get("/stripe/health")
async def stripe_webhook_health():
    """
    Health check for Stripe webhook integration.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "stripe-webhook",
            "stripe_available": STRIPE_AVAILABLE,
            "webhook_secret_configured": bool(STRIPE_WEBHOOK_SECRET),
            "supported_events": len(EVENT_TO_NEXUS_PHASE),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
