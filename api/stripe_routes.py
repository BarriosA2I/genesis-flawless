"""
================================================================================
NEXUS ORCHESTRATOR - Integrated Webhook Handler
================================================================================
Production-grade Stripe webhook handler with full phase handler integration.
Features: Idempotency, signature verification, async processing, observability.
================================================================================
"""

import os
import logging
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None

from fastapi import APIRouter, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import select, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Local imports
try:
    from services.nexus_phases import (
        NexusOrchestrator,
        HandlerResult,
        NexusPhase,
        CustomerStatus
    )
    from services.event_bus import create_event_bus, InMemoryEventBus
    from services.notification_service import NotificationService, OpsAlertService
    from models.customer_lifecycle import Base, WebhookEvent
    NEXUS_AVAILABLE = True
except ImportError as e:
    NEXUS_AVAILABLE = False
    logging.warning(f"Nexus services not available: {e}")

logger = logging.getLogger("nexus.webhook")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration from environment variables."""
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", os.getenv("STRIPE_SECRET_KEY", ""))
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    USE_RABBITMQ = os.getenv("USE_RABBITMQ", "false").lower() == "true"
    SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
    SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
    SLACK_OPS_WEBHOOK_URL = os.getenv("SLACK_OPS_WEBHOOK_URL", "")


# Initialize Stripe
if STRIPE_AVAILABLE and Config.STRIPE_API_KEY:
    stripe.api_key = Config.STRIPE_API_KEY


# =============================================================================
# DATABASE SESSION (Lazy initialization)
# =============================================================================

engine = None
async_session_factory = None


async def init_database():
    """Initialize database connection pool."""
    global engine, async_session_factory

    if not SQLALCHEMY_AVAILABLE or not Config.DATABASE_URL:
        logger.warning("Database not configured - running in memory-only mode")
        return False

    if engine is None:
        # Convert postgres:// to postgresql+asyncpg://
        db_url = Config.DATABASE_URL
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        try:
            engine = create_async_engine(
                db_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                echo=False,
                connect_args={
                    "statement_cache_size": 0,  # Required for pgbouncer compatibility
                }
            )
            async_session_factory = async_sessionmaker(
                engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Create tables if they don't exist
            if NEXUS_AVAILABLE:
                async with engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)

            logger.info("Database initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    return True


async def get_db_session():
    """Get database session."""
    if async_session_factory is None:
        db_ready = await init_database()
        if not db_ready:
            yield None
            return

    if async_session_factory:
        async with async_session_factory() as session:
            yield session
    else:
        yield None


# =============================================================================
# SERVICE INITIALIZATION
# =============================================================================

event_bus = None
notification_service = None
ops_alert_service = None
services_initialized = False


async def init_services():
    """Initialize all services."""
    global event_bus, notification_service, ops_alert_service, services_initialized

    if services_initialized:
        return

    if not NEXUS_AVAILABLE:
        logger.warning("Nexus services not available - using basic mode")
        services_initialized = True
        return

    try:
        # Initialize database first - required for phase routing
        db_ready = await init_database()
        if not db_ready:
            logger.warning("Database not initialized - phase routing will be disabled")
        else:
            logger.info("Database ready for phase routing")

        # Event bus (in-memory by default)
        event_bus = await create_event_bus(
            use_rabbitmq=Config.USE_RABBITMQ,
            rabbitmq_url=Config.RABBITMQ_URL
        )

        # Notification service
        notification_service = NotificationService(
            sendgrid_api_key=Config.SENDGRID_API_KEY,
            slack_webhook_url=Config.SLACK_WEBHOOK_URL
        )

        # Ops alerts
        ops_alert_service = OpsAlertService(
            slack_webhook_url=Config.SLACK_OPS_WEBHOOK_URL
        )

        services_initialized = True
        logger.info("Nexus services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        services_initialized = True  # Mark as done to avoid retrying


# =============================================================================
# IDEMPOTENCY
# =============================================================================

# In-memory cache for idempotency when database is not available
_processed_events: Dict[str, datetime] = {}


async def check_idempotency(db: Optional[AsyncSession], event_id: str) -> bool:
    """
    Check if event has already been processed.
    Returns True if event should be skipped (already processed).
    """
    if db is None:
        # Use in-memory cache
        return event_id in _processed_events

    try:
        result = await db.execute(
            text("SELECT id FROM nexus_webhook_events WHERE stripe_event_id = :event_id"),
            {"event_id": event_id}
        )
        return result.scalar() is not None
    except Exception as e:
        logger.warning(f"Idempotency check failed: {e}")
        return event_id in _processed_events


async def record_event_processing(
    db: Optional[AsyncSession],
    event_id: str,
    event_type: str,
    payload: Dict[str, Any],
    success: bool,
    error: Optional[str] = None,
    processing_time_ms: int = 0,
    phase_triggered: Optional[str] = None
):
    """Record event processing for idempotency and audit using ORM."""
    # Always update in-memory cache
    _processed_events[event_id] = datetime.utcnow()

    # Clean up old entries (keep last 1000)
    if len(_processed_events) > 1000:
        oldest = sorted(_processed_events.keys(), key=lambda k: _processed_events[k])[:500]
        for k in oldest:
            del _processed_events[k]

    if db is None:
        return

    try:
        # Extract customer ID from payload
        customer_id = (
            payload.get("customer_id") or
            payload.get("customer") or
            payload.get("raw_object", {}).get("customer")
        )

        # Check if event already exists
        existing = await db.execute(
            select(WebhookEvent).where(WebhookEvent.stripe_event_id == event_id)
        )
        existing_event = existing.scalar_one_or_none()

        if existing_event:
            # Update existing event
            existing_event.processed_at = datetime.utcnow()
            existing_event.success = success
            existing_event.error = error[:1000] if error else None
            existing_event.processing_time_ms = processing_time_ms
            existing_event.phase_triggered = phase_triggered
        else:
            # Create new event
            webhook_event = WebhookEvent(
                stripe_event_id=event_id,
                event_type=event_type,
                stripe_customer_id=customer_id,
                payload=payload,
                processed_at=datetime.utcnow(),
                success=success,
                error=error[:1000] if error else None,
                phase_triggered=phase_triggered,
                processing_time_ms=processing_time_ms
            )
            db.add(webhook_event)

        await db.commit()
        logger.info(f"Recorded webhook event: {event_id} ({event_type}) success={success}")
    except Exception as e:
        logger.warning(f"Failed to record event processing: {e}")


# =============================================================================
# EVENT DATA EXTRACTION
# =============================================================================

def extract_event_data(event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant data from Stripe event based on type.
    Normalizes data structure for phase handlers.
    """
    obj = data.get("object", {})

    # Common fields
    extracted = {
        "raw_object": obj,
    }

    # Checkout session events
    if event_type.startswith("checkout.session"):
        extracted.update({
            "session_id": obj.get("id"),
            "customer_id": obj.get("customer"),
            "customer_email": obj.get("customer_email") or obj.get("customer_details", {}).get("email"),
            "subscription_id": obj.get("subscription"),
            "payment_intent_id": obj.get("payment_intent"),
            "amount_total": obj.get("amount_total", 0),
            "currency": obj.get("currency", "usd"),
            "status": obj.get("status"),
            "mode": obj.get("mode"),
            "metadata": obj.get("metadata", {}),
        })

    # Customer events
    elif event_type.startswith("customer."):
        if "subscription" in event_type:
            extracted.update({
                "subscription_id": obj.get("id"),
                "customer_id": obj.get("customer"),
                "status": obj.get("status"),
                "current_period_start": obj.get("current_period_start"),
                "current_period_end": obj.get("current_period_end"),
                "cancel_at_period_end": obj.get("cancel_at_period_end"),
                "canceled_at": obj.get("canceled_at"),
                "items": [
                    {
                        "price_id": item.get("price", {}).get("id"),
                        "product_id": item.get("price", {}).get("product"),
                        "quantity": item.get("quantity"),
                    }
                    for item in obj.get("items", {}).get("data", [])
                ],
                "metadata": obj.get("metadata", {}),
            })
        else:
            extracted.update({
                "customer_id": obj.get("id"),
                "email": obj.get("email"),
                "name": obj.get("name"),
                "phone": obj.get("phone"),
                "metadata": obj.get("metadata", {}),
            })

    # Invoice events
    elif event_type.startswith("invoice."):
        extracted.update({
            "invoice_id": obj.get("id"),
            "customer_id": obj.get("customer"),
            "subscription_id": obj.get("subscription"),
            "amount_due": obj.get("amount_due", 0),
            "amount_paid": obj.get("amount_paid", 0),
            "amount_remaining": obj.get("amount_remaining", 0),
            "status": obj.get("status"),
            "billing_reason": obj.get("billing_reason"),
            "attempt_count": obj.get("attempt_count", 0),
            "next_payment_attempt": obj.get("next_payment_attempt"),
            "paid": obj.get("paid", False),
        })

    # Payment intent events
    elif event_type.startswith("payment_intent."):
        extracted.update({
            "payment_intent_id": obj.get("id"),
            "customer_id": obj.get("customer"),
            "amount": obj.get("amount", 0),
            "currency": obj.get("currency", "usd"),
            "status": obj.get("status"),
            "last_payment_error": obj.get("last_payment_error"),
            "metadata": obj.get("metadata", {}),
        })

    # Charge events
    elif event_type.startswith("charge."):
        if "dispute" in event_type:
            extracted.update({
                "dispute_id": obj.get("id"),
                "charge_id": obj.get("charge"),
                "amount": obj.get("amount", 0),
                "reason": obj.get("reason"),
                "status": obj.get("status"),
            })
        else:
            extracted.update({
                "charge_id": obj.get("id"),
                "customer_id": obj.get("customer"),
                "payment_intent_id": obj.get("payment_intent"),
                "amount": obj.get("amount", 0),
                "amount_refunded": obj.get("amount_refunded", 0),
                "currency": obj.get("currency", "usd"),
                "status": obj.get("status"),
                "paid": obj.get("paid", False),
                "refunded": obj.get("refunded", False),
                "failure_code": obj.get("failure_code"),
                "failure_message": obj.get("failure_message"),
            })

    return extracted


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/api/webhooks", tags=["Webhooks"])


@router.get("/stripe/health")
async def health_check():
    """Health check endpoint for the webhook handler."""
    return {
        "status": "healthy",
        "service": "nexus-stripe-webhook",
        "timestamp": datetime.utcnow().isoformat(),
        "stripe_available": STRIPE_AVAILABLE and bool(Config.STRIPE_API_KEY),
        "webhook_secret_configured": bool(Config.STRIPE_WEBHOOK_SECRET),
        "database_url_configured": bool(Config.DATABASE_URL),
        "nexus_available": NEXUS_AVAILABLE,
        "event_bus": "rabbitmq" if Config.USE_RABBITMQ else "in_memory",
        "supported_events": 24
    }


@router.post("/stripe")
async def handle_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Main Stripe webhook endpoint.
    Verifies signature, extracts event data, routes to phase handlers.
    """
    start_time = time.time()

    # Initialize services if needed
    if not services_initialized:
        await init_services()

    # Get database session
    db = None
    if SQLALCHEMY_AVAILABLE and async_session_factory:
        async with async_session_factory() as session:
            db = session
            return await _process_webhook(request, background_tasks, db, start_time)
    else:
        return await _process_webhook(request, background_tasks, None, start_time)


async def _process_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Optional[AsyncSession],
    start_time: float
):
    """Process the webhook with optional database support."""

    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=500, detail="Stripe not configured")

    # Get raw body and signature
    try:
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature", "")
    except Exception as e:
        logger.error(f"Failed to read request: {e}")
        raise HTTPException(status_code=400, detail="Invalid request")

    # Verify signature
    try:
        if Config.STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, Config.STRIPE_WEBHOOK_SECRET
            )
        else:
            import json
            event = json.loads(payload)
            logger.warning("Webhook signature verification disabled!")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Failed to parse event: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")

    # Extract event info
    event_id = event.get("id", f"evt_{hashlib.md5(payload).hexdigest()[:16]}")
    event_type = event.get("type", "unknown")
    event_data = event.get("data", {})

    logger.info(f"Received webhook: {event_type} (id={event_id})")

    # Idempotency check
    if await check_idempotency(db, event_id):
        logger.info(f"Event already processed: {event_id}")
        return JSONResponse(
            status_code=200,
            content={"status": "already_processed", "event_id": event_id}
        )

    # Extract normalized event data
    extracted_data = extract_event_data(event_type, event_data)

    # Process event
    try:
        if NEXUS_AVAILABLE and db is not None:
            # Use full Nexus Orchestrator
            orchestrator = NexusOrchestrator(
                db=db,
                event_bus=event_bus,
                notification_service=notification_service
            )

            result = await orchestrator.route_event(
                event_type=event_type,
                event_id=event_id,
                event_data=extracted_data
            )

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Record processing
            await record_event_processing(
                db=db,
                event_id=event_id,
                event_type=event_type,
                payload=extracted_data,
                success=result.success,
                error=result.error,
                processing_time_ms=processing_time_ms,
                phase_triggered=result.to_phase.value if result.to_phase else None
            )

            # Send ops alerts for important events
            if result.success and ops_alert_service:
                background_tasks.add_task(
                    send_ops_alerts,
                    event_type=event_type,
                    result=result,
                    extracted_data=extracted_data
                )

            logger.info(
                f"Processed {event_type}: success={result.success}, "
                f"phase={result.to_phase.value}, duration={processing_time_ms}ms"
            )

            return JSONResponse(
                status_code=200,
                content={
                    "status": "processed",
                    "event_id": event_id,
                    "event_type": event_type,
                    "handler_action": result.action,
                    "success": result.success,
                    "phase": result.to_phase.value,
                    "processing_time_ms": processing_time_ms
                }
            )
        else:
            # Basic mode - just log the event
            processing_time_ms = int((time.time() - start_time) * 1000)

            await record_event_processing(
                db=db,
                event_id=event_id,
                event_type=event_type,
                payload=extracted_data,
                success=True,
                processing_time_ms=processing_time_ms,
                phase_triggered="basic_mode"
            )

            logger.info(f"Processed {event_type} in basic mode")

            return JSONResponse(
                status_code=200,
                content={
                    "status": "processed",
                    "event_id": event_id,
                    "event_type": event_type,
                    "mode": "basic",
                    "processing_time_ms": processing_time_ms
                }
            )

    except Exception as e:
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.error(f"Failed to process {event_type}: {e}")

        # Record failure
        await record_event_processing(
            db=db,
            event_id=event_id,
            event_type=event_type,
            payload=extracted_data,
            success=False,
            error=str(e),
            processing_time_ms=processing_time_ms,
            phase_triggered=None
        )

        # Return 200 to prevent Stripe retries (we'll handle retry ourselves)
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "event_id": event_id,
                "event_type": event_type,
                "error": str(e),
                "processing_time_ms": processing_time_ms
            }
        )


async def send_ops_alerts(
    event_type: str,
    result: Any,
    extracted_data: Dict[str, Any]
):
    """Send relevant ops alerts based on event type."""
    if not ops_alert_service:
        return

    try:
        if event_type == "invoice.payment_failed":
            await ops_alert_service.alert_payment_failed(
                customer_email=extracted_data.get("customer_id", "unknown"),
                amount=extracted_data.get("amount_due", 0),
                attempt=extracted_data.get("attempt_count", 1)
            )

        elif event_type == "customer.subscription.deleted":
            if hasattr(result, 'metadata') and result.metadata.get("lifetime_value", 0) > 0:
                await ops_alert_service.alert_customer_churned(
                    customer_email=result.customer_id,
                    lifetime_value=result.metadata.get("lifetime_value", 0)
                )

        elif event_type == "checkout.session.completed":
            amount = extracted_data.get("amount_total", 0)
            if amount >= 50000:  # $500+ is high value
                await ops_alert_service.alert_high_value_signup(
                    customer_email=extracted_data.get("customer_email", "unknown"),
                    tier=result.metadata.get("tier", "unknown") if hasattr(result, 'metadata') else "unknown",
                    mrr=amount / 100
                )

    except Exception as e:
        logger.error(f"Failed to send ops alert: {e}")


# =============================================================================
# METRICS ENDPOINT
# =============================================================================

@router.get("/stripe/metrics")
async def get_metrics():
    """Get webhook processing metrics."""
    # Initialize database if needed
    if not services_initialized:
        await init_services()

    if async_session_factory is None:
        return {
            "mode": "in_memory",
            "events_cached": len(_processed_events),
            "timestamp": datetime.utcnow().isoformat()
        }

    async with async_session_factory() as db:
        try:
            # Get event counts by type
            events_by_type = await db.execute(text("""
                SELECT event_type,
                       COUNT(*) as total,
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                       AVG(processing_time_ms) as avg_time_ms
                FROM nexus_webhook_events
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY event_type
                ORDER BY total DESC
            """))

            # Get customer phase distribution
            phase_distribution = await db.execute(text("""
                SELECT current_phase, status, COUNT(*) as count
                FROM nexus_customers
                GROUP BY current_phase, status
            """))

            # Get recent errors
            recent_errors = await db.execute(text("""
                SELECT event_type, error, created_at
                FROM nexus_webhook_events
                WHERE success = false
                AND created_at > NOW() - INTERVAL '1 hour'
                ORDER BY created_at DESC
                LIMIT 10
            """))

            return {
                "events_last_24h": [dict(row._mapping) for row in events_by_type],
                "phase_distribution": [dict(row._mapping) for row in phase_distribution],
                "recent_errors": [dict(row._mapping) for row in recent_errors],
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "error": str(e),
                "mode": "in_memory",
                "events_cached": len(_processed_events),
                "timestamp": datetime.utcnow().isoformat()
            }


# =============================================================================
# MANUAL TRIGGER ENDPOINTS (For testing/ops)
# =============================================================================

class ManualTriggerRequest(BaseModel):
    customer_id: str
    action: str
    metadata: Dict[str, Any] = {}


@router.post("/stripe/manual-trigger")
async def manual_trigger(
    request_body: ManualTriggerRequest,
):
    """
    Manually trigger a phase handler action.
    Useful for testing and ops recovery.
    """
    if not NEXUS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Nexus services not available")

    if async_session_factory is None:
        await init_database()

    if async_session_factory is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    async with async_session_factory() as db:
        orchestrator = NexusOrchestrator(
            db=db,
            event_bus=event_bus,
            notification_service=notification_service
        )

        # Map action to event type
        action_event_map = {
            "convert_lead": "checkout.session.completed",
            "create_customer_record": "customer.created",
            "provision_services": "customer.subscription.created",
            "record_payment": "invoice.payment_succeeded",
            "initiate_dunning": "invoice.payment_failed",
            "deprovision_services": "customer.subscription.deleted",
        }

        event_type = action_event_map.get(request_body.action)
        if not event_type:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request_body.action}")

        result = await orchestrator.route_event(
            event_type=event_type,
            event_id=f"manual_{datetime.utcnow().timestamp()}",
            event_data={
                "customer_id": request_body.customer_id,
                **request_body.metadata
            }
        )

        return result.to_dict()


class TestEventRequest(BaseModel):
    """Request model for testing webhook handlers."""
    event_type: str
    event_data: Dict[str, Any] = {}


@router.post("/stripe/test-handler")
async def test_handler(request_body: TestEventRequest):
    """
    Test a webhook handler directly without signature verification.
    For development/testing only - validates handler routing works.
    """
    if not NEXUS_AVAILABLE:
        raise HTTPException(status_code=500, detail="Nexus services not available")

    if async_session_factory is None:
        await init_database()

    if async_session_factory is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    async with async_session_factory() as db:
        orchestrator = NexusOrchestrator(
            db=db,
            event_bus=event_bus,
            notification_service=notification_service
        )

        # Check if handler exists
        mapping = orchestrator.EVENT_HANDLER_MAP.get(request_body.event_type)
        if not mapping:
            return {
                "status": "error",
                "error": f"No handler mapped for event type: {request_body.event_type}",
                "available_events": list(orchestrator.EVENT_HANDLER_MAP.keys())
            }

        result = await orchestrator.route_event(
            event_type=request_body.event_type,
            event_id=f"test_{datetime.utcnow().timestamp()}",
            event_data=request_body.event_data
        )

        return {
            "status": "success" if result.success else "handler_error",
            "event_type": request_body.event_type,
            "handler_action": result.action,
            "result": result.to_dict()
        }


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@router.post("/stripe/admin/migrate")
async def run_migration():
    """
    Run database migration to create Nexus tables.
    Call this endpoint to ensure all required tables exist.
    """
    if not Config.DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")

    if not SQLALCHEMY_AVAILABLE:
        raise HTTPException(status_code=500, detail="SQLAlchemy not available")

    try:
        # Initialize database (creates tables if they don't exist)
        db_ready = await init_database()

        if not db_ready:
            raise HTTPException(
                status_code=500,
                detail="Database initialization failed - check logs"
            )

        return {
            "status": "success",
            "message": "Migration completed - Nexus tables created",
            "database_configured": True,
            "nexus_available": NEXUS_AVAILABLE,
            "async_session_factory_ready": async_session_factory is not None,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


@router.post("/stripe/admin/fix-schema")
async def fix_schema():
    """
    Add missing columns to existing tables.
    Safe to run multiple times - uses IF NOT EXISTS.
    """
    if async_session_factory is None:
        await init_database()

    if async_session_factory is None:
        raise HTTPException(status_code=500, detail="Database not available")

    fixes_applied = []

    async with async_session_factory() as db:
        try:
            # Add metadata column to nexus_customers if missing
            await db.execute(text("""
                ALTER TABLE nexus_customers
                ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb
            """))
            fixes_applied.append("nexus_customers.metadata")

            # Add metadata column to other tables if missing
            for table in ['nexus_phase_transitions', 'nexus_payments', 'nexus_entitlements', 'nexus_notifications']:
                await db.execute(text(f"""
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{{}}'::jsonb
                """))
                fixes_applied.append(f"{table}.metadata")

            # Add AT_RISK to customerstatus enum if missing
            try:
                await db.execute(text("""
                    ALTER TYPE customerstatus ADD VALUE IF NOT EXISTS 'at_risk'
                """))
                fixes_applied.append("customerstatus.at_risk")
            except Exception as e:
                # May fail if value already exists or in transaction
                logger.warning(f"Could not add at_risk to enum: {e}")

            await db.commit()

            return {
                "status": "success",
                "fixes_applied": fixes_applied,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Schema fix failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/stripe/admin/fix-enums")
async def fix_database_enums():
    """Add missing enum values to PostgreSQL types. Uses autocommit (required for ALTER TYPE)."""
    import os
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    results = []

    try:
        database_url = os.getenv("DATABASE_URL", "")
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgresql://") and "asyncpg" not in database_url:
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        temp_engine = create_async_engine(database_url, isolation_level="AUTOCOMMIT")

        async with temp_engine.connect() as conn:
            enum_updates = [
                ("customerstatus", "AT_RISK"),
                ("customerstatus", "PAST_DUE"),
            ]

            for enum_type, enum_value in enum_updates:
                try:
                    check = await conn.execute(text("""
                        SELECT 1 FROM pg_enum
                        WHERE enumlabel = :value
                        AND enumtypid = (SELECT oid FROM pg_type WHERE typname = :enum_type)
                    """), {"value": enum_value, "enum_type": enum_type})

                    if not check.fetchone():
                        await conn.execute(text(f"ALTER TYPE {enum_type} ADD VALUE '{enum_value}'"))
                        results.append({"enum": enum_type, "value": enum_value, "status": "added"})
                        logger.info(f"Added '{enum_value}' to {enum_type} enum")
                    else:
                        results.append({"enum": enum_type, "value": enum_value, "status": "exists"})

                except Exception as e:
                    results.append({"enum": enum_type, "value": enum_value, "status": "error", "error": str(e)})

        await temp_engine.dispose()

        return {
            "status": "success",
            "message": "Enum values processed",
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fix enums: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stripe/admin/status")
async def get_service_status():
    """
    Get detailed status of all webhook services.
    Useful for debugging why phase routing might not be working.
    """
    return {
        "services": {
            "stripe_available": STRIPE_AVAILABLE,
            "stripe_api_key_configured": bool(Config.STRIPE_API_KEY),
            "webhook_secret_configured": bool(Config.STRIPE_WEBHOOK_SECRET),
            "sqlalchemy_available": SQLALCHEMY_AVAILABLE,
            "database_url_configured": bool(Config.DATABASE_URL),
            "nexus_available": NEXUS_AVAILABLE,
            "services_initialized": services_initialized,
            "async_session_factory_ready": async_session_factory is not None,
            "engine_ready": engine is not None,
            "event_bus_type": "rabbitmq" if Config.USE_RABBITMQ else "in_memory",
            "event_bus_ready": event_bus is not None,
            "notification_service_ready": notification_service is not None,
            "ops_alert_service_ready": ops_alert_service is not None,
        },
        "routing_mode": "full_nexus" if (NEXUS_AVAILABLE and async_session_factory is not None) else "basic",
        "timestamp": datetime.utcnow().isoformat()
    }
