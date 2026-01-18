"""
================================================================================
NEXUS ORCHESTRATOR - Phase Handlers
================================================================================
Production-grade phase handlers for the 13-phase customer lifecycle.
Each handler implements circuit breakers, observability, and event publishing.
================================================================================
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from functools import wraps
import time

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from sqlalchemy.orm import selectinload

# Local imports (adjust path as needed)
from models.customer_lifecycle import (
    Customer, PhaseTransition, Payment, Entitlement, Notification, DunningSchedule,
    NexusPhase, CustomerStatus, SubscriptionTier, PaymentStatus, NotificationType
)

logger = logging.getLogger("nexus.phases")


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-handler circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: int = 30,
        half_open_max: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max = half_open_max
        
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN
        return self.half_open_calls < self.half_open_max
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max:
                self.state = CircuitState.CLOSED
                self.failures = 0
                logger.info(f"Circuit {self.name} CLOSED after successful recovery")
        self.failures = 0
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} OPEN after half-open failure")
        elif self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} OPEN after {self.failures} failures")


def with_circuit_breaker(breaker: CircuitBreaker):
    """Decorator to wrap handlers with circuit breaker logic."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise CircuitOpenError(f"Circuit {breaker.name} is OPEN")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        return wrapper
    return decorator


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# HANDLER RESULT
# =============================================================================

@dataclass
class HandlerResult:
    """Result from a phase handler execution."""
    success: bool
    action: str
    from_phase: Optional[NexusPhase]
    to_phase: NexusPhase
    from_status: Optional[CustomerStatus]
    to_status: CustomerStatus
    customer_id: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "action": self.action,
            "from_phase": self.from_phase.value if self.from_phase else None,
            "to_phase": self.to_phase.value,
            "from_status": self.from_status.value if self.from_status else None,
            "to_status": self.to_status.value,
            "customer_id": self.customer_id,
            "metadata": self.metadata,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# TIER ENTITLEMENTS CONFIGURATION
# =============================================================================

TIER_ENTITLEMENTS: Dict[SubscriptionTier, List[Dict[str, Any]]] = {
    SubscriptionTier.STARTER: [
        {"feature_key": "ragnarok_basic", "feature_name": "RAGNAROK Basic Access", "quota": 4},
        {"feature_key": "support_email", "feature_name": "Email Support", "quota": None},
    ],
    SubscriptionTier.GROWTH: [
        {"feature_key": "ragnarok_standard", "feature_name": "RAGNAROK Standard Access", "quota": 12},
        {"feature_key": "trinity_basic", "feature_name": "Trinity Basic Intelligence", "quota": 50},
        {"feature_key": "support_priority", "feature_name": "Priority Support", "quota": None},
    ],
    SubscriptionTier.SCALE: [
        {"feature_key": "ragnarok_unlimited", "feature_name": "RAGNAROK Unlimited Access", "quota": None},
        {"feature_key": "trinity_full", "feature_name": "Trinity Full Intelligence", "quota": None},
        {"feature_key": "voice_clone", "feature_name": "Voice Clone Access", "quota": 2},
        {"feature_key": "support_dedicated", "feature_name": "Dedicated Support", "quota": None},
    ],
    SubscriptionTier.ENTERPRISE: [
        {"feature_key": "ragnarok_unlimited", "feature_name": "RAGNAROK Unlimited Access", "quota": None},
        {"feature_key": "trinity_full", "feature_name": "Trinity Full Intelligence", "quota": None},
        {"feature_key": "voice_clone", "feature_name": "Voice Clone Access", "quota": None},
        {"feature_key": "avatar_clone", "feature_name": "Avatar Clone Access", "quota": None},
        {"feature_key": "white_label", "feature_name": "White Label Branding", "quota": None},
        {"feature_key": "support_enterprise", "feature_name": "Enterprise Support + SLA", "quota": None},
        {"feature_key": "api_access", "feature_name": "Full API Access", "quota": None},
    ],
    SubscriptionTier.NEXUS_PERSONAL: [
        {"feature_key": "nexus_assistant", "feature_name": "Personal AI Assistant", "quota": None},
        {"feature_key": "smart_home", "feature_name": "Smart Home Integration", "quota": None},
        {"feature_key": "family_profiles", "feature_name": "Family Profiles", "quota": 5},
        {"feature_key": "support_personal", "feature_name": "Personal Support Line", "quota": None},
    ],
}


# =============================================================================
# BASE PHASE HANDLER
# =============================================================================

class BasePhaseHandler:
    """Base class for all phase handlers with common functionality."""
    
    def __init__(self, db: AsyncSession, event_bus=None, notification_service=None):
        self.db = db
        self.event_bus = event_bus
        self.notification_service = notification_service
        self.circuit_breaker = CircuitBreaker(name=self.__class__.__name__)
    
    async def get_customer_by_stripe_id(self, stripe_customer_id: str) -> Optional[Customer]:
        """Fetch customer by Stripe ID."""
        result = await self.db.execute(
            select(Customer)
            .where(Customer.stripe_customer_id == stripe_customer_id)
            .options(selectinload(Customer.entitlements))
        )
        return result.scalar_one_or_none()
    
    async def get_customer_by_id(self, customer_id: str) -> Optional[Customer]:
        """Fetch customer by internal ID."""
        result = await self.db.execute(
            select(Customer)
            .where(Customer.id == customer_id)
            .options(selectinload(Customer.entitlements))
        )
        return result.scalar_one_or_none()
    
    async def record_transition(
        self,
        customer: Customer,
        from_phase: Optional[NexusPhase],
        to_phase: NexusPhase,
        from_status: Optional[CustomerStatus],
        to_status: CustomerStatus,
        trigger_event: str,
        trigger_event_id: Optional[str],
        action: str,
        success: bool,
        error: Optional[str] = None,
        duration_ms: int = 0,
        metadata: Dict[str, Any] = None
    ) -> PhaseTransition:
        """Record a phase transition in the audit log."""
        transition = PhaseTransition(
            customer_id=customer.id,
            from_phase=from_phase,
            to_phase=to_phase,
            from_status=from_status,
            to_status=to_status,
            trigger_event=trigger_event,
            trigger_event_id=trigger_event_id,
            handler_action=action,
            handler_success=success,
            handler_error=error,
            handler_duration_ms=duration_ms,
            customer_version=customer.version,
            custom_metadata=metadata or {}
        )
        self.db.add(transition)
        return transition
    
    async def publish_event(self, event_type: str, payload: Dict[str, Any]):
        """Publish event to event bus (RabbitMQ)."""
        if self.event_bus:
            await self.event_bus.publish(event_type, payload)
        logger.info(f"Event published: {event_type}")


# =============================================================================
# PHASE 1: ACQUISITION HANDLER
# =============================================================================

class Phase1AcquisitionHandler(BasePhaseHandler):
    """
    Phase 1: Lead Conversion & Acquisition
    Triggered by: checkout.session.completed, checkout.session.expired
    """
    
    async def convert_lead(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Convert a checkout session into a customer record.
        Called when checkout.session.completed fires.
        """
        start_time = time.time()
        
        stripe_customer_id = event_data.get("customer_id")
        email = event_data.get("customer_email")
        amount_total = event_data.get("amount_total", 0)
        subscription_id = event_data.get("subscription_id")
        metadata = event_data.get("metadata", {})
        
        try:
            # Check if customer already exists
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if customer:
                # Existing customer - upgrade path
                from_phase = customer.current_phase
                from_status = customer.status
                
                customer.current_phase = NexusPhase.PHASE_2_ONBOARDING
                customer.status = CustomerStatus.ACTIVE
                customer.subscription_id = subscription_id
                customer.subscription_started_at = datetime.utcnow()
                customer.version += 1
                customer.updated_at = datetime.utcnow()
                
                logger.info(f"Existing customer {customer.id} converted from {from_status} to ACTIVE")
            else:
                # New customer
                from_phase = None
                from_status = None
                
                customer = Customer(
                    stripe_customer_id=stripe_customer_id,
                    email=email,
                    name=metadata.get("customer_name"),
                    current_phase=NexusPhase.PHASE_2_ONBOARDING,
                    status=CustomerStatus.ACTIVE,
                    subscription_id=subscription_id,
                    subscription_started_at=datetime.utcnow(),
                    mrr=amount_total / 100 if amount_total else 0,
                    custom_metadata=metadata
                )
                self.db.add(customer)
                
                logger.info(f"New customer created: {email}")
            
            await self.db.flush()
            
            # Record payment
            if amount_total > 0:
                payment = Payment(
                    customer_id=customer.id,
                    stripe_subscription_id=subscription_id,
                    amount=amount_total,
                    status=PaymentStatus.SUCCEEDED,
                    description="Initial subscription payment"
                )
                self.db.add(payment)
                customer.last_payment_at = datetime.utcnow()
                customer.lifetime_value += amount_total / 100
            
            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_2_ONBOARDING,
                from_status=from_status,
                to_status=CustomerStatus.ACTIVE,
                trigger_event="checkout.session.completed",
                trigger_event_id=event_id,
                action="convert_lead",
                success=True,
                duration_ms=duration_ms,
                metadata={"amount": amount_total, "subscription_id": subscription_id}
            )
            
            await self.db.commit()
            
            # Publish event for downstream handlers
            await self.publish_event("nexus.customer.converted", {
                "customer_id": str(customer.id),
                "email": customer.email,
                "phase": "onboarding",
                "subscription_id": subscription_id
            })
            
            # Trigger welcome notification
            if self.notification_service:
                await self.notification_service.send_welcome(customer)
            
            return HandlerResult(
                success=True,
                action="convert_lead",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_2_ONBOARDING,
                from_status=from_status,
                to_status=CustomerStatus.ACTIVE,
                customer_id=str(customer.id),
                metadata={"amount": amount_total, "subscription_id": subscription_id},
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to convert lead: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="convert_lead",
                from_phase=None,
                to_phase=NexusPhase.PHASE_1_ACQUISITION,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
    
    async def trigger_recovery(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle abandoned checkout - trigger recovery flow.
        Called when checkout.session.expired fires.
        """
        start_time = time.time()
        
        email = event_data.get("customer_email")
        session_id = event_data.get("session_id")
        amount_total = event_data.get("amount_total", 0)
        
        logger.info(f"Checkout abandoned: {email}, session {session_id}")
        
        # Trigger recovery email sequence
        if self.notification_service:
            await self.notification_service.send_abandonment_recovery(
                email=email,
                session_id=session_id,
                amount=amount_total
            )
        
        # Publish event for analytics
        await self.publish_event("nexus.checkout.abandoned", {
            "email": email,
            "session_id": session_id,
            "amount": amount_total
        })
        
        return HandlerResult(
            success=True,
            action="trigger_recovery",
            from_phase=None,
            to_phase=NexusPhase.PHASE_1_ACQUISITION,
            from_status=None,
            to_status=CustomerStatus.LEAD,
            customer_id=email,
            metadata={"session_id": session_id, "amount": amount_total},
            duration_ms=int((time.time() - start_time) * 1000)
        )


# =============================================================================
# PHASE 2: ONBOARDING HANDLER
# =============================================================================

class Phase2OnboardingHandler(BasePhaseHandler):
    """
    Phase 2: Customer Onboarding
    Triggered by: customer.created, customer.updated
    """
    
    async def create_customer_record(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Create or update customer record from Stripe customer event.
        """
        start_time = time.time()
        
        stripe_customer_id = event_data.get("customer_id")
        email = event_data.get("email")
        name = event_data.get("name")
        metadata = event_data.get("metadata", {})
        
        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if customer:
                # Update existing
                from_phase = customer.current_phase
                from_status = customer.status
                
                if name:
                    customer.name = name
                if metadata:
                    customer.custom_metadata = {**customer.custom_metadata, **metadata}
                customer.version += 1
                customer.updated_at = datetime.utcnow()
                
                logger.info(f"Customer record updated: {customer.id}")
            else:
                # Create new (lead from Stripe)
                from_phase = None
                from_status = None
                
                customer = Customer(
                    stripe_customer_id=stripe_customer_id,
                    email=email,
                    name=name,
                    current_phase=NexusPhase.PHASE_2_ONBOARDING,
                    status=CustomerStatus.LEAD,
                    custom_metadata=metadata
                )
                self.db.add(customer)
                
                logger.info(f"Customer record created: {email}")
            
            await self.db.flush()
            
            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_2_ONBOARDING,
                from_status=from_status,
                to_status=customer.status,
                trigger_event="customer.created",
                trigger_event_id=event_id,
                action="create_customer_record",
                success=True,
                duration_ms=duration_ms
            )
            
            await self.db.commit()
            
            return HandlerResult(
                success=True,
                action="create_customer_record",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_2_ONBOARDING,
                from_status=from_status,
                to_status=customer.status,
                customer_id=str(customer.id),
                metadata={"email": email, "name": name},
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to create customer record: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="create_customer_record",
                from_phase=None,
                to_phase=NexusPhase.PHASE_2_ONBOARDING,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000)
            )
    
    async def sync_customer_data(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Sync customer data from Stripe update event.
        """
        start_time = time.time()
        
        stripe_customer_id = event_data.get("customer_id")
        email = event_data.get("email")
        name = event_data.get("name")
        
        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if not customer:
                logger.warning(f"Customer not found for sync: {stripe_customer_id}")
                return HandlerResult(
                    success=False,
                    action="sync_customer_data",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_2_ONBOARDING,
                    from_status=None,
                    to_status=CustomerStatus.LEAD,
                    customer_id=stripe_customer_id,
                    metadata=event_data,
                    error="Customer not found"
                )
            
            from_phase = customer.current_phase
            from_status = customer.status
            
            # Update fields
            if email and email != customer.email:
                customer.email = email
            if name and name != customer.name:
                customer.name = name
            
            customer.version += 1
            customer.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            return HandlerResult(
                success=True,
                action="sync_customer_data",
                from_phase=from_phase,
                to_phase=from_phase,
                from_status=from_status,
                to_status=from_status,
                customer_id=str(customer.id),
                metadata={"email": email, "name": name},
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
        except Exception as e:
            logger.error(f"Failed to sync customer data: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="sync_customer_data",
                from_phase=None,
                to_phase=NexusPhase.PHASE_2_ONBOARDING,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e)
            )


# =============================================================================
# PHASE 3: PROVISIONING HANDLER
# =============================================================================

class Phase3ProvisioningHandler(BasePhaseHandler):
    """
    Phase 3: Service Provisioning
    Triggered by: customer.subscription.created
    """
    
    async def provision_services(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Provision services and entitlements for new subscription.
        """
        start_time = time.time()
        
        subscription_id = event_data.get("subscription_id")
        stripe_customer_id = event_data.get("customer_id")
        status = event_data.get("status")
        items = event_data.get("items", [])
        
        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if not customer:
                logger.error(f"Customer not found for provisioning: {stripe_customer_id}")
                return HandlerResult(
                    success=False,
                    action="provision_services",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_3_PROVISIONING,
                    from_status=None,
                    to_status=CustomerStatus.LEAD,
                    customer_id=stripe_customer_id,
                    metadata=event_data,
                    error="Customer not found"
                )
            
            from_phase = customer.current_phase
            from_status = customer.status
            
            # Determine tier from subscription items
            tier = self._determine_tier_from_items(items)
            
            # Update customer
            customer.subscription_id = subscription_id
            customer.tier = tier
            customer.current_phase = NexusPhase.PHASE_4_ACTIVE
            customer.status = CustomerStatus.ACTIVE
            customer.subscription_started_at = datetime.utcnow()
            customer.version += 1
            
            # Calculate next billing date (30 days from now)
            customer.next_billing_at = datetime.utcnow() + timedelta(days=30)
            
            await self.db.flush()
            
            # Provision entitlements
            entitlements_granted = await self._grant_tier_entitlements(customer, tier)
            
            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=from_status,
                to_status=CustomerStatus.ACTIVE,
                trigger_event="customer.subscription.created",
                trigger_event_id=event_id,
                action="provision_services",
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "subscription_id": subscription_id,
                    "tier": tier.value if tier else None,
                    "entitlements": entitlements_granted
                }
            )
            
            await self.db.commit()
            
            # Publish event
            await self.publish_event("nexus.services.provisioned", {
                "customer_id": str(customer.id),
                "subscription_id": subscription_id,
                "tier": tier.value if tier else None,
                "entitlements": entitlements_granted
            })
            
            # Send provisioning confirmation
            if self.notification_service:
                await self.notification_service.send_provisioning_complete(
                    customer,
                    entitlements_granted
                )
            
            logger.info(f"Services provisioned for {customer.email}: tier={tier}, entitlements={len(entitlements_granted)}")
            
            return HandlerResult(
                success=True,
                action="provision_services",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=from_status,
                to_status=CustomerStatus.ACTIVE,
                customer_id=str(customer.id),
                metadata={
                    "subscription_id": subscription_id,
                    "tier": tier.value if tier else None,
                    "entitlements": entitlements_granted
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to provision services: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="provision_services",
                from_phase=None,
                to_phase=NexusPhase.PHASE_3_PROVISIONING,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e)
            )
    
    def _determine_tier_from_items(self, items: List[Dict]) -> Optional[SubscriptionTier]:
        """Determine subscription tier from line items."""
        # Map price IDs to tiers (configure based on your Stripe products)
        price_tier_map = {
            "price_starter": SubscriptionTier.STARTER,
            "price_growth": SubscriptionTier.GROWTH,
            "price_scale": SubscriptionTier.SCALE,
            "price_enterprise": SubscriptionTier.ENTERPRISE,
            "price_nexus": SubscriptionTier.NEXUS_PERSONAL,
        }
        
        for item in items:
            price_id = item.get("price_id", "")
            for key, tier in price_tier_map.items():
                if key in price_id.lower():
                    return tier
        
        # Default to starter
        return SubscriptionTier.STARTER
    
    async def _grant_tier_entitlements(
        self,
        customer: Customer,
        tier: SubscriptionTier
    ) -> List[str]:
        """Grant entitlements based on subscription tier."""
        entitlements_granted = []
        
        if tier not in TIER_ENTITLEMENTS:
            return entitlements_granted
        
        for ent_config in TIER_ENTITLEMENTS[tier]:
            # Check if already exists
            existing = next(
                (e for e in customer.entitlements if e.feature_key == ent_config["feature_key"]),
                None
            )
            
            if existing:
                # Reactivate if revoked
                existing.is_active = True
                existing.revoked_at = None
                existing.quota = ent_config.get("quota")
                existing.used = 0
            else:
                # Create new entitlement
                entitlement = Entitlement(
                    customer_id=customer.id,
                    feature_key=ent_config["feature_key"],
                    feature_name=ent_config["feature_name"],
                    quota=ent_config.get("quota"),
                    source_subscription_id=customer.subscription_id
                )
                self.db.add(entitlement)
            
            entitlements_granted.append(ent_config["feature_key"])
        
        return entitlements_granted


# =============================================================================
# PHASE 4: ACTIVE CUSTOMER HANDLER
# =============================================================================

class Phase4ActiveHandler(BasePhaseHandler):
    """
    Phase 4: Active Customer Management
    Triggered by: invoice.payment_succeeded, invoice.paid
    """
    
    async def record_payment(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Record successful payment and update customer state.
        """
        start_time = time.time()
        
        stripe_customer_id = event_data.get("customer_id")
        invoice_id = event_data.get("invoice_id")
        subscription_id = event_data.get("subscription_id")
        amount_paid = event_data.get("amount_paid", 0)
        
        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if not customer:
                logger.warning(f"Customer not found for payment: {stripe_customer_id}")
                return HandlerResult(
                    success=False,
                    action="record_payment",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_4_ACTIVE,
                    from_status=None,
                    to_status=CustomerStatus.ACTIVE,
                    customer_id=stripe_customer_id,
                    metadata=event_data,
                    error="Customer not found"
                )
            
            from_phase = customer.current_phase
            from_status = customer.status
            
            # Record payment
            payment = Payment(
                customer_id=customer.id,
                stripe_invoice_id=invoice_id,
                stripe_subscription_id=subscription_id,
                amount=amount_paid,
                status=PaymentStatus.SUCCEEDED
            )
            self.db.add(payment)
            
            # Update customer
            customer.last_payment_at = datetime.utcnow()
            customer.lifetime_value += amount_paid / 100
            customer.next_billing_at = datetime.utcnow() + timedelta(days=30)
            
            # Clear any dunning state
            if customer.status == CustomerStatus.PAST_DUE:
                customer.status = CustomerStatus.ACTIVE
                customer.dunning_started_at = None
                customer.dunning_attempts = 0
                customer.last_dunning_at = None
                customer.current_phase = NexusPhase.PHASE_4_ACTIVE
            
            customer.version += 1
            
            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=from_status,
                to_status=customer.status,
                trigger_event="invoice.payment_succeeded",
                trigger_event_id=event_id,
                action="record_payment",
                success=True,
                duration_ms=duration_ms,
                metadata={"amount": amount_paid, "invoice_id": invoice_id}
            )
            
            await self.db.commit()
            
            # Send receipt
            if self.notification_service:
                await self.notification_service.send_payment_receipt(customer, amount_paid)
            
            logger.info(f"Payment recorded: {customer.email}, ${amount_paid/100:.2f}")
            
            return HandlerResult(
                success=True,
                action="record_payment",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=from_status,
                to_status=customer.status,
                customer_id=str(customer.id),
                metadata={"amount": amount_paid, "invoice_id": invoice_id},
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to record payment: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="record_payment",
                from_phase=None,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e)
            )

    async def track_invoice(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle invoice.created / invoice.finalized - Track invoice.
        No phase change, just records the invoice for auditing.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        invoice_id = event_data.get("invoice_id") or event_data.get("id")
        amount_due = event_data.get("amount_due", 0)

        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)

            if not customer:
                # Customer may not exist yet - that's OK for invoice tracking
                logger.info(f"Invoice tracked (no customer yet): {invoice_id}")
                return HandlerResult(
                    success=True,
                    action="track_invoice",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_4_ACTIVE,
                    from_status=None,
                    to_status=CustomerStatus.ACTIVE,
                    customer_id=stripe_customer_id or "unknown",
                    metadata={"invoice_id": invoice_id, "amount_due": amount_due},
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            # Record invoice as pending payment
            payment = Payment(
                customer_id=customer.id,
                stripe_invoice_id=invoice_id,
                amount=amount_due,
                status=PaymentStatus.PENDING,
                description="Invoice created"
            )
            self.db.add(payment)
            await self.db.commit()

            logger.info(f"Invoice tracked: {invoice_id} for {customer.email}, ${amount_due/100:.2f}")

            return HandlerResult(
                success=True,
                action="track_invoice",
                from_phase=customer.current_phase,
                to_phase=customer.current_phase,  # No phase change
                from_status=customer.status,
                to_status=customer.status,
                customer_id=str(customer.id),
                metadata={"invoice_id": invoice_id, "amount_due": amount_due},
                duration_ms=int((time.time() - start_time) * 1000)
            )

        except Exception as e:
            logger.error(f"Failed to track invoice: {e}")
            return HandlerResult(
                success=False,
                action="track_invoice",
                from_phase=None,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=stripe_customer_id or "unknown",
                metadata=event_data,
                error=str(e)
            )

    async def track_payment_intent(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle payment_intent.created - Track new payment intent.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        intent_id = event_data.get("payment_intent_id") or event_data.get("id")
        amount = event_data.get("amount", 0)

        logger.info(f"Payment intent tracked: {intent_id}, ${amount/100:.2f}")

        return HandlerResult(
            success=True,
            action="track_payment_intent",
            from_phase=None,
            to_phase=NexusPhase.PHASE_4_ACTIVE,
            from_status=None,
            to_status=CustomerStatus.ACTIVE,
            customer_id=stripe_customer_id or "unknown",
            metadata={"intent_id": intent_id, "amount": amount},
            duration_ms=int((time.time() - start_time) * 1000)
        )

    async def handle_payment_intent_succeeded(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle payment_intent.succeeded - Record successful payment.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        intent_id = event_data.get("payment_intent_id") or event_data.get("id")
        amount = event_data.get("amount", 0)

        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)

            if customer:
                payment = Payment(
                    customer_id=customer.id,
                    stripe_payment_intent_id=intent_id,
                    amount=amount,
                    status=PaymentStatus.SUCCEEDED
                )
                self.db.add(payment)
                customer.last_payment_at = datetime.utcnow()
                customer.lifetime_value += amount / 100
                await self.db.commit()

            logger.info(f"Payment intent succeeded: {intent_id}, ${amount/100:.2f}")

            return HandlerResult(
                success=True,
                action="handle_payment_intent_succeeded",
                from_phase=customer.current_phase if customer else None,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=customer.status if customer else None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=str(customer.id) if customer else stripe_customer_id,
                metadata={"intent_id": intent_id, "amount": amount},
                duration_ms=int((time.time() - start_time) * 1000)
            )

        except Exception as e:
            logger.error(f"Failed to handle payment intent: {e}")
            return HandlerResult(
                success=False,
                action="handle_payment_intent_succeeded",
                from_phase=None,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=stripe_customer_id or "unknown",
                metadata=event_data,
                error=str(e)
            )

    async def handle_payment_intent_failed(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle payment_intent.payment_failed - Log failure, may trigger dunning.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        intent_id = event_data.get("payment_intent_id") or event_data.get("id")
        error = event_data.get("last_payment_error", {})

        logger.warning(f"Payment intent failed: {intent_id} - {error.get('message', 'Unknown error')}")

        return HandlerResult(
            success=True,  # We successfully handled the failure event
            action="handle_payment_intent_failed",
            from_phase=None,
            to_phase=NexusPhase.PHASE_4_ACTIVE,  # Don't move to dunning for single intent failure
            from_status=None,
            to_status=CustomerStatus.ACTIVE,
            customer_id=stripe_customer_id or "unknown",
            metadata={"intent_id": intent_id, "error_code": error.get("code")},
            duration_ms=int((time.time() - start_time) * 1000)
        )

    async def handle_charge_succeeded(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle charge.succeeded - Record revenue.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        charge_id = event_data.get("charge_id") or event_data.get("id")
        amount = event_data.get("amount", 0)

        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)

            if customer:
                payment = Payment(
                    customer_id=customer.id,
                    stripe_charge_id=charge_id,
                    amount=amount,
                    status=PaymentStatus.SUCCEEDED,
                    description="Charge succeeded"
                )
                self.db.add(payment)
                customer.last_payment_at = datetime.utcnow()
                await self.db.commit()

            logger.info(f"Charge succeeded: {charge_id}, ${amount/100:.2f}")

            return HandlerResult(
                success=True,
                action="handle_charge_succeeded",
                from_phase=customer.current_phase if customer else None,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=customer.status if customer else None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=str(customer.id) if customer else stripe_customer_id,
                metadata={"charge_id": charge_id, "amount": amount},
                duration_ms=int((time.time() - start_time) * 1000)
            )

        except Exception as e:
            logger.error(f"Failed to handle charge: {e}")
            return HandlerResult(
                success=False,
                action="handle_charge_succeeded",
                from_phase=None,
                to_phase=NexusPhase.PHASE_4_ACTIVE,
                from_status=None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=stripe_customer_id or "unknown",
                metadata=event_data,
                error=str(e)
            )

    async def handle_charge_failed(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle charge.failed - Log failure.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        charge_id = event_data.get("charge_id") or event_data.get("id")
        failure_message = event_data.get("failure_message", "Unknown")

        logger.warning(f"Charge failed: {charge_id} - {failure_message}")

        return HandlerResult(
            success=True,
            action="handle_charge_failed",
            from_phase=None,
            to_phase=NexusPhase.PHASE_4_ACTIVE,
            from_status=None,
            to_status=CustomerStatus.ACTIVE,
            customer_id=stripe_customer_id or "unknown",
            metadata={"charge_id": charge_id, "failure_message": failure_message},
            duration_ms=int((time.time() - start_time) * 1000)
        )

    async def handle_payment_intent_canceled(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle payment_intent.canceled - Log cancellation.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        intent_id = event_data.get("payment_intent_id") or event_data.get("id")
        cancellation_reason = event_data.get("cancellation_reason", "unknown")

        logger.info(f"Payment intent canceled: {intent_id} - reason: {cancellation_reason}")

        return HandlerResult(
            success=True,
            action="handle_payment_intent_canceled",
            from_phase=None,
            to_phase=NexusPhase.PHASE_4_ACTIVE,
            from_status=None,
            to_status=CustomerStatus.ACTIVE,
            customer_id=stripe_customer_id or "unknown",
            metadata={"intent_id": intent_id, "cancellation_reason": cancellation_reason},
            duration_ms=int((time.time() - start_time) * 1000)
        )


# =============================================================================
# PHASE 5: EXPANSION HANDLER
# =============================================================================

class Phase5ExpansionHandler(BasePhaseHandler):
    """
    Phase 5: Customer Expansion & Upgrades
    Triggered by: customer.subscription.updated
    """

    async def update_entitlements(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Handle subscription updates - upgrade/downgrade entitlements.
        """
        start_time = time.time()

        stripe_customer_id = event_data.get("customer_id")
        subscription_id = event_data.get("subscription_id")
        previous_attributes = event_data.get("previous_attributes", {})
        items = event_data.get("items", [])

        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)

            if not customer:
                logger.warning(f"Customer not found for entitlement update: {stripe_customer_id}")
                return HandlerResult(
                    success=False,
                    action="update_entitlements",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_5_EXPANSION,
                    from_status=None,
                    to_status=CustomerStatus.ACTIVE,
                    customer_id=stripe_customer_id or "unknown",
                    metadata=event_data,
                    error="Customer not found"
                )

            from_phase = customer.current_phase
            from_status = customer.status

            # Determine new tier from subscription items
            new_tier = self._determine_tier_from_items(items)
            old_tier = customer.tier

            # Update customer tier if changed
            tier_changed = new_tier and new_tier != old_tier
            if tier_changed:
                customer.tier = new_tier
                customer.version += 1
                customer.updated_at = datetime.utcnow()

                # Update entitlements based on new tier
                await self._update_tier_entitlements(customer, old_tier, new_tier)

                logger.info(f"Customer {customer.email} tier changed: {old_tier} -> {new_tier}")

            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_5_EXPANSION if tier_changed else from_phase,
                from_status=from_status,
                to_status=customer.status,
                trigger_event="customer.subscription.updated",
                trigger_event_id=event_id,
                action="update_entitlements",
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "subscription_id": subscription_id,
                    "old_tier": old_tier.value if old_tier else None,
                    "new_tier": new_tier.value if new_tier else None,
                    "tier_changed": tier_changed
                }
            )

            await self.db.commit()

            # Publish event if tier changed
            if tier_changed:
                await self.publish_event("nexus.customer.tier_changed", {
                    "customer_id": str(customer.id),
                    "email": customer.email,
                    "old_tier": old_tier.value if old_tier else None,
                    "new_tier": new_tier.value if new_tier else None
                })

            return HandlerResult(
                success=True,
                action="update_entitlements",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_5_EXPANSION if tier_changed else from_phase,
                from_status=from_status,
                to_status=customer.status,
                customer_id=str(customer.id),
                metadata={
                    "subscription_id": subscription_id,
                    "tier_changed": tier_changed,
                    "new_tier": new_tier.value if new_tier else None
                },
                duration_ms=duration_ms
            )

        except Exception as e:
            logger.error(f"Failed to update entitlements: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="update_entitlements",
                from_phase=None,
                to_phase=NexusPhase.PHASE_5_EXPANSION,
                from_status=None,
                to_status=CustomerStatus.ACTIVE,
                customer_id=stripe_customer_id or "unknown",
                metadata=event_data,
                error=str(e)
            )

    def _determine_tier_from_items(self, items: List[Dict]) -> Optional[SubscriptionTier]:
        """Determine subscription tier from line items."""
        price_tier_map = {
            "price_starter": SubscriptionTier.STARTER,
            "price_growth": SubscriptionTier.GROWTH,
            "price_scale": SubscriptionTier.SCALE,
            "price_enterprise": SubscriptionTier.ENTERPRISE,
            "price_nexus": SubscriptionTier.NEXUS_PERSONAL,
        }

        for item in items:
            price_id = item.get("price_id", "")
            for key, tier in price_tier_map.items():
                if key in price_id.lower():
                    return tier

        return None

    async def _update_tier_entitlements(
        self,
        customer: Customer,
        old_tier: Optional[SubscriptionTier],
        new_tier: SubscriptionTier
    ):
        """Update entitlements when tier changes."""
        # Revoke old tier entitlements not in new tier
        new_tier_features = {e["feature_key"] for e in TIER_ENTITLEMENTS.get(new_tier, [])}

        for entitlement in customer.entitlements:
            if entitlement.is_active and entitlement.feature_key not in new_tier_features:
                entitlement.is_active = False
                entitlement.revoked_at = datetime.utcnow()

        # Grant new tier entitlements
        for ent_config in TIER_ENTITLEMENTS.get(new_tier, []):
            existing = next(
                (e for e in customer.entitlements if e.feature_key == ent_config["feature_key"]),
                None
            )

            if existing:
                existing.is_active = True
                existing.revoked_at = None
                existing.quota = ent_config.get("quota")
                existing.used = 0
            else:
                entitlement = Entitlement(
                    customer_id=customer.id,
                    feature_key=ent_config["feature_key"],
                    feature_name=ent_config["feature_name"],
                    quota=ent_config.get("quota"),
                    source_subscription_id=customer.subscription_id
                )
                self.db.add(entitlement)


# =============================================================================
# PHASE 10: DUNNING HANDLER
# =============================================================================

class Phase10DunningHandler(BasePhaseHandler):
    """
    Phase 10: Payment Recovery (Dunning)
    Triggered by: invoice.payment_failed
    """
    
    DUNNING_SCHEDULE = [
        {"days": 0, "notification": NotificationType.DUNNING_1, "retry": True},
        {"days": 3, "notification": NotificationType.DUNNING_2, "retry": True},
        {"days": 7, "notification": NotificationType.DUNNING_FINAL, "retry": True},
        {"days": 14, "notification": None, "retry": False, "suspend": True},
    ]
    
    async def initiate_dunning(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Start dunning sequence for failed payment.
        """
        start_time = time.time()
        
        stripe_customer_id = event_data.get("customer_id")
        invoice_id = event_data.get("invoice_id")
        amount_due = event_data.get("amount_due", 0)
        
        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if not customer:
                logger.warning(f"Customer not found for dunning: {stripe_customer_id}")
                return HandlerResult(
                    success=False,
                    action="initiate_dunning",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_10_DUNNING,
                    from_status=None,
                    to_status=CustomerStatus.PAST_DUE,
                    customer_id=stripe_customer_id,
                    metadata=event_data,
                    error="Customer not found"
                )
            
            from_phase = customer.current_phase
            from_status = customer.status
            
            # Record failed payment
            payment = Payment(
                customer_id=customer.id,
                stripe_invoice_id=invoice_id,
                amount=amount_due,
                status=PaymentStatus.FAILED,
                failure_message="Payment declined"
            )
            self.db.add(payment)
            
            # Update customer to dunning state
            if not customer.dunning_started_at:
                customer.dunning_started_at = datetime.utcnow()
            
            customer.dunning_attempts += 1
            customer.last_dunning_at = datetime.utcnow()
            customer.status = CustomerStatus.PAST_DUE
            customer.current_phase = NexusPhase.PHASE_10_DUNNING
            customer.version += 1
            
            # Create dunning schedule
            await self._create_dunning_schedule(customer, invoice_id, amount_due)
            
            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_10_DUNNING,
                from_status=from_status,
                to_status=CustomerStatus.PAST_DUE,
                trigger_event="invoice.payment_failed",
                trigger_event_id=event_id,
                action="initiate_dunning",
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "invoice_id": invoice_id,
                    "amount_due": amount_due,
                    "attempt": customer.dunning_attempts
                }
            )
            
            await self.db.commit()
            
            # Send first dunning notification
            if self.notification_service:
                await self.notification_service.send_dunning_notification(
                    customer,
                    NotificationType.DUNNING_1,
                    amount_due
                )
            
            logger.warning(f"Dunning initiated: {customer.email}, attempt {customer.dunning_attempts}")
            
            return HandlerResult(
                success=True,
                action="initiate_dunning",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_10_DUNNING,
                from_status=from_status,
                to_status=CustomerStatus.PAST_DUE,
                customer_id=str(customer.id),
                metadata={
                    "invoice_id": invoice_id,
                    "amount_due": amount_due,
                    "attempt": customer.dunning_attempts
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to initiate dunning: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="initiate_dunning",
                from_phase=None,
                to_phase=NexusPhase.PHASE_10_DUNNING,
                from_status=None,
                to_status=CustomerStatus.PAST_DUE,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e)
            )
    
    async def _create_dunning_schedule(
        self,
        customer: Customer,
        invoice_id: str,
        amount_due: int
    ):
        """Create dunning retry schedule."""
        for i, step in enumerate(self.DUNNING_SCHEDULE):
            schedule = DunningSchedule(
                customer_id=customer.id,
                stripe_invoice_id=invoice_id,
                amount_due=amount_due,
                attempt_number=i + 1,
                max_attempts=len(self.DUNNING_SCHEDULE),
                scheduled_at=datetime.utcnow() + timedelta(days=step["days"])
            )
            self.db.add(schedule)


# =============================================================================
# PHASE 13: CHURN HANDLER
# =============================================================================

class Phase13ChurnHandler(BasePhaseHandler):
    """
    Phase 13: Customer Churn & Offboarding
    Triggered by: customer.subscription.deleted
    """
    
    async def deprovision_services(
        self,
        event_data: Dict[str, Any],
        event_id: str
    ) -> HandlerResult:
        """
        Deprovision services and archive customer.
        """
        start_time = time.time()
        
        subscription_id = event_data.get("subscription_id")
        stripe_customer_id = event_data.get("customer_id")
        
        try:
            customer = await self.get_customer_by_stripe_id(stripe_customer_id)
            
            if not customer:
                logger.warning(f"Customer not found for deprovisioning: {stripe_customer_id}")
                return HandlerResult(
                    success=False,
                    action="deprovision_services",
                    from_phase=None,
                    to_phase=NexusPhase.PHASE_13_CHURN,
                    from_status=None,
                    to_status=CustomerStatus.CHURNED,
                    customer_id=stripe_customer_id,
                    metadata=event_data,
                    error="Customer not found"
                )
            
            from_phase = customer.current_phase
            from_status = customer.status
            
            # Revoke all entitlements
            entitlements_revoked = []
            for entitlement in customer.entitlements:
                if entitlement.is_active:
                    entitlement.is_active = False
                    entitlement.revoked_at = datetime.utcnow()
                    entitlements_revoked.append(entitlement.feature_key)
            
            # Update customer
            customer.status = CustomerStatus.CHURNED
            customer.current_phase = NexusPhase.PHASE_13_CHURN
            customer.subscription_ends_at = datetime.utcnow()
            customer.version += 1
            
            # Record transition
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_transition(
                customer=customer,
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_13_CHURN,
                from_status=from_status,
                to_status=CustomerStatus.CHURNED,
                trigger_event="customer.subscription.deleted",
                trigger_event_id=event_id,
                action="deprovision_services",
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "subscription_id": subscription_id,
                    "entitlements_revoked": entitlements_revoked
                }
            )
            
            await self.db.commit()
            
            # Publish event
            await self.publish_event("nexus.customer.churned", {
                "customer_id": str(customer.id),
                "email": customer.email,
                "subscription_id": subscription_id,
                "entitlements_revoked": entitlements_revoked,
                "lifetime_value": customer.lifetime_value
            })
            
            # Send offboarding and win-back sequence
            if self.notification_service:
                await self.notification_service.send_offboarding(customer)
            
            logger.info(f"Customer churned: {customer.email}, LTV=${customer.lifetime_value:.2f}")
            
            return HandlerResult(
                success=True,
                action="deprovision_services",
                from_phase=from_phase,
                to_phase=NexusPhase.PHASE_13_CHURN,
                from_status=from_status,
                to_status=CustomerStatus.CHURNED,
                customer_id=str(customer.id),
                metadata={
                    "subscription_id": subscription_id,
                    "entitlements_revoked": entitlements_revoked,
                    "lifetime_value": customer.lifetime_value
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Failed to deprovision services: {e}")
            await self.db.rollback()
            return HandlerResult(
                success=False,
                action="deprovision_services",
                from_phase=None,
                to_phase=NexusPhase.PHASE_13_CHURN,
                from_status=None,
                to_status=CustomerStatus.CHURNED,
                customer_id=stripe_customer_id,
                metadata=event_data,
                error=str(e)
            )


# =============================================================================
# NEXUS ORCHESTRATOR - MAIN DISPATCHER
# =============================================================================

class NexusOrchestrator:
    """
    Central orchestrator that routes Stripe events to appropriate phase handlers.
    """
    
    EVENT_HANDLER_MAP = {
        # Phase 1: Acquisition
        "checkout.session.completed": ("phase_1", "convert_lead"),
        "checkout.session.expired": ("phase_1", "trigger_recovery"),

        # Phase 2: Onboarding
        "customer.created": ("phase_2", "create_customer_record"),
        "customer.updated": ("phase_2", "sync_customer_data"),

        # Phase 3: Provisioning
        "customer.subscription.created": ("phase_3", "provision_services"),

        # Phase 4: Active - Invoice lifecycle
        "invoice.created": ("phase_4", "track_invoice"),
        "invoice.finalized": ("phase_4", "track_invoice"),
        "invoice.paid": ("phase_4", "record_payment"),
        "invoice.payment_succeeded": ("phase_4", "record_payment"),

        # Phase 4: Active - Payment Intent lifecycle
        "payment_intent.created": ("phase_4", "track_payment_intent"),
        "payment_intent.succeeded": ("phase_4", "handle_payment_intent_succeeded"),
        "payment_intent.payment_failed": ("phase_4", "handle_payment_intent_failed"),
        "payment_intent.canceled": ("phase_4", "handle_payment_intent_canceled"),

        # Phase 4: Active - Charge lifecycle
        "charge.succeeded": ("phase_4", "handle_charge_succeeded"),
        "charge.failed": ("phase_4", "handle_charge_failed"),

        # Phase 5: Expansion (subscription changes)
        "customer.subscription.updated": ("phase_5", "update_entitlements"),

        # Phase 10: Dunning
        "invoice.payment_failed": ("phase_10", "initiate_dunning"),

        # Phase 13: Churn
        "customer.subscription.deleted": ("phase_13", "deprovision_services"),
    }
    
    def __init__(
        self,
        db: AsyncSession,
        event_bus=None,
        notification_service=None
    ):
        self.db = db
        self.event_bus = event_bus
        self.notification_service = notification_service
        
        # Initialize handlers
        self.handlers = {
            "phase_1": Phase1AcquisitionHandler(db, event_bus, notification_service),
            "phase_2": Phase2OnboardingHandler(db, event_bus, notification_service),
            "phase_3": Phase3ProvisioningHandler(db, event_bus, notification_service),
            "phase_4": Phase4ActiveHandler(db, event_bus, notification_service),
            "phase_5": Phase5ExpansionHandler(db, event_bus, notification_service),
            "phase_10": Phase10DunningHandler(db, event_bus, notification_service),
            "phase_13": Phase13ChurnHandler(db, event_bus, notification_service),
        }
    
    async def route_event(
        self,
        event_type: str,
        event_id: str,
        event_data: Dict[str, Any]
    ) -> HandlerResult:
        """
        Route a Stripe event to the appropriate phase handler.
        """
        mapping = self.EVENT_HANDLER_MAP.get(event_type)
        
        if not mapping:
            logger.warning(f"No handler mapped for event: {event_type}")
            return HandlerResult(
                success=False,
                action="unmapped",
                from_phase=None,
                to_phase=NexusPhase.PHASE_1_ACQUISITION,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id="unknown",
                metadata={"event_type": event_type},
                error=f"No handler for event type: {event_type}"
            )
        
        phase_key, action = mapping
        handler = self.handlers.get(phase_key)
        
        if not handler:
            logger.error(f"Handler not found: {phase_key}")
            return HandlerResult(
                success=False,
                action=action,
                from_phase=None,
                to_phase=NexusPhase.PHASE_1_ACQUISITION,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id="unknown",
                metadata={"event_type": event_type, "phase": phase_key},
                error=f"Handler not initialized: {phase_key}"
            )
        
        # Get the action method
        handler_method = getattr(handler, action, None)
        if not handler_method:
            logger.error(f"Action method not found: {action}")
            return HandlerResult(
                success=False,
                action=action,
                from_phase=None,
                to_phase=NexusPhase.PHASE_1_ACQUISITION,
                from_status=None,
                to_status=CustomerStatus.LEAD,
                customer_id="unknown",
                metadata={"event_type": event_type, "action": action},
                error=f"Action method not found: {action}"
            )
        
        logger.info(f"Routing {event_type} to {phase_key}.{action}")
        
        # Execute handler
        result = await handler_method(event_data, event_id)
        
        logger.info(
            f"Handler result: success={result.success}, "
            f"phase={result.to_phase.value}, "
            f"duration={result.duration_ms}ms"
        )
        
        return result
