"""
================================================================================
NEXUS ORCHESTRATOR - Customer Lifecycle Models
================================================================================
SQLAlchemy models for tracking customer state through all 13 Nexus phases.
Supports MVCC-style versioning for audit trails and rollback capabilities.
================================================================================
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, JSON, Text,
    ForeignKey, Index, Enum as SQLEnum, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

Base = declarative_base()


# =============================================================================
# ENUMS
# =============================================================================

class NexusPhase(str, Enum):
    """All 13 phases of the Nexus customer lifecycle."""
    PHASE_1_ACQUISITION = "acquisition"
    PHASE_2_ONBOARDING = "onboarding"
    PHASE_3_PROVISIONING = "provisioning"
    PHASE_4_ACTIVE = "active"
    PHASE_5_EXPANSION = "expansion"
    PHASE_6_ENGAGEMENT = "engagement"
    PHASE_7_RETENTION = "retention"
    PHASE_8_BILLING = "billing"
    PHASE_9_RENEWAL = "renewal"
    PHASE_10_DUNNING = "dunning"
    PHASE_11_SUSPENSION = "suspension"
    PHASE_12_CANCELLATION = "cancellation"
    PHASE_13_CHURN = "churn"


class CustomerStatus(str, Enum):
    """Customer account status."""
    LEAD = "lead"
    TRIAL = "trial"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    CHURNED = "churned"


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""
    STARTER = "starter"
    GROWTH = "growth"
    SCALE = "scale"
    ENTERPRISE = "enterprise"
    NEXUS_PERSONAL = "nexus_personal"


class PaymentStatus(str, Enum):
    """Payment transaction status."""
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class NotificationType(str, Enum):
    """Notification/email types."""
    WELCOME = "welcome"
    ONBOARDING = "onboarding"
    PAYMENT_SUCCEEDED = "payment_succeeded"
    PAYMENT_FAILED = "payment_failed"
    RENEWAL_REMINDER = "renewal_reminder"
    DUNNING_1 = "dunning_1"
    DUNNING_2 = "dunning_2"
    DUNNING_FINAL = "dunning_final"
    SUSPENSION_WARNING = "suspension_warning"
    ACCOUNT_SUSPENDED = "account_suspended"
    CANCELLATION_CONFIRMED = "cancellation_confirmed"
    WIN_BACK = "win_back"


# =============================================================================
# CUSTOMER MODEL
# =============================================================================

class Customer(Base):
    """
    Core customer record with lifecycle state tracking.
    Maintains current phase and status with full audit history.
    """
    __tablename__ = "nexus_customers"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # External IDs
    stripe_customer_id = Column(String(255), unique=True, nullable=False, index=True)
    clerk_user_id = Column(String(255), unique=True, nullable=True, index=True)
    
    # Profile
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)
    
    # Lifecycle state
    current_phase = Column(SQLEnum(NexusPhase), default=NexusPhase.PHASE_1_ACQUISITION)
    status = Column(SQLEnum(CustomerStatus), default=CustomerStatus.LEAD)
    tier = Column(SQLEnum(SubscriptionTier), nullable=True)
    
    # Subscription tracking
    subscription_id = Column(String(255), nullable=True, index=True)
    subscription_started_at = Column(DateTime, nullable=True)
    subscription_ends_at = Column(DateTime, nullable=True)
    trial_ends_at = Column(DateTime, nullable=True)
    
    # Billing
    mrr = Column(Float, default=0.0)  # Monthly recurring revenue
    lifetime_value = Column(Float, default=0.0)
    last_payment_at = Column(DateTime, nullable=True)
    next_billing_at = Column(DateTime, nullable=True)
    
    # Dunning state
    dunning_started_at = Column(DateTime, nullable=True)
    dunning_attempts = Column(Integer, default=0)
    last_dunning_at = Column(DateTime, nullable=True)
    
    # Engagement metrics
    last_active_at = Column(DateTime, nullable=True)
    login_count = Column(Integer, default=0)
    feature_usage = Column(JSONB, default=dict)
    
    # Metadata
    metadata = Column("metadata", JSONB, default=dict)
    tags = Column(JSONB, default=list)
    
    # Versioning for MVCC
    version = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    phase_transitions = relationship("PhaseTransition", back_populates="customer", order_by="PhaseTransition.created_at.desc()")
    payments = relationship("Payment", back_populates="customer", order_by="Payment.created_at.desc()")
    notifications = relationship("Notification", back_populates="customer", order_by="Notification.created_at.desc()")
    entitlements = relationship("Entitlement", back_populates="customer")
    
    # Indexes
    __table_args__ = (
        Index('idx_customer_phase_status', 'current_phase', 'status'),
        Index('idx_customer_tier', 'tier'),
        Index('idx_customer_next_billing', 'next_billing_at'),
        Index('idx_customer_dunning', 'dunning_started_at', 'dunning_attempts'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "stripe_customer_id": self.stripe_customer_id,
            "email": self.email,
            "name": self.name,
            "current_phase": self.current_phase.value if self.current_phase else None,
            "status": self.status.value if self.status else None,
            "tier": self.tier.value if self.tier else None,
            "mrr": self.mrr,
            "lifetime_value": self.lifetime_value,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# PHASE TRANSITION MODEL
# =============================================================================

class PhaseTransition(Base):
    """
    Audit log of all phase transitions.
    Enables rollback and lifecycle analysis.
    """
    __tablename__ = "nexus_phase_transitions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("nexus_customers.id"), nullable=False, index=True)
    
    # Transition details
    from_phase = Column(SQLEnum(NexusPhase), nullable=True)
    to_phase = Column(SQLEnum(NexusPhase), nullable=False)
    from_status = Column(SQLEnum(CustomerStatus), nullable=True)
    to_status = Column(SQLEnum(CustomerStatus), nullable=False)
    
    # Trigger info
    trigger_event = Column(String(100), nullable=False)  # e.g., "checkout.session.completed"
    trigger_event_id = Column(String(255), nullable=True)  # Stripe event ID
    trigger_source = Column(String(50), default="stripe")  # stripe, manual, system
    
    # Handler result
    handler_action = Column(String(100), nullable=False)  # e.g., "convert_lead"
    handler_success = Column(Boolean, default=True)
    handler_error = Column(Text, nullable=True)
    handler_duration_ms = Column(Integer, nullable=True)

    # Metadata
    metadata = Column("metadata", JSONB, default=dict)

    # Customer version at transition time (for MVCC)
    customer_version = Column(Integer, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer", back_populates="phase_transitions")
    
    __table_args__ = (
        Index('idx_transition_customer_time', 'customer_id', 'created_at'),
        Index('idx_transition_trigger', 'trigger_event'),
    )


# =============================================================================
# PAYMENT MODEL
# =============================================================================

class Payment(Base):
    """
    Payment transaction records linked to customer lifecycle.
    """
    __tablename__ = "nexus_payments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("nexus_customers.id"), nullable=False, index=True)
    
    # Stripe IDs
    stripe_payment_intent_id = Column(String(255), unique=True, nullable=True, index=True)
    stripe_charge_id = Column(String(255), unique=True, nullable=True, index=True)
    stripe_invoice_id = Column(String(255), nullable=True, index=True)
    stripe_subscription_id = Column(String(255), nullable=True, index=True)
    
    # Amount
    amount = Column(Integer, nullable=False)  # In cents
    currency = Column(String(3), default="usd")
    
    # Status
    status = Column(SQLEnum(PaymentStatus), default=PaymentStatus.PENDING)
    failure_code = Column(String(100), nullable=True)
    failure_message = Column(Text, nullable=True)
    
    # Refund tracking
    refunded_amount = Column(Integer, default=0)
    refund_reason = Column(String(255), nullable=True)
    
    # Metadata
    description = Column(String(500), nullable=True)
    metadata = Column("metadata", JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="payments")
    
    @property
    def amount_display(self) -> str:
        return f"${self.amount / 100:.2f} {self.currency.upper()}"


# =============================================================================
# ENTITLEMENT MODEL
# =============================================================================

class Entitlement(Base):
    """
    Service entitlements granted to customers based on their subscription.
    """
    __tablename__ = "nexus_entitlements"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("nexus_customers.id"), nullable=False, index=True)
    
    # Entitlement details
    feature_key = Column(String(100), nullable=False)  # e.g., "ragnarok_access", "trinity_access"
    feature_name = Column(String(255), nullable=False)
    
    # Limits
    quota = Column(Integer, nullable=True)  # null = unlimited
    used = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    granted_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
    
    # Source
    source_subscription_id = Column(String(255), nullable=True)
    source_product_id = Column(String(255), nullable=True)

    metadata = Column("metadata", JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="entitlements")
    
    __table_args__ = (
        UniqueConstraint('customer_id', 'feature_key', name='uq_customer_feature'),
        Index('idx_entitlement_active', 'customer_id', 'is_active'),
    )


# =============================================================================
# NOTIFICATION MODEL
# =============================================================================

class Notification(Base):
    """
    Notification/email tracking for customer communications.
    """
    __tablename__ = "nexus_notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("nexus_customers.id"), nullable=False, index=True)
    
    # Notification details
    type = Column(SQLEnum(NotificationType), nullable=False)
    channel = Column(String(50), default="email")  # email, sms, push, slack
    
    # Content
    subject = Column(String(500), nullable=True)
    template_id = Column(String(100), nullable=True)
    
    # Delivery status
    sent_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    opened_at = Column(DateTime, nullable=True)
    clicked_at = Column(DateTime, nullable=True)
    bounced_at = Column(DateTime, nullable=True)
    
    # External tracking
    external_id = Column(String(255), nullable=True)  # e.g., SendGrid message ID
    
    # Trigger context
    trigger_phase = Column(SQLEnum(NexusPhase), nullable=True)
    trigger_event = Column(String(100), nullable=True)

    metadata = Column("metadata", JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    customer = relationship("Customer", back_populates="notifications")
    
    __table_args__ = (
        Index('idx_notification_type_sent', 'customer_id', 'type', 'sent_at'),
    )


# =============================================================================
# DUNNING SCHEDULE MODEL
# =============================================================================

class DunningSchedule(Base):
    """
    Dunning attempt schedule and tracking.
    """
    __tablename__ = "nexus_dunning_schedules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("nexus_customers.id"), nullable=False, index=True)
    
    # Invoice being dunned
    stripe_invoice_id = Column(String(255), nullable=False, index=True)
    amount_due = Column(Integer, nullable=False)
    
    # Schedule
    attempt_number = Column(Integer, default=1)
    max_attempts = Column(Integer, default=4)
    
    # Timing
    scheduled_at = Column(DateTime, nullable=False)
    executed_at = Column(DateTime, nullable=True)
    
    # Result
    retry_succeeded = Column(Boolean, nullable=True)
    failure_reason = Column(Text, nullable=True)
    
    # Actions taken
    notification_sent = Column(Boolean, default=False)
    card_update_requested = Column(Boolean, default=False)
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_dunning_active_scheduled', 'is_active', 'scheduled_at'),
    )
