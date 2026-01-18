"""
================================================================================
NEXUS ORCHESTRATOR - Notification Service
================================================================================
Multi-channel notification service for customer lifecycle communications.
Supports: Email (SendGrid), Slack, SMS (Twilio), and webhooks.
================================================================================
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import httpx
import json

logger = logging.getLogger("nexus.notifications")


# =============================================================================
# NOTIFICATION TYPES
# =============================================================================

class NotificationChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    PUSH = "push"


class NotificationTemplate(Enum):
    # Onboarding
    WELCOME = "welcome"
    ONBOARDING_STEP_1 = "onboarding_step_1"
    ONBOARDING_STEP_2 = "onboarding_step_2"
    ONBOARDING_COMPLETE = "onboarding_complete"
    
    # Provisioning
    PROVISIONING_STARTED = "provisioning_started"
    PROVISIONING_COMPLETE = "provisioning_complete"
    
    # Payments
    PAYMENT_SUCCEEDED = "payment_succeeded"
    PAYMENT_RECEIPT = "payment_receipt"
    PAYMENT_FAILED = "payment_failed"
    
    # Dunning
    DUNNING_REMINDER_1 = "dunning_reminder_1"
    DUNNING_REMINDER_2 = "dunning_reminder_2"
    DUNNING_FINAL_WARNING = "dunning_final_warning"
    DUNNING_CARD_UPDATE = "dunning_card_update"
    
    # Renewal
    RENEWAL_REMINDER_7D = "renewal_reminder_7d"
    RENEWAL_REMINDER_3D = "renewal_reminder_3d"
    RENEWAL_REMINDER_1D = "renewal_reminder_1d"
    
    # Suspension
    SUSPENSION_WARNING = "suspension_warning"
    ACCOUNT_SUSPENDED = "account_suspended"
    ACCOUNT_REACTIVATED = "account_reactivated"
    
    # Cancellation
    CANCELLATION_CONFIRMED = "cancellation_confirmed"
    CANCELLATION_FEEDBACK = "cancellation_feedback"
    
    # Win-back
    WIN_BACK_7D = "win_back_7d"
    WIN_BACK_30D = "win_back_30d"
    WIN_BACK_90D = "win_back_90d"
    
    # Checkout Recovery
    CHECKOUT_ABANDONED_1H = "checkout_abandoned_1h"
    CHECKOUT_ABANDONED_24H = "checkout_abandoned_24h"


@dataclass
class NotificationRequest:
    """Request to send a notification."""
    template: NotificationTemplate
    recipient_email: str
    recipient_name: Optional[str]
    channel: NotificationChannel
    variables: Dict[str, Any]
    customer_id: Optional[str] = None
    priority: str = "normal"  # low, normal, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template": self.template.value,
            "recipient_email": self.recipient_email,
            "recipient_name": self.recipient_name,
            "channel": self.channel.value,
            "variables": self.variables,
            "customer_id": self.customer_id,
            "priority": self.priority,
        }


@dataclass
class NotificationResult:
    """Result of sending a notification."""
    success: bool
    channel: NotificationChannel
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


# =============================================================================
# EMAIL TEMPLATES
# =============================================================================

EMAIL_TEMPLATES: Dict[NotificationTemplate, Dict[str, str]] = {
    NotificationTemplate.WELCOME: {
        "subject": "Welcome to {company_name}! 🚀",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #00CED1;">Welcome, {customer_name}!</h1>
            <p>We're thrilled to have you on board. Your journey to AI-powered automation starts now.</p>
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #00CED1; margin-top: 0;">Your subscription includes:</h3>
                <ul style="color: #ffffff;">
                    {entitlements_list}
                </ul>
            </div>
            <p>Get started with our quick setup guide:</p>
            <a href="{onboarding_url}" style="display: inline-block; background: #00CED1; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Start Onboarding →</a>
            <p style="color: #666; margin-top: 30px; font-size: 12px;">
                Questions? Reply to this email or visit our <a href="{support_url}">support center</a>.
            </p>
        </div>
        """,
    },
    
    NotificationTemplate.PAYMENT_SUCCEEDED: {
        "subject": "Payment Confirmed - ${amount}",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #00CED1;">Payment Received ✓</h2>
            <p>Hi {customer_name},</p>
            <p>We've received your payment of <strong>${amount}</strong>.</p>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 20px 0;">
                <p style="margin: 5px 0;"><strong>Invoice:</strong> {invoice_id}</p>
                <p style="margin: 5px 0;"><strong>Date:</strong> {payment_date}</p>
                <p style="margin: 5px 0;"><strong>Card:</strong> •••• {card_last4}</p>
            </div>
            <a href="{receipt_url}" style="color: #00CED1;">View Receipt →</a>
        </div>
        """,
    },
    
    NotificationTemplate.PAYMENT_FAILED: {
        "subject": "⚠️ Payment Failed - Action Required",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #e74c3c;">Payment Failed</h2>
            <p>Hi {customer_name},</p>
            <p>We couldn't process your payment of <strong>${amount}</strong>.</p>
            <div style="background: #fdf2f2; border-left: 4px solid #e74c3c; padding: 15px; margin: 20px 0;">
                <p style="margin: 0;"><strong>Reason:</strong> {failure_reason}</p>
            </div>
            <p>Please update your payment method to avoid service interruption:</p>
            <a href="{update_payment_url}" style="display: inline-block; background: #e74c3c; color: #fff; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Update Payment Method →</a>
            <p style="color: #666; margin-top: 20px;">We'll automatically retry in 3 days.</p>
        </div>
        """,
    },
    
    NotificationTemplate.DUNNING_REMINDER_1: {
        "subject": "Friendly Reminder: Update Your Payment Method",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #f39c12;">Payment Update Needed</h2>
            <p>Hi {customer_name},</p>
            <p>We noticed your recent payment of <strong>${amount}</strong> didn't go through.</p>
            <p>To keep your services running smoothly, please update your payment information:</p>
            <a href="{update_payment_url}" style="display: inline-block; background: #00CED1; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Update Payment →</a>
            <p style="color: #666; margin-top: 20px;">Your services will continue for the next {days_remaining} days.</p>
        </div>
        """,
    },
    
    NotificationTemplate.DUNNING_FINAL_WARNING: {
        "subject": "🚨 Final Notice: Service Suspension in 24 Hours",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: #e74c3c; color: white; padding: 15px; border-radius: 4px 4px 0 0;">
                <h2 style="margin: 0;">⚠️ Final Payment Notice</h2>
            </div>
            <div style="border: 1px solid #e74c3c; padding: 20px; border-radius: 0 0 4px 4px;">
                <p>Hi {customer_name},</p>
                <p><strong>Your account will be suspended in 24 hours</strong> unless payment is received.</p>
                <p>Outstanding amount: <strong>${amount}</strong></p>
                <a href="{update_payment_url}" style="display: inline-block; background: #e74c3c; color: #fff; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold; margin: 15px 0;">Pay Now →</a>
                <p style="color: #666; font-size: 12px;">Need help? Contact us at {support_email}</p>
            </div>
        </div>
        """,
    },
    
    NotificationTemplate.ACCOUNT_SUSPENDED: {
        "subject": "Account Suspended - {company_name}",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #e74c3c;">Account Suspended</h2>
            <p>Hi {customer_name},</p>
            <p>Your account has been suspended due to an unpaid balance of <strong>${amount}</strong>.</p>
            <p>Your data is safe and your account can be reactivated immediately upon payment:</p>
            <a href="{reactivate_url}" style="display: inline-block; background: #00CED1; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Reactivate Account →</a>
            <p style="color: #666; margin-top: 20px;">
                <strong>What happens to your data?</strong><br>
                Your data will be retained for 30 days. After that, it may be permanently deleted.
            </p>
        </div>
        """,
    },
    
    NotificationTemplate.CANCELLATION_CONFIRMED: {
        "subject": "We're sorry to see you go 😢",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Cancellation Confirmed</h2>
            <p>Hi {customer_name},</p>
            <p>Your subscription has been cancelled. You'll continue to have access until <strong>{access_end_date}</strong>.</p>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 20px 0;">
                <p style="margin: 0;"><strong>Before you go:</strong></p>
                <ul>
                    <li><a href="{export_data_url}">Export your data</a></li>
                    <li><a href="{feedback_url}">Share feedback</a></li>
                </ul>
            </div>
            <p>Changed your mind? You can resubscribe anytime:</p>
            <a href="{resubscribe_url}" style="color: #00CED1;">Resubscribe →</a>
        </div>
        """,
    },
    
    NotificationTemplate.WIN_BACK_7D: {
        "subject": "We miss you! Here's 20% off to come back 🎁",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #00CED1;">We'd Love to Have You Back!</h2>
            <p>Hi {customer_name},</p>
            <p>It's been a week since you left, and we wanted to check in.</p>
            <div style="background: linear-gradient(135deg, #00CED1 0%, #0099aa 100%); color: #fff; padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;">
                <p style="font-size: 24px; margin: 0;">20% OFF</p>
                <p style="margin: 10px 0 0 0;">Use code: <strong>COMEBACK20</strong></p>
            </div>
            <a href="{resubscribe_url}?code=COMEBACK20" style="display: inline-block; background: #00CED1; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Reactivate with Discount →</a>
        </div>
        """,
    },
    
    NotificationTemplate.CHECKOUT_ABANDONED_1H: {
        "subject": "You left something behind... 🛒",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Complete Your Order</h2>
            <p>Hi there,</p>
            <p>Looks like you didn't finish your checkout. Your cart is still waiting for you!</p>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 20px 0;">
                <p style="margin: 0;"><strong>{product_name}</strong></p>
                <p style="margin: 5px 0; color: #00CED1; font-size: 20px;">${amount}</p>
            </div>
            <a href="{checkout_url}" style="display: inline-block; background: #00CED1; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Complete Purchase →</a>
            <p style="color: #666; margin-top: 20px; font-size: 12px;">
                Questions about the product? <a href="{support_url}">Chat with us</a>
            </p>
        </div>
        """,
    },
    
    NotificationTemplate.PROVISIONING_COMPLETE: {
        "subject": "🎉 Your services are ready!",
        "html": """
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #00CED1;">You're All Set!</h2>
            <p>Hi {customer_name},</p>
            <p>Great news! Your services have been provisioned and are ready to use.</p>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 20px 0;">
                <h3 style="margin-top: 0;">Your Active Services:</h3>
                <ul>
                    {entitlements_list}
                </ul>
            </div>
            <a href="{dashboard_url}" style="display: inline-block; background: #00CED1; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold;">Go to Dashboard →</a>
        </div>
        """,
    },
}


# =============================================================================
# NOTIFICATION SERVICE
# =============================================================================

class NotificationService:
    """
    Multi-channel notification service.
    Handles email, Slack, SMS, and webhook notifications.
    """
    
    def __init__(
        self,
        sendgrid_api_key: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_phone: Optional[str] = None,
        from_email: str = "hello@barriosa2i.com",
        company_name: str = "Barrios A2I",
        base_url: str = "https://www.barriosa2i.com"
    ):
        self.sendgrid_api_key = sendgrid_api_key
        self.slack_webhook_url = slack_webhook_url
        self.twilio_account_sid = twilio_account_sid
        self.twilio_auth_token = twilio_auth_token
        self.twilio_phone = twilio_phone
        self.from_email = from_email
        self.company_name = company_name
        self.base_url = base_url
        
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    # =========================================================================
    # HIGH-LEVEL METHODS (Called by phase handlers)
    # =========================================================================
    
    async def send_welcome(self, customer) -> NotificationResult:
        """Send welcome email to new customer."""
        entitlements_html = "".join(
            f"<li>{e.feature_name}</li>" 
            for e in customer.entitlements 
            if e.is_active
        )
        
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.WELCOME,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "company_name": self.company_name,
                "entitlements_list": entitlements_html or "<li>Your subscription is being set up...</li>",
                "onboarding_url": f"{self.base_url}/onboarding",
                "support_url": f"{self.base_url}/support",
            },
            customer_id=str(customer.id)
        ))
    
    async def send_payment_receipt(
        self,
        customer,
        amount: int,
        invoice_id: str = None,
        card_last4: str = "****"
    ) -> NotificationResult:
        """Send payment receipt email."""
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.PAYMENT_SUCCEEDED,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "amount": f"{amount / 100:.2f}",
                "invoice_id": invoice_id or "N/A",
                "payment_date": datetime.utcnow().strftime("%B %d, %Y"),
                "card_last4": card_last4,
                "receipt_url": f"{self.base_url}/billing/receipts/{invoice_id}",
            },
            customer_id=str(customer.id)
        ))
    
    async def send_payment_failed(
        self,
        customer,
        amount: int,
        failure_reason: str = "Card declined"
    ) -> NotificationResult:
        """Send payment failed notification."""
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.PAYMENT_FAILED,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "amount": f"{amount / 100:.2f}",
                "failure_reason": failure_reason,
                "update_payment_url": f"{self.base_url}/billing/update-payment",
            },
            customer_id=str(customer.id),
            priority="high"
        ))
    
    async def send_dunning_notification(
        self,
        customer,
        template: NotificationTemplate,
        amount: int,
        days_remaining: int = 7
    ) -> NotificationResult:
        """Send dunning sequence notification."""
        return await self.send_notification(NotificationRequest(
            template=template,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "amount": f"{amount / 100:.2f}",
                "days_remaining": days_remaining,
                "update_payment_url": f"{self.base_url}/billing/update-payment",
                "support_email": "support@barriosa2i.com",
            },
            customer_id=str(customer.id),
            priority="high"
        ))
    
    async def send_suspension_notice(self, customer, amount: int) -> NotificationResult:
        """Send account suspension notice."""
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.ACCOUNT_SUSPENDED,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "amount": f"{amount / 100:.2f}",
                "reactivate_url": f"{self.base_url}/billing/reactivate",
            },
            customer_id=str(customer.id),
            priority="critical"
        ))
    
    async def send_provisioning_complete(
        self,
        customer,
        entitlements: List[str]
    ) -> NotificationResult:
        """Send provisioning complete notification."""
        entitlements_html = "".join(f"<li>{e}</li>" for e in entitlements)
        
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.PROVISIONING_COMPLETE,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "entitlements_list": entitlements_html,
                "dashboard_url": f"{self.base_url}/dashboard",
            },
            customer_id=str(customer.id)
        ))
    
    async def send_offboarding(self, customer) -> NotificationResult:
        """Send cancellation confirmation and offboarding info."""
        from datetime import timedelta
        access_end = datetime.utcnow() + timedelta(days=30)
        
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.CANCELLATION_CONFIRMED,
            recipient_email=customer.email,
            recipient_name=customer.name,
            channel=NotificationChannel.EMAIL,
            variables={
                "customer_name": customer.name or "there",
                "access_end_date": access_end.strftime("%B %d, %Y"),
                "export_data_url": f"{self.base_url}/settings/export",
                "feedback_url": f"{self.base_url}/feedback",
                "resubscribe_url": f"{self.base_url}/pricing",
            },
            customer_id=str(customer.id)
        ))
    
    async def send_abandonment_recovery(
        self,
        email: str,
        session_id: str,
        amount: int
    ) -> NotificationResult:
        """Send checkout abandonment recovery email."""
        return await self.send_notification(NotificationRequest(
            template=NotificationTemplate.CHECKOUT_ABANDONED_1H,
            recipient_email=email,
            recipient_name=None,
            channel=NotificationChannel.EMAIL,
            variables={
                "product_name": "AI Automation Services",
                "amount": f"{amount / 100:.2f}" if amount else "Custom pricing",
                "checkout_url": f"{self.base_url}/checkout/recover/{session_id}",
                "support_url": f"{self.base_url}/support",
            }
        ))
    
    # =========================================================================
    # CORE SEND METHOD
    # =========================================================================
    
    async def send_notification(
        self,
        request: NotificationRequest
    ) -> NotificationResult:
        """Send a notification through the specified channel."""
        try:
            if request.channel == NotificationChannel.EMAIL:
                return await self._send_email(request)
            elif request.channel == NotificationChannel.SLACK:
                return await self._send_slack(request)
            elif request.channel == NotificationChannel.SMS:
                return await self._send_sms(request)
            elif request.channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(request)
            else:
                return NotificationResult(
                    success=False,
                    channel=request.channel,
                    error=f"Unsupported channel: {request.channel}"
                )
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return NotificationResult(
                success=False,
                channel=request.channel,
                error=str(e)
            )
    
    async def _send_email(self, request: NotificationRequest) -> NotificationResult:
        """Send email via SendGrid."""
        if not self.sendgrid_api_key:
            logger.warning("SendGrid not configured, logging email instead")
            logger.info(f"EMAIL to {request.recipient_email}: {request.template.value}")
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id="simulated"
            )
        
        template_data = EMAIL_TEMPLATES.get(request.template)
        if not template_data:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error=f"Template not found: {request.template}"
            )
        
        # Render template
        subject = template_data["subject"].format(**request.variables)
        html_content = template_data["html"].format(**request.variables)
        
        # Send via SendGrid
        response = await self.client.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {self.sendgrid_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "personalizations": [{
                    "to": [{"email": request.recipient_email, "name": request.recipient_name}],
                }],
                "from": {"email": self.from_email, "name": self.company_name},
                "subject": subject,
                "content": [{"type": "text/html", "value": html_content}],
            }
        )
        
        if response.status_code in (200, 202):
            message_id = response.headers.get("X-Message-Id", "unknown")
            logger.info(f"Email sent: {request.template.value} to {request.recipient_email}")
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id=message_id
            )
        else:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error=f"SendGrid error: {response.status_code}"
            )
    
    async def _send_slack(self, request: NotificationRequest) -> NotificationResult:
        """Send message to Slack webhook."""
        if not self.slack_webhook_url:
            logger.warning("Slack not configured")
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error="Slack webhook not configured"
            )
        
        # Build Slack message
        message = {
            "text": f"[{request.template.value}] Notification for {request.recipient_email}",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{request.template.value}*\n{json.dumps(request.variables, indent=2)}"
                    }
                }
            ]
        }
        
        response = await self.client.post(
            self.slack_webhook_url,
            json=message
        )
        
        return NotificationResult(
            success=response.status_code == 200,
            channel=NotificationChannel.SLACK,
            error=None if response.status_code == 200 else f"Slack error: {response.status_code}"
        )
    
    async def _send_sms(self, request: NotificationRequest) -> NotificationResult:
        """Send SMS via Twilio."""
        if not all([self.twilio_account_sid, self.twilio_auth_token, self.twilio_phone]):
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SMS,
                error="Twilio not configured"
            )
        
        # SMS would require phone number in variables
        phone = request.variables.get("phone")
        if not phone:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SMS,
                error="Phone number not provided"
            )
        
        # Implementation for Twilio would go here
        return NotificationResult(
            success=False,
            channel=NotificationChannel.SMS,
            error="SMS not yet implemented"
        )
    
    async def _send_webhook(self, request: NotificationRequest) -> NotificationResult:
        """Send notification to external webhook."""
        webhook_url = request.variables.get("webhook_url")
        if not webhook_url:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.WEBHOOK,
                error="Webhook URL not provided"
            )
        
        response = await self.client.post(
            webhook_url,
            json=request.to_dict()
        )
        
        return NotificationResult(
            success=response.status_code in (200, 201, 202),
            channel=NotificationChannel.WEBHOOK,
            error=None if response.status_code < 400 else f"Webhook error: {response.status_code}"
        )


# =============================================================================
# INTERNAL SLACK ALERTS (For ops team)
# =============================================================================

class OpsAlertService:
    """
    Send internal alerts to ops team for critical events.
    """
    
    def __init__(self, slack_webhook_url: Optional[str] = None):
        self.slack_webhook_url = slack_webhook_url
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def alert_payment_failed(
        self,
        customer_email: str,
        amount: int,
        attempt: int
    ):
        """Alert ops about failed payment."""
        await self._send_alert(
            title="💳 Payment Failed",
            color="#e74c3c",
            fields=[
                {"title": "Customer", "value": customer_email, "short": True},
                {"title": "Amount", "value": f"${amount/100:.2f}", "short": True},
                {"title": "Attempt", "value": str(attempt), "short": True},
            ]
        )
    
    async def alert_customer_churned(
        self,
        customer_email: str,
        lifetime_value: float
    ):
        """Alert ops about churned customer."""
        await self._send_alert(
            title="👋 Customer Churned",
            color="#f39c12",
            fields=[
                {"title": "Customer", "value": customer_email, "short": True},
                {"title": "LTV", "value": f"${lifetime_value:.2f}", "short": True},
            ]
        )
    
    async def alert_high_value_signup(
        self,
        customer_email: str,
        tier: str,
        mrr: float
    ):
        """Alert ops about high-value signup."""
        await self._send_alert(
            title="🎉 New Signup!",
            color="#00CED1",
            fields=[
                {"title": "Customer", "value": customer_email, "short": True},
                {"title": "Tier", "value": tier, "short": True},
                {"title": "MRR", "value": f"${mrr:.2f}", "short": True},
            ]
        )
    
    async def _send_alert(
        self,
        title: str,
        color: str,
        fields: List[Dict]
    ):
        """Send formatted alert to Slack."""
        if not self.slack_webhook_url:
            logger.info(f"OPS ALERT: {title}")
            return
        
        try:
            await self.client.post(
                self.slack_webhook_url,
                json={
                    "attachments": [{
                        "color": color,
                        "title": title,
                        "fields": fields,
                        "footer": "Nexus Orchestrator",
                        "ts": int(datetime.utcnow().timestamp())
                    }]
                }
            )
        except Exception as e:
            logger.error(f"Failed to send ops alert: {e}")
