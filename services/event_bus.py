"""
================================================================================
NEXUS ORCHESTRATOR - Event Bus (RabbitMQ)
================================================================================
Production-grade event bus with dead letter queues, retry logic, and observability.
================================================================================
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType
from aio_pika.abc import AbstractIncomingMessage
import hashlib

logger = logging.getLogger("nexus.eventbus")


# =============================================================================
# EVENT DEFINITIONS
# =============================================================================

class EventPriority(Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class NexusEvent:
    """Base event structure for all Nexus events."""
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=lambda: hashlib.md5(
        f"{datetime.utcnow().isoformat()}-{id(object())}".encode()
    ).hexdigest()[:16])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"
    source: str = "nexus-orchestrator"
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "version": self.version,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "priority": self.priority.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NexusEvent":
        return cls(
            event_id=data.get("event_id"),
            event_type=data["event_type"],
            payload=data["payload"],
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            version=data.get("version", "1.0"),
            source=data.get("source", "unknown"),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            priority=EventPriority(data.get("priority", 5)),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# =============================================================================
# DEAD LETTER QUEUE HANDLER
# =============================================================================

@dataclass
class PoisonMessage:
    """Tracks poison messages for analysis."""
    event: NexusEvent
    error: str
    failed_at: str
    handler: str
    stack_trace: Optional[str] = None


class DeadLetterHandler:
    """
    Handles messages that fail processing after max retries.
    Stores for manual intervention and analysis.
    """
    
    def __init__(self, storage_path: str = "/tmp/nexus_dlq"):
        self.storage_path = storage_path
        self.poison_messages: List[PoisonMessage] = []
    
    async def handle_dead_letter(
        self,
        event: NexusEvent,
        error: str,
        handler: str,
        stack_trace: Optional[str] = None
    ):
        """Store failed message for manual review."""
        poison = PoisonMessage(
            event=event,
            error=error,
            failed_at=datetime.utcnow().isoformat(),
            handler=handler,
            stack_trace=stack_trace
        )
        self.poison_messages.append(poison)
        
        logger.error(
            f"Message moved to DLQ: event_id={event.event_id}, "
            f"type={event.event_type}, error={error}"
        )
        
        # Persist to file for durability
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            filepath = f"{self.storage_path}/{event.event_id}.json"
            with open(filepath, "w") as f:
                json.dump({
                    "event": event.to_dict(),
                    "error": error,
                    "failed_at": poison.failed_at,
                    "handler": handler,
                    "stack_trace": stack_trace
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist DLQ message: {e}")
    
    async def get_poison_messages(self) -> List[PoisonMessage]:
        """Get all poison messages for review."""
        return self.poison_messages
    
    async def reprocess_message(self, event_id: str) -> bool:
        """Attempt to reprocess a poison message."""
        for i, poison in enumerate(self.poison_messages):
            if poison.event.event_id == event_id:
                # Reset retry count and remove from DLQ
                poison.event.retry_count = 0
                self.poison_messages.pop(i)
                return True
        return False


# =============================================================================
# RABBITMQ EVENT BUS
# =============================================================================

class RabbitMQEventBus:
    """
    Production-grade event bus using RabbitMQ.
    Features:
    - Topic-based routing
    - Dead letter queues
    - Retry with exponential backoff
    - Priority queues
    - Graceful shutdown
    """
    
    # Exchange names
    MAIN_EXCHANGE = "nexus.events"
    DLQ_EXCHANGE = "nexus.events.dlq"
    RETRY_EXCHANGE = "nexus.events.retry"
    
    def __init__(
        self,
        connection_url: str = "amqp://guest:guest@localhost:5672/",
        prefetch_count: int = 10
    ):
        self.connection_url = connection_url
        self.prefetch_count = prefetch_count
        
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.main_exchange: Optional[aio_pika.Exchange] = None
        self.dlq_exchange: Optional[aio_pika.Exchange] = None
        self.retry_exchange: Optional[aio_pika.Exchange] = None
        
        self.handlers: Dict[str, List[Callable]] = {}
        self.dlq_handler = DeadLetterHandler()
        self._running = False
        self._consumer_tags: List[str] = []
    
    async def connect(self):
        """Establish connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(
                self.connection_url,
                heartbeat=60,
                blocked_connection_timeout=300
            )
            
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.prefetch_count)
            
            # Declare exchanges
            self.main_exchange = await self.channel.declare_exchange(
                self.MAIN_EXCHANGE,
                ExchangeType.TOPIC,
                durable=True
            )
            
            self.dlq_exchange = await self.channel.declare_exchange(
                self.DLQ_EXCHANGE,
                ExchangeType.DIRECT,
                durable=True
            )
            
            self.retry_exchange = await self.channel.declare_exchange(
                self.RETRY_EXCHANGE,
                ExchangeType.DIRECT,
                durable=True
            )
            
            # Declare DLQ
            dlq = await self.channel.declare_queue(
                "nexus.dlq",
                durable=True
            )
            await dlq.bind(self.dlq_exchange, routing_key="dead_letter")
            
            self._running = True
            logger.info("RabbitMQ event bus connected")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self):
        """Gracefully disconnect from RabbitMQ."""
        self._running = False
        
        # Cancel all consumers
        for tag in self._consumer_tags:
            try:
                await self.channel.cancel(tag)
            except Exception:
                pass
        
        if self.connection:
            await self.connection.close()
        
        logger.info("RabbitMQ event bus disconnected")
    
    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL
    ) -> str:
        """
        Publish an event to the bus.
        Returns the event ID.
        """
        event = NexusEvent(
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority
        )
        
        message = Message(
            body=event.to_json().encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json",
            message_id=event.event_id,
            timestamp=datetime.utcnow(),
            priority=priority.value
        )
        
        # Route based on event type (e.g., nexus.customer.converted -> nexus.customer.*)
        routing_key = event_type
        
        await self.main_exchange.publish(
            message,
            routing_key=routing_key
        )
        
        logger.info(f"Published event: {event_type}, id={event.event_id}")
        return event.event_id
    
    async def subscribe(
        self,
        event_pattern: str,
        handler: Callable[[NexusEvent], Any],
        queue_name: Optional[str] = None
    ):
        """
        Subscribe to events matching a pattern.
        Pattern examples:
        - "nexus.customer.*" - All customer events
        - "nexus.*.completed" - All completed events
        - "nexus.payment.#" - Payment events at any depth
        """
        if event_pattern not in self.handlers:
            self.handlers[event_pattern] = []
        self.handlers[event_pattern].append(handler)
        
        # Create queue for this subscription
        queue_name = queue_name or f"nexus.handler.{event_pattern.replace('.', '_')}"
        
        # Declare queue with DLQ binding
        queue = await self.channel.declare_queue(
            queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.DLQ_EXCHANGE,
                "x-dead-letter-routing-key": "dead_letter",
                "x-message-ttl": 86400000,  # 24 hours
            }
        )
        
        # Bind to exchange with pattern
        await queue.bind(self.main_exchange, routing_key=event_pattern)
        
        # Start consuming
        async def process_message(message: AbstractIncomingMessage):
            async with message.process(requeue=False):
                await self._handle_message(message, event_pattern)
        
        consumer_tag = await queue.consume(process_message)
        self._consumer_tags.append(consumer_tag)
        
        logger.info(f"Subscribed to: {event_pattern}")
    
    async def _handle_message(
        self,
        message: AbstractIncomingMessage,
        pattern: str
    ):
        """Process incoming message with retry logic."""
        try:
            event = NexusEvent.from_dict(json.loads(message.body.decode()))
        except Exception as e:
            logger.error(f"Failed to parse message: {e}")
            await self.dlq_handler.handle_dead_letter(
                NexusEvent(event_type="unknown", payload={"raw": message.body.decode()}),
                str(e),
                pattern
            )
            return
        
        handlers = self.handlers.get(pattern, [])
        
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
                
                logger.debug(f"Handled event: {event.event_type}")
                
            except Exception as e:
                event.retry_count += 1
                
                if event.retry_count >= event.max_retries:
                    # Move to DLQ
                    import traceback
                    await self.dlq_handler.handle_dead_letter(
                        event,
                        str(e),
                        handler.__name__,
                        traceback.format_exc()
                    )
                else:
                    # Retry with exponential backoff
                    delay = min(300, 2 ** event.retry_count)  # Max 5 minutes
                    logger.warning(
                        f"Retrying event {event.event_id} in {delay}s "
                        f"(attempt {event.retry_count}/{event.max_retries})"
                    )
                    await self._schedule_retry(event, delay)
    
    async def _schedule_retry(self, event: NexusEvent, delay_seconds: int):
        """Schedule event for retry with delay."""
        # Create retry queue with TTL
        retry_queue_name = f"nexus.retry.{delay_seconds}s"
        
        retry_queue = await self.channel.declare_queue(
            retry_queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.MAIN_EXCHANGE,
                "x-dead-letter-routing-key": event.event_type,
                "x-message-ttl": delay_seconds * 1000,
            }
        )
        
        await retry_queue.bind(self.retry_exchange, routing_key=retry_queue_name)
        
        message = Message(
            body=event.to_json().encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json"
        )
        
        await self.retry_exchange.publish(
            message,
            routing_key=retry_queue_name
        )


# =============================================================================
# IN-MEMORY EVENT BUS (FOR TESTING/SIMPLE DEPLOYMENTS)
# =============================================================================

class InMemoryEventBus:
    """
    Simple in-memory event bus for testing and single-instance deployments.
    """
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.events: List[NexusEvent] = []
        self.dlq: List[NexusEvent] = []
    
    async def connect(self):
        """No-op for in-memory bus."""
        logger.info("In-memory event bus initialized")
    
    async def disconnect(self):
        """No-op for in-memory bus."""
        pass
    
    async def publish(
        self,
        event_type: str,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
        priority: EventPriority = EventPriority.NORMAL
    ) -> str:
        """Publish event and immediately dispatch to handlers."""
        event = NexusEvent(
            event_type=event_type,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority
        )
        
        self.events.append(event)
        
        # Find matching handlers
        for pattern, handlers in self.handlers.items():
            if self._matches_pattern(event_type, pattern):
                for handler in handlers:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Handler error: {e}")
                        self.dlq.append(event)
        
        logger.info(f"Published event: {event_type}")
        return event.event_id
    
    async def subscribe(
        self,
        event_pattern: str,
        handler: Callable[[NexusEvent], Any],
        queue_name: Optional[str] = None
    ):
        """Subscribe to event pattern."""
        if event_pattern not in self.handlers:
            self.handlers[event_pattern] = []
        self.handlers[event_pattern].append(handler)
        logger.info(f"Subscribed to: {event_pattern}")
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches subscription pattern."""
        if pattern == event_type:
            return True
        
        # Handle wildcards
        pattern_parts = pattern.split(".")
        event_parts = event_type.split(".")
        
        if len(pattern_parts) != len(event_parts) and "#" not in pattern:
            return False
        
        for p, e in zip(pattern_parts, event_parts):
            if p == "*":
                continue
            if p == "#":
                return True
            if p != e:
                return False
        
        return True


# =============================================================================
# FACTORY
# =============================================================================

async def create_event_bus(
    use_rabbitmq: bool = True,
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
):
    """Create and connect appropriate event bus."""
    if use_rabbitmq:
        bus = RabbitMQEventBus(rabbitmq_url)
    else:
        bus = InMemoryEventBus()
    
    await bus.connect()
    return bus
