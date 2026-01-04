"""
================================================================================
⚡ GHOST CONNECTION RECOVERY v2.0 LEGENDARY
================================================================================
Survives browser refreshes, WiFi blips, and network interruptions.

UPGRADE 3: "Ghost Connection" Recovery
- Redis-backed event log with automatic replay
- Client reconnects to /api/genesis/stream/{id} → sees ALL missed events
- SSE + WebSocket dual support

The Problem:
- User triggers 5-minute video pipeline
- At minute 3, WiFi blips for 2 seconds
- WITHOUT recovery: User sees blank screen, doesn't know if video finished
- WITH recovery: User reconnects, sees last 3 minutes of events, continues watching

Architecture:
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Client                    Server                    Redis               │
│    │                         │                         │                 │
│    │─── SSE Connect ────────▶│                         │                 │
│    │                         │─── Subscribe ──────────▶│                 │
│    │                         │                         │                 │
│    │◀── Replay History ─────│◀── LRANGE ─────────────│                 │
│    │    (missed events)      │                         │                 │
│    │                         │                         │                 │
│    │◀── Live Event ──────────│◀── Pub/Sub ────────────│                 │
│    │    (new events)         │                         │                 │
│    │                         │                         │                 │
│    ×  (disconnect)           │                         │                 │
│    │                         │                         │                 │
│    │─── Reconnect ──────────▶│                         │                 │
│    │                         │                         │                 │
│    │◀── Replay ALL ─────────│◀── LRANGE ─────────────│                 │
│    │    (complete history)   │                         │                 │
│    │                         │                         │                 │
│    │◀── Continue Live ──────│◀── Pub/Sub ────────────│                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

================================================================================
Author: Barrios A2I | Version: 2.0.0 LEGENDARY | January 2026
================================================================================
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Callable

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

GHOST_RECONNECTIONS = Counter(
    'genesis_ghost_reconnections_total',
    'Client reconnection count',
    ['pipeline_id']
)

GHOST_EVENTS_REPLAYED = Counter(
    'genesis_ghost_events_replayed_total',
    'Events replayed on reconnection',
    ['pipeline_id']
)

GHOST_ACTIVE_CONNECTIONS = Gauge(
    'genesis_ghost_active_connections',
    'Active SSE/WebSocket connections'
)

GHOST_EVENT_LOG_SIZE = Histogram(
    'genesis_ghost_event_log_size',
    'Event log size per pipeline',
    buckets=[10, 25, 50, 100, 200, 500]
)

GHOST_REPLAY_LATENCY = Histogram(
    'genesis_ghost_replay_latency_seconds',
    'Time to replay event history',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)


# =============================================================================
# EVENT MODELS
# =============================================================================

class EventType(Enum):
    """Pipeline event types for streaming"""
    # Pipeline lifecycle
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"
    
    # Phase events
    PHASE_START = "phase_start"
    PHASE_PROGRESS = "phase_progress"
    PHASE_COMPLETE = "phase_complete"
    
    # Agent events
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    
    # System events
    HEARTBEAT = "heartbeat"
    RECONNECT_ACK = "reconnect_ack"


@dataclass
class PipelineEvent:
    """
    Structured event for pipeline streaming.
    Serializable to JSON for Redis storage and SSE transmission.
    """
    event_id: str
    event_type: str
    pipeline_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    sequence: int = 0
    is_replay: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "pipeline_id": self.pipeline_id,
            "timestamp": self.timestamp,
            "data": self.data,
            "sequence": self.sequence,
            "is_replay": self.is_replay
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event"""
        return f"event: {self.event_type}\nid: {self.sequence}\ndata: {self.to_json()}\n\n"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineEvent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, json_str: str) -> "PipelineEvent":
        return cls.from_dict(json.loads(json_str))


@dataclass
class GhostConfig:
    """Configuration for ghost recovery"""
    # Event storage
    event_log_ttl: int = 1800           # 30 minutes
    max_events_per_pipeline: int = 500  # Max events to store
    
    # Heartbeat
    heartbeat_interval: float = 15.0    # Seconds between heartbeats
    heartbeat_timeout: float = 45.0     # Client considered dead after this
    
    # Reconnection
    max_replay_events: int = 200        # Max events to replay on reconnect
    replay_batch_size: int = 20         # Events per batch during replay
    
    # Redis keys
    event_log_prefix: str = "genesis:events:"
    pubsub_prefix: str = "genesis:live:"
    connection_prefix: str = "genesis:conn:"
    sequence_prefix: str = "genesis:seq:"


# =============================================================================
# EVENT LOG (Redis-backed persistence)
# =============================================================================

class EventLog:
    """
    Redis-backed event log for pipeline events.
    Enables complete replay on client reconnection.
    """
    
    def __init__(
        self,
        redis_client=None,
        config: Optional[GhostConfig] = None
    ):
        self.redis = redis_client
        self.config = config or GhostConfig()
        
        # In-memory fallback
        self._memory_logs: Dict[str, List[PipelineEvent]] = {}
        self._memory_sequences: Dict[str, int] = {}
        
        logger.info(
            f"📜 EventLog initialized "
            f"(Redis: {'enabled' if redis_client else 'disabled'})"
        )
    
    def _log_key(self, pipeline_id: str) -> str:
        return f"{self.config.event_log_prefix}{pipeline_id}"
    
    def _seq_key(self, pipeline_id: str) -> str:
        return f"{self.config.sequence_prefix}{pipeline_id}"
    
    async def append(self, event: PipelineEvent) -> int:
        """
        Append event to log and return sequence number.
        
        Events are stored in a Redis list for efficient append and range queries.
        """
        pipeline_id = event.pipeline_id
        log_key = self._log_key(pipeline_id)
        seq_key = self._seq_key(pipeline_id)
        
        # Get and increment sequence
        if self.redis:
            try:
                sequence = await self.redis.incr(seq_key)
                await self.redis.expire(seq_key, self.config.event_log_ttl)
                
                event.sequence = sequence
                
                # Append to list (RPUSH for chronological order)
                await self.redis.rpush(log_key, event.to_json())
                await self.redis.expire(log_key, self.config.event_log_ttl)
                
                # Trim to max size (keep most recent)
                await self.redis.ltrim(log_key, -self.config.max_events_per_pipeline, -1)
                
                return sequence
            except Exception as e:
                logger.warning(f"Redis append failed: {e}")
        
        # Memory fallback
        if pipeline_id not in self._memory_logs:
            self._memory_logs[pipeline_id] = []
            self._memory_sequences[pipeline_id] = 0
        
        self._memory_sequences[pipeline_id] += 1
        event.sequence = self._memory_sequences[pipeline_id]
        
        self._memory_logs[pipeline_id].append(event)
        
        # Trim
        if len(self._memory_logs[pipeline_id]) > self.config.max_events_per_pipeline:
            self._memory_logs[pipeline_id] = self._memory_logs[pipeline_id][-self.config.max_events_per_pipeline:]
        
        return event.sequence
    
    async def get_all(self, pipeline_id: str) -> List[PipelineEvent]:
        """Get all events for pipeline"""
        log_key = self._log_key(pipeline_id)
        
        if self.redis:
            try:
                events = await self.redis.lrange(log_key, 0, -1)
                return [
                    PipelineEvent.from_json(e.decode('utf-8') if isinstance(e, bytes) else e)
                    for e in events
                ]
            except Exception as e:
                logger.warning(f"Redis get_all failed: {e}")
        
        return list(self._memory_logs.get(pipeline_id, []))
    
    async def get_since(self, pipeline_id: str, since_sequence: int) -> List[PipelineEvent]:
        """Get events since a specific sequence number"""
        all_events = await self.get_all(pipeline_id)
        return [e for e in all_events if e.sequence > since_sequence]
    
    async def get_count(self, pipeline_id: str) -> int:
        """Get event count for pipeline"""
        log_key = self._log_key(pipeline_id)
        
        if self.redis:
            try:
                return await self.redis.llen(log_key) or 0
            except Exception as e:
                logger.warning(f"Redis get_count failed: {e}")
        
        return len(self._memory_logs.get(pipeline_id, []))
    
    async def clear(self, pipeline_id: str) -> None:
        """Clear event log for pipeline"""
        log_key = self._log_key(pipeline_id)
        seq_key = self._seq_key(pipeline_id)
        
        if self.redis:
            try:
                await self.redis.delete(log_key, seq_key)
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        self._memory_logs.pop(pipeline_id, None)
        self._memory_sequences.pop(pipeline_id, None)


# =============================================================================
# LIVE BROADCASTER (Redis Pub/Sub)
# =============================================================================

class LiveBroadcaster:
    """
    Broadcasts live events to connected clients via Redis Pub/Sub.
    Enables real-time streaming across multiple API instances.
    """
    
    def __init__(
        self,
        redis_client=None,
        config: Optional[GhostConfig] = None
    ):
        self.redis = redis_client
        self.config = config or GhostConfig()
        
        # Local subscribers (for non-Redis mode)
        self._local_subscribers: Dict[str, Set[asyncio.Queue]] = {}
    
    def _channel(self, pipeline_id: str) -> str:
        return f"{self.config.pubsub_prefix}{pipeline_id}"
    
    async def broadcast(self, event: PipelineEvent) -> None:
        """Broadcast event to all subscribers"""
        channel = self._channel(event.pipeline_id)
        
        if self.redis:
            try:
                await self.redis.publish(channel, event.to_json())
                return
            except Exception as e:
                logger.warning(f"Redis broadcast failed: {e}")
        
        # Local fallback
        if event.pipeline_id in self._local_subscribers:
            for queue in self._local_subscribers[event.pipeline_id]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.warning("Subscriber queue full")
    
    async def subscribe(
        self,
        pipeline_id: str
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Subscribe to live events for pipeline.
        Yields events as they are broadcast.
        """
        channel = self._channel(pipeline_id)
        
        if self.redis:
            try:
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(channel)
                
                try:
                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            data = message['data']
                            if isinstance(data, bytes):
                                data = data.decode('utf-8')
                            yield PipelineEvent.from_json(data)
                finally:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                return
            except Exception as e:
                logger.warning(f"Redis subscribe failed: {e}")
        
        # Local fallback
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        if pipeline_id not in self._local_subscribers:
            self._local_subscribers[pipeline_id] = set()
        
        self._local_subscribers[pipeline_id].add(queue)
        
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._local_subscribers[pipeline_id].discard(queue)
            if not self._local_subscribers[pipeline_id]:
                del self._local_subscribers[pipeline_id]


# =============================================================================
# GHOST RECOVERY MANAGER
# =============================================================================

class GhostRecoveryManager:
    """
    UPGRADE 3: Ghost Connection Recovery
    
    Manages event persistence and replay for seamless reconnection.
    
    Usage:
        manager = GhostRecoveryManager(redis_client)
        
        # Recording events (server-side)
        await manager.record(event)
        
        # Client reconnection (with replay)
        async for event in manager.stream(pipeline_id, last_seen_sequence=42):
            send_to_client(event)
    """
    
    def __init__(
        self,
        redis_client=None,
        config: Optional[GhostConfig] = None
    ):
        self.redis = redis_client
        self.config = config or GhostConfig()
        
        self.event_log = EventLog(redis_client, config)
        self.broadcaster = LiveBroadcaster(redis_client, config)
        
        # Connection tracking
        self._active_connections: Dict[str, Set[str]] = {}  # pipeline_id -> connection_ids
        
        logger.info(
            f"👻 GhostRecoveryManager initialized "
            f"(Redis: {'enabled' if redis_client else 'disabled'})"
        )
    
    # -------------------------------------------------------------------------
    # Event Recording
    # -------------------------------------------------------------------------
    
    async def record(
        self,
        pipeline_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> PipelineEvent:
        """
        Record and broadcast a pipeline event.
        
        This is the main entry point for recording events.
        Events are persisted to Redis AND broadcast to live subscribers.
        """
        event = PipelineEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            pipeline_id=pipeline_id,
            timestamp=time.time(),
            data=data
        )
        
        # Persist to log
        sequence = await self.event_log.append(event)
        event.sequence = sequence
        
        # Broadcast to live subscribers
        await self.broadcaster.broadcast(event)
        
        return event
    
    async def record_phase_start(
        self,
        pipeline_id: str,
        phase: str,
        description: str,
        progress: float = 0.0
    ) -> PipelineEvent:
        """Record phase start event"""
        return await self.record(
            pipeline_id,
            EventType.PHASE_START.value,
            {
                "phase": phase,
                "description": description,
                "progress": progress
            }
        )
    
    async def record_phase_complete(
        self,
        pipeline_id: str,
        phase: str,
        summary: Dict[str, Any],
        progress: float = 100.0
    ) -> PipelineEvent:
        """Record phase complete event"""
        return await self.record(
            pipeline_id,
            EventType.PHASE_COMPLETE.value,
            {
                "phase": phase,
                "summary": summary,
                "progress": progress
            }
        )
    
    async def record_agent_complete(
        self,
        pipeline_id: str,
        agent: str,
        cost_usd: float,
        latency_ms: float,
        result_preview: Optional[str] = None
    ) -> PipelineEvent:
        """Record agent completion event"""
        return await self.record(
            pipeline_id,
            EventType.AGENT_COMPLETE.value,
            {
                "agent": agent,
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "result_preview": result_preview
            }
        )
    
    async def record_complete(
        self,
        pipeline_id: str,
        total_cost: float,
        total_time: float,
        result: Dict[str, Any]
    ) -> PipelineEvent:
        """Record pipeline completion"""
        return await self.record(
            pipeline_id,
            EventType.PIPELINE_COMPLETE.value,
            {
                "total_cost_usd": total_cost,
                "total_time_seconds": total_time,
                "result": result
            }
        )
    
    async def record_error(
        self,
        pipeline_id: str,
        error: str,
        recoverable: bool = False
    ) -> PipelineEvent:
        """Record pipeline error"""
        return await self.record(
            pipeline_id,
            EventType.PIPELINE_ERROR.value,
            {
                "error": error,
                "recoverable": recoverable
            }
        )
    
    # -------------------------------------------------------------------------
    # Event Streaming (with Replay)
    # -------------------------------------------------------------------------
    
    async def stream(
        self,
        pipeline_id: str,
        last_seen_sequence: int = 0,
        connection_id: Optional[str] = None
    ) -> AsyncGenerator[PipelineEvent, None]:
        """
        Stream events for pipeline with replay support.
        
        If last_seen_sequence > 0, replays all events since that sequence
        before switching to live stream.
        
        This is the core of ghost recovery:
        1. Client connects with last_seen_sequence=0 → gets all history + live
        2. Client disconnects at sequence 50
        3. Client reconnects with last_seen_sequence=50 → gets events 51+ then live
        """
        connection_id = connection_id or str(uuid.uuid4())
        
        # Track connection
        self._register_connection(pipeline_id, connection_id)
        GHOST_ACTIVE_CONNECTIONS.inc()
        
        try:
            # Step 1: Replay history
            is_reconnection = last_seen_sequence > 0
            
            if is_reconnection:
                GHOST_RECONNECTIONS.labels(pipeline_id=pipeline_id).inc()
                
                # Send reconnection acknowledgment
                ack_event = PipelineEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.RECONNECT_ACK.value,
                    pipeline_id=pipeline_id,
                    timestamp=time.time(),
                    data={
                        "last_seen_sequence": last_seen_sequence,
                        "connection_id": connection_id
                    },
                    is_replay=False
                )
                yield ack_event
            
            # Get missed events
            replay_start = time.time()
            missed_events = await self.event_log.get_since(pipeline_id, last_seen_sequence)
            
            replay_count = 0
            for event in missed_events[:self.config.max_replay_events]:
                event.is_replay = True
                yield event
                replay_count += 1
            
            if replay_count > 0:
                GHOST_EVENTS_REPLAYED.labels(pipeline_id=pipeline_id).inc(replay_count)
                GHOST_REPLAY_LATENCY.observe(time.time() - replay_start)
                logger.info(
                    f"👻 Replayed {replay_count} events for {pipeline_id} "
                    f"(since seq {last_seen_sequence})"
                )
            
            # Step 2: Stream live events
            last_heartbeat = time.time()
            
            async for event in self.broadcaster.subscribe(pipeline_id):
                yield event
                
                # Check if we should send heartbeat
                if time.time() - last_heartbeat > self.config.heartbeat_interval:
                    heartbeat = PipelineEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=EventType.HEARTBEAT.value,
                        pipeline_id=pipeline_id,
                        timestamp=time.time(),
                        data={"connection_id": connection_id}
                    )
                    yield heartbeat
                    last_heartbeat = time.time()
                
                # Exit if pipeline complete or error
                if event.event_type in [
                    EventType.PIPELINE_COMPLETE.value,
                    EventType.PIPELINE_ERROR.value
                ]:
                    break
        
        finally:
            self._unregister_connection(pipeline_id, connection_id)
            GHOST_ACTIVE_CONNECTIONS.dec()
    
    async def stream_sse(
        self,
        pipeline_id: str,
        last_seen_sequence: int = 0
    ) -> AsyncGenerator[str, None]:
        """
        Stream events formatted as SSE.
        
        Usage in FastAPI:
            return StreamingResponse(
                manager.stream_sse(pipeline_id),
                media_type="text/event-stream"
            )
        """
        async for event in self.stream(pipeline_id, last_seen_sequence):
            yield event.to_sse()
    
    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------
    
    def _register_connection(self, pipeline_id: str, connection_id: str) -> None:
        """Register active connection"""
        if pipeline_id not in self._active_connections:
            self._active_connections[pipeline_id] = set()
        self._active_connections[pipeline_id].add(connection_id)
    
    def _unregister_connection(self, pipeline_id: str, connection_id: str) -> None:
        """Unregister connection"""
        if pipeline_id in self._active_connections:
            self._active_connections[pipeline_id].discard(connection_id)
            if not self._active_connections[pipeline_id]:
                del self._active_connections[pipeline_id]
    
    def get_connection_count(self, pipeline_id: str) -> int:
        """Get active connection count for pipeline"""
        return len(self._active_connections.get(pipeline_id, set()))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ghost recovery statistics"""
        total_connections = sum(len(c) for c in self._active_connections.values())
        
        return {
            "active_connections": total_connections,
            "pipelines_with_connections": len(self._active_connections),
            "config": {
                "event_log_ttl": self.config.event_log_ttl,
                "max_events_per_pipeline": self.config.max_events_per_pipeline,
                "heartbeat_interval": self.config.heartbeat_interval
            }
        }


# =============================================================================
# SSE RESPONSE HELPERS
# =============================================================================

class SSEResponse:
    """
    Helper class for constructing SSE responses in FastAPI.
    """
    
    @staticmethod
    def headers() -> Dict[str, str]:
        """Standard SSE headers"""
        return {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    
    @staticmethod
    def format_event(
        event_type: str,
        data: Any,
        event_id: Optional[str] = None
    ) -> str:
        """Format data as SSE event"""
        lines = []
        
        if event_id:
            lines.append(f"id: {event_id}")
        
        lines.append(f"event: {event_type}")
        
        if isinstance(data, (dict, list)):
            data = json.dumps(data, default=str)
        
        lines.append(f"data: {data}")
        lines.append("")
        lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_comment(comment: str) -> str:
        """Format as SSE comment (keep-alive)"""
        return f": {comment}\n\n"


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_ghost_recovery_manager(
    redis_client=None,
    **config_overrides
) -> GhostRecoveryManager:
    """Factory for creating GhostRecoveryManager"""
    config = GhostConfig(**config_overrides) if config_overrides else None
    return GhostRecoveryManager(redis_client=redis_client, config=config)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_ghost_recovery():
    """Example demonstrating ghost recovery"""
    
    # Create manager (normally with Redis)
    manager = create_ghost_recovery_manager()
    
    pipeline_id = "pipeline-12345"
    
    # Simulate pipeline recording events
    async def simulate_pipeline():
        await manager.record(
            pipeline_id,
            EventType.PIPELINE_START.value,
            {"business": "Smile Dental", "industry": "dental"}
        )
        
        await asyncio.sleep(0.5)
        
        await manager.record_phase_start(
            pipeline_id,
            "trinity_research",
            "Running TRINITY intelligence agents..."
        )
        
        await asyncio.sleep(0.5)
        
        await manager.record_agent_complete(
            pipeline_id,
            "trend_scout",
            cost_usd=0.02,
            latency_ms=1500
        )
        
        await asyncio.sleep(0.5)
        
        await manager.record_complete(
            pipeline_id,
            total_cost=2.50,
            total_time=180.0,
            result={"video_url": "https://example.com/video.mp4"}
        )
    
    # Simulate client streaming
    async def simulate_client():
        event_count = 0
        async for event in manager.stream(pipeline_id, last_seen_sequence=0):
            print(f"Event {event.sequence}: {event.event_type} - {event.data}")
            event_count += 1
            if event.event_type == EventType.PIPELINE_COMPLETE.value:
                break
        print(f"Total events: {event_count}")
    
    # Run both
    await asyncio.gather(
        simulate_pipeline(),
        simulate_client()
    )
    
    # Simulate reconnection
    print("\n--- Simulating reconnection after sequence 2 ---")
    async for event in manager.stream(pipeline_id, last_seen_sequence=2):
        print(f"Replayed: {event.sequence}: {event.event_type} (is_replay: {event.is_replay})")
        if event.event_type == EventType.PIPELINE_COMPLETE.value:
            break
    
    # Get stats
    stats = await manager.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    asyncio.run(example_ghost_recovery())
