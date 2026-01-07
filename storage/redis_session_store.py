"""Redis-backed session storage for Creative Director.

Enables session persistence across multiple workers/processes.
Uses JSON serialization for CreativeDirectorSession objects.

Version: 1.0.0
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Session TTL: 24 hours
SESSION_TTL_SECONDS = 86400


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SessionData:
    """Serializable session data structure."""
    session_id: str
    client_name: str
    project_type: str
    phase: str = "intake"
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "client_name": self.client_name,
            "project_type": self.project_type,
            "phase": self.phase,
            "context": self.context,
            "history": self.history,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            client_name=data.get("client_name", ""),
            project_type=data.get("project_type", "brand_identity"),
            phase=data.get("phase", "intake"),
            context=data.get("context", {}),
            history=data.get("history", []),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat())
        )


class RedisSessionStore:
    """Redis-backed session storage with automatic serialization."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self._client = None
        self._prefix = "cd:session:"  # creative-director:session:
        self._available = False
        self._init_client()

    def _init_client(self):
        """Initialize Redis client."""
        try:
            import redis
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self._client.ping()
            self._available = True
            logger.info(f"Redis session store connected: {self._mask_url(self.redis_url)}")
        except ImportError:
            logger.warning("redis package not installed, using in-memory fallback")
            self._available = False
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory fallback")
            self._available = False

    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of Redis URL for logging."""
        if "@" in url:
            return url.split("@")[-1]
        return url

    @property
    def client(self):
        """Get Redis client."""
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._available and self._client is not None

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self._prefix}{session_id}"

    def save(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Save session to Redis with TTL."""
        if not self.is_available:
            return False

        try:
            key = self._key(session_id)
            # Update timestamp
            session_data["updated_at"] = datetime.utcnow().isoformat()

            self._client.setex(
                key,
                SESSION_TTL_SECONDS,
                json.dumps(session_data, default=str)
            )
            logger.debug(f"Session {session_id[:8]}... saved to Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to save session {session_id[:8]}...: {e}")
            return False

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from Redis."""
        if not self.is_available:
            return None

        try:
            key = self._key(session_id)
            data = self._client.get(key)

            if data:
                logger.debug(f"Session {session_id[:8]}... loaded from Redis")
                return json.loads(data)

            logger.debug(f"Session {session_id[:8]}... not found in Redis")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id[:8]}...: {e}")
            return None

    def delete(self, session_id: str) -> bool:
        """Delete session from Redis."""
        if not self.is_available:
            return False

        try:
            key = self._key(session_id)
            self._client.delete(key)
            logger.debug(f"Session {session_id[:8]}... deleted from Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id[:8]}...: {e}")
            return False

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        if not self.is_available:
            return False

        try:
            return self._client.exists(self._key(session_id)) > 0
        except Exception as e:
            logger.error(f"Failed to check session {session_id[:8]}...: {e}")
            return False

    def extend_ttl(self, session_id: str) -> bool:
        """Extend session TTL on activity."""
        if not self.is_available:
            return False

        try:
            key = self._key(session_id)
            return bool(self._client.expire(key, SESSION_TTL_SECONDS))
        except Exception as e:
            logger.error(f"Failed to extend TTL for {session_id[:8]}...: {e}")
            return False

    def list_sessions(self, limit: int = 100) -> List[str]:
        """List all session IDs."""
        if not self.is_available:
            return []

        try:
            keys = self._client.keys(f"{self._prefix}*")
            session_ids = [k.replace(self._prefix, "") for k in keys[:limit]]
            return session_ids
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        if not self.is_available:
            return {"status": "unavailable", "backend": "none"}

        try:
            self._client.ping()
            session_count = len(self._client.keys(f"{self._prefix}*"))
            return {
                "status": "healthy",
                "backend": "redis",
                "url": self._mask_url(self.redis_url),
                "sessions": session_count
            }
        except Exception as e:
            return {"status": "unhealthy", "backend": "redis", "error": str(e)}


class HybridSessionStore:
    """Hybrid store: Redis primary, in-memory fallback.

    Provides seamless fallback to in-memory storage when Redis
    is unavailable, ensuring the application remains functional.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self._redis = RedisSessionStore(redis_url)
        self._memory: Dict[str, Dict[str, Any]] = {}
        self._use_redis = self._redis.is_available

        if self._use_redis:
            logger.info("HybridSessionStore: Using Redis backend")
        else:
            logger.warning("HybridSessionStore: Using in-memory fallback (sessions won't persist across workers)")

    @property
    def backend(self) -> str:
        """Get current backend type."""
        return "redis" if self._use_redis else "memory"

    def save(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Save to Redis or memory."""
        # Always save to memory as local cache
        self._memory[session_id] = session_data.copy()

        # Also save to Redis if available
        if self._use_redis:
            return self._redis.save(session_id, session_data)
        return True

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load from Redis or memory."""
        # Check local cache first (faster)
        if session_id in self._memory:
            logger.debug(f"Session {session_id[:8]}... loaded from local cache")
            return self._memory[session_id]

        # Try Redis
        if self._use_redis:
            data = self._redis.load(session_id)
            if data:
                # Populate local cache
                self._memory[session_id] = data
                return data

        return None

    def delete(self, session_id: str) -> bool:
        """Delete from Redis and memory."""
        self._memory.pop(session_id, None)

        if self._use_redis:
            return self._redis.delete(session_id)
        return True

    def exists(self, session_id: str) -> bool:
        """Check existence in memory or Redis."""
        if session_id in self._memory:
            return True

        if self._use_redis:
            return self._redis.exists(session_id)
        return False

    def extend_ttl(self, session_id: str) -> bool:
        """Extend TTL in Redis."""
        if self._use_redis:
            return self._redis.extend_ttl(session_id)
        return True

    def health_check(self) -> Dict[str, Any]:
        """Return storage health status."""
        if self._use_redis:
            redis_health = self._redis.health_check()
            redis_health["local_cache_size"] = len(self._memory)
            return redis_health

        return {
            "status": "healthy",
            "backend": "in-memory",
            "sessions": len(self._memory),
            "warning": "Sessions will not persist across workers"
        }


# Singleton instance for application-wide use
_session_store: Optional[HybridSessionStore] = None


def get_session_store() -> HybridSessionStore:
    """Get or create the singleton session store."""
    global _session_store
    if _session_store is None:
        _session_store = HybridSessionStore()
    return _session_store


def init_session_store(redis_url: Optional[str] = None) -> HybridSessionStore:
    """Initialize the session store with custom Redis URL."""
    global _session_store
    _session_store = HybridSessionStore(redis_url)
    return _session_store
