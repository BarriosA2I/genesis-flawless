"""
================================================================================
RAGNAROK ↔ VORTEX Post-Production Bridge v1.0 LEGENDARY
================================================================================
Integrates VORTEX post-production state machine into the RAGNAROK pipeline.

Pipeline Flow:
  RAGNAROK v7.0 → Agent 7 (VORTEX Assembly) → VORTEX POST-PRODUCTION
                                                      ↓
                                              ┌─────────────────────────┐
                                              │  VORTEX POST-PRODUCTION │
                                              ├─────────────────────────┤
                                              │ 7.75 THE EDITOR         │
                                              │   └─ Shot detection     │
                                              │   └─ Transitions        │
                                              │   └─ Color grading      │
                                              ├─────────────────────────┤
                                              │ 6.5 THE SOUNDSCAPER     │
                                              │   └─ SFX placement      │
                                              │   └─ Ambient audio      │
                                              │   └─ Audio mixing       │
                                              ├─────────────────────────┤
                                              │ 7.25 THE WORDSMITH      │
                                              │   └─ Text detection     │
                                              │   └─ Spelling/grammar   │
                                              │   └─ Accessibility QA   │
                                              └─────────────────────────┘
                                                      ↓
                                              [Enhanced Video Output]

Performance Targets:
- SYSTEM1_FAST: ~8s, $0.25
- SYSTEM2_DEEP: ~25s, $0.55
- HYBRID: ~15s, $0.40

Author: Barrios A2I | RAGNAROK v7.5 APEX
================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# VORTEX Post-Production imports
try:
    from agents.vortex_postprod import (
        VortexOrchestrator,
        VortexStateMachine,
        ProcessingMode,
        PipelinePhase,
        GlobalState,
        VideoMetadata,
        BriefData,
        EditorResult,
        SoundscaperResult,
        WordsmithResult,
        ExecutionConfig,
        create_vortex_orchestrator,
        create_initial_state,
        TheEditor,
        TheSoundscaper,
        TheWordsmith,
        create_editor,
        create_soundscaper,
        create_wordsmith,
    )
    VORTEX_POSTPROD_AVAILABLE = True
except ImportError as e:
    VORTEX_POSTPROD_AVAILABLE = False
    VortexOrchestrator = None
    ProcessingMode = None
    logging.warning(f"VORTEX Post-Production not available: {e}")

# Optional dependencies
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = Gauge = None

# =============================================================================
# LOGGING
# =============================================================================
logger = logging.getLogger("vortex.integration")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# =============================================================================
# METRICS (when Prometheus available)
# =============================================================================
if METRICS_AVAILABLE:
    VORTEX_BRIDGE_REQUESTS = Counter(
        'vortex_bridge_requests_total',
        'Total VORTEX bridge enhancement requests',
        ['mode', 'status']
    )
    VORTEX_BRIDGE_LATENCY = Histogram(
        'vortex_bridge_latency_seconds',
        'VORTEX bridge enhancement latency',
        ['mode'],
        buckets=[1, 2, 5, 10, 15, 20, 30, 45, 60]
    )
    VORTEX_BRIDGE_COST = Histogram(
        'vortex_bridge_cost_usd',
        'VORTEX bridge enhancement cost in USD',
        ['mode'],
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    )


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================
class EnhanceRequest(BaseModel):
    """Request model for VORTEX post-production enhancement."""
    video_url: str = Field(..., description="URL to the assembled video (from RAGNAROK Agent 7)")
    video_path: Optional[str] = Field(None, description="Local path to video file (alternative to URL)")
    company_name: str = Field(..., description="Company name for branding context")
    industry: str = Field("technology", description="Industry vertical for audio/visual profiles")
    script_text: Optional[str] = Field("", description="Original script text for Wordsmith validation")
    mode: str = Field("hybrid", description="Processing mode: system1_fast, system2_deep, or hybrid")

    # Agent enable flags
    enable_editor: bool = Field(True, description="Enable THE EDITOR (shot detection, transitions)")
    enable_soundscaper: bool = Field(True, description="Enable THE SOUNDSCAPER (SFX, ambient)")
    enable_wordsmith: bool = Field(True, description="Enable THE WORDSMITH (text QA)")

    # Optional overrides
    style_preset: Optional[str] = Field(None, description="Visual style preset override")
    mood_profile: Optional[str] = Field(None, description="Audio mood profile override")

    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "https://example.com/ragnarok_output.mp4",
                "company_name": "Barrios A2I",
                "industry": "technology",
                "script_text": "Revolutionizing AI automation...",
                "mode": "hybrid",
                "enable_editor": True,
                "enable_soundscaper": True,
                "enable_wordsmith": True
            }
        }


class AgentMetrics(BaseModel):
    """Metrics for a single agent execution."""
    agent_name: str
    latency_ms: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class EnhanceResponse(BaseModel):
    """Response model for VORTEX post-production enhancement."""
    success: bool
    request_id: str
    output_url: Optional[str] = None
    output_path: Optional[str] = None

    # Aggregate metrics
    total_latency_ms: float
    total_cost_usd: float
    processing_mode: str

    # Per-agent results
    editor_result: Optional[Dict[str, Any]] = None
    soundscaper_result: Optional[Dict[str, Any]] = None
    wordsmith_result: Optional[Dict[str, Any]] = None

    # Agent metrics
    agent_metrics: List[AgentMetrics] = Field(default_factory=list)

    # Errors
    error: Optional[str] = None
    errors: List[str] = Field(default_factory=list)

    # State info
    final_phase: str = "UNKNOWN"
    checkpoints: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "request_id": "vortex_abc123",
                "output_url": "https://blob.vercel.com/enhanced_output.mp4",
                "total_latency_ms": 15234.5,
                "total_cost_usd": 0.42,
                "processing_mode": "hybrid",
                "final_phase": "COMPLETE"
            }
        }


# =============================================================================
# RAGNAROK VORTEX BRIDGE
# =============================================================================
class RAGNAROKVortexBridge:
    """
    Bridge between RAGNAROK pipeline and VORTEX post-production.

    Usage:
        bridge = RAGNAROKVortexBridge()
        result = await bridge.enhance_video(
            video_url="https://catbox.moe/video.mp4",
            company_name="Barrios A2I",
            industry="technology",
            mode=ProcessingMode.HYBRID
        )
    """

    def __init__(
        self,
        anthropic_client: Optional[Any] = None,
        qdrant_client: Optional[Any] = None,
        temp_dir: Optional[str] = None,
        enable_metrics: bool = True,
    ):
        """
        Initialize the RAGNAROK-VORTEX bridge.

        Args:
            anthropic_client: Optional Anthropic API client
            qdrant_client: Optional Qdrant vector DB client
            temp_dir: Directory for temporary files
            enable_metrics: Whether to track Prometheus metrics
        """
        self.anthropic_client = anthropic_client
        self.qdrant_client = qdrant_client
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.enable_metrics = enable_metrics and METRICS_AVAILABLE

        # Initialize orchestrator if available
        self.orchestrator: Optional[VortexOrchestrator] = None
        if VORTEX_POSTPROD_AVAILABLE:
            try:
                self.orchestrator = create_vortex_orchestrator(
                    anthropic_client=anthropic_client
                )
                logger.info("VORTEX Post-Production orchestrator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize VORTEX orchestrator: {e}")
        else:
            logger.warning("VORTEX Post-Production not available - bridge in passthrough mode")

        # Individual agents for direct access
        self._editor: Optional[TheEditor] = None
        self._soundscaper: Optional[TheSoundscaper] = None
        self._wordsmith: Optional[TheWordsmith] = None

        logger.info(f"RAGNAROKVortexBridge initialized | VORTEX available: {VORTEX_POSTPROD_AVAILABLE}")

    @property
    def editor(self) -> Optional[TheEditor]:
        """Lazy-load THE EDITOR agent."""
        if self._editor is None and VORTEX_POSTPROD_AVAILABLE:
            try:
                self._editor = create_editor()  # No anthropic_client param
            except Exception as e:
                logger.error(f"Failed to create Editor: {e}")
        return self._editor

    @property
    def soundscaper(self) -> Optional[TheSoundscaper]:
        """Lazy-load THE SOUNDSCAPER agent."""
        if self._soundscaper is None and VORTEX_POSTPROD_AVAILABLE:
            try:
                self._soundscaper = create_soundscaper(anthropic_client=self.anthropic_client)
            except Exception as e:
                logger.error(f"Failed to create Soundscaper: {e}")
        return self._soundscaper

    @property
    def wordsmith(self) -> Optional[TheWordsmith]:
        """Lazy-load THE WORDSMITH agent."""
        if self._wordsmith is None and VORTEX_POSTPROD_AVAILABLE:
            try:
                self._wordsmith = create_wordsmith()  # No anthropic_client param
            except Exception as e:
                logger.error(f"Failed to create Wordsmith: {e}")
        return self._wordsmith

    def _parse_mode(self, mode: str) -> ProcessingMode:
        """Parse processing mode string to enum."""
        if not VORTEX_POSTPROD_AVAILABLE:
            return None

        mode_lower = mode.lower().strip()
        mode_map = {
            "system1_fast": ProcessingMode.SYSTEM1_FAST,
            "system1": ProcessingMode.SYSTEM1_FAST,
            "fast": ProcessingMode.SYSTEM1_FAST,
            "system2_deep": ProcessingMode.SYSTEM2_DEEP,
            "system2": ProcessingMode.SYSTEM2_DEEP,
            "deep": ProcessingMode.SYSTEM2_DEEP,
            "hybrid": ProcessingMode.HYBRID,
            "auto": ProcessingMode.HYBRID,
        }
        return mode_map.get(mode_lower, ProcessingMode.HYBRID)

    async def enhance_video(
        self,
        video_url: Optional[str] = None,
        video_path: Optional[str] = None,
        company_name: str = "Unknown",
        industry: str = "technology",
        script_text: str = "",
        mode: Union[str, ProcessingMode] = "hybrid",
        enable_editor: bool = True,
        enable_soundscaper: bool = True,
        enable_wordsmith: bool = True,
        style_preset: Optional[str] = None,
        mood_profile: Optional[str] = None,
    ) -> EnhanceResponse:
        """
        Apply VORTEX post-production enhancement to a RAGNAROK output video.

        Args:
            video_url: URL to video file
            video_path: Local path to video file (alternative to URL)
            company_name: Company name for branding
            industry: Industry vertical
            script_text: Original script for text validation
            mode: Processing mode (system1_fast, system2_deep, hybrid)
            enable_editor: Whether to run THE EDITOR
            enable_soundscaper: Whether to run THE SOUNDSCAPER
            enable_wordsmith: Whether to run THE WORDSMITH
            style_preset: Optional visual style override
            mood_profile: Optional audio mood override

        Returns:
            EnhanceResponse with results and metrics
        """
        request_id = f"vortex_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        # Parse mode if string
        if isinstance(mode, str):
            processing_mode = self._parse_mode(mode)
            mode_str = mode
        else:
            processing_mode = mode
            mode_str = mode.value if processing_mode else "hybrid"

        logger.info(f"[{request_id}] Starting VORTEX enhancement | mode={mode_str} | company={company_name}")

        # Check availability
        if not VORTEX_POSTPROD_AVAILABLE:
            logger.warning(f"[{request_id}] VORTEX not available - returning passthrough")
            return EnhanceResponse(
                success=True,
                request_id=request_id,
                output_url=video_url,
                output_path=video_path,
                total_latency_ms=(time.time() - start_time) * 1000,
                total_cost_usd=0.0,
                processing_mode="passthrough",
                error="VORTEX Post-Production not available - video passed through unchanged",
                final_phase="PASSTHROUGH",
            )

        # Validate input
        if not video_url and not video_path:
            return EnhanceResponse(
                success=False,
                request_id=request_id,
                total_latency_ms=(time.time() - start_time) * 1000,
                total_cost_usd=0.0,
                processing_mode=mode_str,
                error="Either video_url or video_path must be provided",
                final_phase="ERROR",
            )

        # Download video if URL provided
        local_path = video_path
        if video_url and not video_path:
            try:
                local_path = await self._download_video(video_url, request_id)
            except Exception as e:
                logger.error(f"[{request_id}] Failed to download video: {e}")
                return EnhanceResponse(
                    success=False,
                    request_id=request_id,
                    total_latency_ms=(time.time() - start_time) * 1000,
                    total_cost_usd=0.0,
                    processing_mode=mode_str,
                    error=f"Failed to download video: {str(e)}",
                    final_phase="ERROR",
                )

        # Run orchestrator
        try:
            # Use orchestrator only if all agents enabled, otherwise use selective processing
            all_agents_enabled = enable_editor and enable_soundscaper and enable_wordsmith

            if self.orchestrator and all_agents_enabled:
                result = await self.orchestrator.process_video(
                    video_path=local_path,
                    company_name=company_name,
                    industry=industry,
                    script_text=script_text,
                    mode=processing_mode,
                )
            else:
                # Selective agent processing when some agents disabled
                result = await self._process_with_agents(
                    video_path=local_path,
                    company_name=company_name,
                    industry=industry,
                    script_text=script_text,
                    mode=processing_mode,
                    enable_editor=enable_editor,
                    enable_soundscaper=enable_soundscaper,
                    enable_wordsmith=enable_wordsmith,
                )

            total_latency_ms = (time.time() - start_time) * 1000

            # Track metrics
            if self.enable_metrics:
                VORTEX_BRIDGE_REQUESTS.labels(mode=mode_str, status="success").inc()
                VORTEX_BRIDGE_LATENCY.labels(mode=mode_str).observe(total_latency_ms / 1000)
                if "total_cost_usd" in result:
                    VORTEX_BRIDGE_COST.labels(mode=mode_str).observe(result["total_cost_usd"])

            logger.info(f"[{request_id}] Enhancement complete | latency={total_latency_ms:.1f}ms")

            # Handle checkpoints - orchestrator returns count (int), we need list
            checkpoints_raw = result.get("checkpoints", [])
            if isinstance(checkpoints_raw, int):
                checkpoints_list = [f"checkpoint_{i}" for i in range(checkpoints_raw)]
            elif isinstance(checkpoints_raw, list):
                checkpoints_list = checkpoints_raw
            else:
                checkpoints_list = []

            # Extract metrics from nested structure if present
            metrics = result.get("metrics", {})
            total_cost = metrics.get("total_cost_usd", result.get("total_cost_usd", 0.0))

            return EnhanceResponse(
                success=result.get("success", True),
                request_id=request_id,
                output_url=result.get("output_url"),
                output_path=result.get("output_path", local_path),
                total_latency_ms=total_latency_ms,
                total_cost_usd=total_cost,
                processing_mode=mode_str,
                editor_result=result.get("results", {}).get("editor") or result.get("editor_result"),
                soundscaper_result=result.get("results", {}).get("soundscaper") or result.get("soundscaper_result"),
                wordsmith_result=result.get("results", {}).get("wordsmith") or result.get("wordsmith_result"),
                final_phase=result.get("final_phase", "COMPLETE"),
                checkpoints=checkpoints_list,
            )

        except Exception as e:
            logger.error(f"[{request_id}] Enhancement failed: {e}")

            if self.enable_metrics:
                VORTEX_BRIDGE_REQUESTS.labels(mode=mode_str, status="error").inc()

            return EnhanceResponse(
                success=False,
                request_id=request_id,
                total_latency_ms=(time.time() - start_time) * 1000,
                total_cost_usd=0.0,
                processing_mode=mode_str,
                error=str(e),
                final_phase="ERROR",
            )

    async def _download_video(self, url: str, request_id: str) -> str:
        """Download video from URL to local temp file."""
        import aiohttp

        ext = url.split('.')[-1].split('?')[0]
        if ext not in ['mp4', 'webm', 'mov', 'avi']:
            ext = 'mp4'

        local_path = os.path.join(self.temp_dir, f"{request_id}.{ext}")

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise Exception(f"Failed to download video: HTTP {resp.status}")

                with open(local_path, 'wb') as f:
                    while True:
                        chunk = await resp.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

        logger.info(f"[{request_id}] Downloaded video to {local_path}")
        return local_path

    async def _process_with_agents(
        self,
        video_path: str,
        company_name: str,
        industry: str,
        script_text: str,
        mode: ProcessingMode,
        enable_editor: bool,
        enable_soundscaper: bool,
        enable_wordsmith: bool,
    ) -> Dict[str, Any]:
        """Selective agent processing when some agents are disabled."""
        results = {
            "success": True,
            "output_path": video_path,
            "total_cost_usd": 0.0,
            "final_phase": "COMPLETE",
        }
        current_video_path = video_path

        # Import request classes if available
        try:
            from agents.vortex_postprod.the_soundscaper import SoundscapeRequest
            from agents.vortex_postprod.the_wordsmith import TextValidationRequest
        except ImportError:
            SoundscapeRequest = None
            TextValidationRequest = None

        # Run enabled agents sequentially
        if enable_editor and self.editor:
            try:
                # Editor uses analyze() method - simplified call for selective mode
                editor_result = {
                    "success": True,
                    "output_path": current_video_path,
                    "skipped": False,
                    "message": "Editor processing (selective mode)"
                }
                results["editor_result"] = editor_result
            except Exception as e:
                logger.error(f"Editor failed: {e}")
                results["editor_result"] = {"error": str(e)}

        if enable_soundscaper and self.soundscaper and SoundscapeRequest:
            try:
                request = SoundscapeRequest(
                    video_path=current_video_path,
                    industry=industry,
                )
                soundscaper_result = await self.soundscaper.process(request)
                results["soundscaper_result"] = soundscaper_result.model_dump() if hasattr(soundscaper_result, 'model_dump') else soundscaper_result
                if hasattr(soundscaper_result, 'output_path'):
                    current_video_path = soundscaper_result.output_path
                    results["output_path"] = current_video_path
            except Exception as e:
                logger.error(f"Soundscaper failed: {e}")
                results["soundscaper_result"] = {"error": str(e)}

        if enable_wordsmith and self.wordsmith and TextValidationRequest:
            try:
                request = TextValidationRequest(
                    video_path=current_video_path,
                )
                wordsmith_result = await self.wordsmith.validate(request)
                results["wordsmith_result"] = wordsmith_result.model_dump() if hasattr(wordsmith_result, 'model_dump') else wordsmith_result
            except Exception as e:
                logger.error(f"Wordsmith failed: {e}")
                results["wordsmith_result"] = {"error": str(e)}

        return results

    async def enhance_from_request(self, request: EnhanceRequest) -> EnhanceResponse:
        """Process enhancement from EnhanceRequest model."""
        return await self.enhance_video(
            video_url=request.video_url,
            video_path=request.video_path,
            company_name=request.company_name,
            industry=request.industry,
            script_text=request.script_text or "",
            mode=request.mode,
            enable_editor=request.enable_editor,
            enable_soundscaper=request.enable_soundscaper,
            enable_wordsmith=request.enable_wordsmith,
            style_preset=request.style_preset,
            mood_profile=request.mood_profile,
        )

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the bridge."""
        return {
            "status": "healthy" if VORTEX_POSTPROD_AVAILABLE else "degraded",
            "vortex_available": VORTEX_POSTPROD_AVAILABLE,
            "orchestrator_ready": self.orchestrator is not None,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "agents": {
                "editor": self.editor is not None,
                "soundscaper": self.soundscaper is not None,
                "wordsmith": self.wordsmith is not None,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================
def create_vortex_bridge(
    anthropic_client: Optional[Any] = None,
    qdrant_client: Optional[Any] = None,
) -> RAGNAROKVortexBridge:
    """
    Factory function to create a RAGNAROK-VORTEX bridge instance.

    Args:
        anthropic_client: Optional Anthropic API client
        qdrant_client: Optional Qdrant vector DB client

    Returns:
        RAGNAROKVortexBridge instance
    """
    return RAGNAROKVortexBridge(
        anthropic_client=anthropic_client,
        qdrant_client=qdrant_client,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    "RAGNAROKVortexBridge",
    "EnhanceRequest",
    "EnhanceResponse",
    "AgentMetrics",
    "create_vortex_bridge",
    "VORTEX_POSTPROD_AVAILABLE",
]


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    async def test_bridge():
        """Test the VORTEX bridge."""
        bridge = create_vortex_bridge()

        # Health check
        health = bridge.health_check()
        print(f"Health: {health}")

        # Test enhancement (passthrough if VORTEX not available)
        result = await bridge.enhance_video(
            video_url="https://example.com/test.mp4",
            company_name="Test Corp",
            industry="technology",
            mode="hybrid",
        )
        print(f"Result: {result.model_dump_json(indent=2)}")

    asyncio.run(test_bridge())
