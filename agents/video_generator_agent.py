"""
GENESIS Video Generator Agent (Agent 4)
===============================================================================
AI video generation using KIE.ai VEO 3.1 API.

Features:
- KIE.ai VEO 3.1 video generation
- Circuit breaker pattern for fault tolerance
- Async polling for generation status
- Graceful fallback to placeholders
- Cost tracking per scene

Cost:
- VEO 3.1 Fast: $0.40/8s (720p)
- VEO 3.1 Quality: $2.00/8s (1080p)

Author: Barrios A2I
Version: 2.0.0 (KIE.ai Integration)
===============================================================================
"""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("genesis.video_generator")


def debug_print(msg: str):
    """Print debug message that definitely reaches Render logs."""
    print(f"[VIDEO-DEBUG] {msg}", flush=True)
    sys.stdout.flush()


# =============================================================================
# ENUMS
# =============================================================================

class VideoModel(str, Enum):
    """Supported video generation models via KIE.ai"""
    VEO_FAST = "veo3_fast"      # $0.40/8s, 720p
    VEO_QUALITY = "veo3"        # $2.00/8s, 1080p


class GenerationStatus(str, Enum):
    """Video generation status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# =============================================================================
# DATA MODELS
# =============================================================================

class VideoRequest(BaseModel):
    """Video generation request."""
    prompt: str = Field(..., description="Scene description prompt")
    duration: float = Field(default=5.0, ge=1.0, le=60.0, description="Duration in seconds")
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio")
    resolution: str = Field(default="1080p", description="Resolution")
    style: str = Field(default="cinematic", description="Visual style")
    scene_number: int = Field(default=1, description="Scene number")
    model: str = Field(default="auto", description="Model: auto, sora2, veo3.1")


class VideoResult(BaseModel):
    """Video generation result."""
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    status: GenerationStatus
    model_used: Optional[str] = None
    cost_usd: float = 0.0
    generation_time_seconds: float = 0.0
    scene_number: int = 1
    error: Optional[str] = None
    source: str = "kie"  # kie, placeholder, error


class GenerationStats(BaseModel):
    """Agent statistics."""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_cost_usd: float = 0.0
    avg_generation_time: float = 0.0


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

@dataclass
class CircuitBreaker:
    """Simple circuit breaker for API reliability."""
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    failure_threshold: int = 5  # More tolerant of intermittent failures
    timeout_seconds: int = 30  # Recover faster from open state
    last_failure_time: Optional[float] = None
    half_open_successes: int = 0
    half_open_required: int = 2

    def can_attempt(self) -> bool:
        """Check if request can be attempted."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.timeout_seconds:
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                    self.state = CircuitState.HALF_OPEN
                    return True
            return False

        return True  # HALF_OPEN

    def record_success(self):
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_required:
                logger.info(f"Circuit breaker {self.name} closing after recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_successes = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit breaker {self.name} reopening after failure")
            self.state = CircuitState.OPEN
            self.half_open_successes = 0
        elif self.failure_count >= self.failure_threshold:
            logger.error(f"Circuit breaker {self.name} opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN

    def reset(self):
        """Reset circuit breaker to closed state (used after code deploy)."""
        debug_print(f"Circuit breaker {self.name} reset to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_successes = 0
        self.last_failure_time = None


# =============================================================================
# VIDEO GENERATOR AGENT
# =============================================================================

class VideoGeneratorAgent:
    """
    Agent 4: Video Generator using KIE.ai VEO 3.1 API.

    Uses KIE.ai for professional video generation
    with automatic fallback to placeholders.

    Pricing:
    - VEO 3.1 Fast: $0.40 per 8 seconds (720p)
    - VEO 3.1 Quality: $2.00 per 8 seconds (1080p)

    Environment: KIE_API_KEY
    """

    # KIE.ai API endpoints
    BASE_URL = "https://api.kie.ai/api/v1"
    GENERATE_ENDPOINT = f"{BASE_URL}/veo/generate"
    STATUS_ENDPOINT = f"{BASE_URL}/veo/record-info"

    # Model costs via KIE.ai
    COSTS = {
        VideoModel.VEO_FAST: 0.40,
        VideoModel.VEO_QUALITY: 2.00,
    }

    # Routing keywords - Quality tier for complex scenes
    QUALITY_KEYWORDS = [
        "cinematic", "dramatic", "emotional", "people", "human", "character",
        "tracking shot", "dolly", "zoom", "pan", "realistic", "professional",
        "complex", "motion", "action", "narrative", "story", "1080p", "hd"
    ]

    # Fast tier for simpler scenes
    FAST_KEYWORDS = [
        "product", "logo", "text", "graphics", "simple", "static",
        "animation", "2d", "icon", "minimal", "clean", "modern", "quick"
    ]

    def __init__(
        self,
        kie_api_key: Optional[str] = None,
        max_retries: int = 3,
        poll_interval: int = 10,
        max_poll_time: int = 600
    ):
        self.api_key = kie_api_key or os.getenv("KIE_API_KEY")
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time

        # Circuit breaker - reset on startup to give fresh code a chance after deploy
        self.circuit = CircuitBreaker(name="kie")
        self.circuit.reset()

        # Statistics
        self.stats = GenerationStats()

        # Credits tracking - set True when 402 received
        self.credits_exhausted = False
        self.credits_exhausted_time: Optional[float] = None

        # Check if API is configured
        self.is_configured = bool(self.api_key)

        if self.is_configured:
            logger.info("[VideoGenerator] Initialized with KIE.ai API")
        else:
            logger.warning("[VideoGenerator] No KIE_API_KEY - will use placeholders only")

    def _determine_model(self, request: VideoRequest) -> VideoModel:
        """
        Intelligently route to VEO Fast or VEO Quality based on prompt content.

        VEO Quality ($2.00): Better for cinematic, complex, 1080p scenes
        VEO Fast ($0.40): Better for simple scenes, cost-effective 720p
        """
        if request.model == "quality" or request.model == "veo3":
            return VideoModel.VEO_QUALITY
        elif request.model == "fast" or request.model == "veo3_fast":
            return VideoModel.VEO_FAST

        # Auto routing
        prompt_lower = request.prompt.lower()

        quality_score = sum(1 for kw in self.QUALITY_KEYWORDS if kw in prompt_lower)
        fast_score = sum(1 for kw in self.FAST_KEYWORDS if kw in prompt_lower)

        # Long duration favors Quality tier
        if request.duration >= 10:
            quality_score += 1

        # 1080p/4K resolution favors Quality tier
        if request.resolution in ["1080p", "4k"]:
            quality_score += 2

        # Cinematic style favors Quality tier
        if "cinematic" in request.style.lower():
            quality_score += 2

        if quality_score > fast_score + 1:
            logger.info(f"Routing to VEO Quality (score: {quality_score} vs {fast_score})")
            return VideoModel.VEO_QUALITY
        else:
            logger.info(f"Routing to VEO Fast (score: {fast_score} vs {quality_score})")
            return VideoModel.VEO_FAST

    async def generate(self, request: VideoRequest) -> VideoResult:
        """
        Generate video for a single scene.

        Flow:
        1. Determine optimal model (VEO Fast or Quality)
        2. Submit to KIE.ai VEO 3.1 API
        3. Poll for completion
        4. Return result or fallback to placeholder
        """
        start_time = time.time()
        self.stats.total_requests += 1

        # DEBUG: Trace generate() entry
        debug_print(f"generate() called for scene {request.scene_number}")
        debug_print(f"  is_configured: {self.is_configured}")
        debug_print(f"  circuit.state: {self.circuit.state.value}")
        debug_print(f"  circuit.can_attempt(): {self.circuit.can_attempt()}")
        debug_print(f"  credits_exhausted: {self.credits_exhausted}")

        # Check if credits are exhausted - skip API calls entirely
        if self.credits_exhausted:
            debug_print("  -> BYPASS: KIE.ai credits exhausted, using placeholder")
            return await self._generate_placeholder(request, start_time)

        # Check if API is configured
        if not self.is_configured:
            debug_print("  -> BYPASS: Not configured, using placeholder")
            return await self._generate_placeholder(request, start_time)

        # Check circuit breaker
        if not self.circuit.can_attempt():
            debug_print("  -> BYPASS: Circuit breaker blocking, using placeholder")
            return await self._generate_placeholder(request, start_time)

        debug_print("  -> PROCEED: Attempting API call")
        # Determine model
        model = self._determine_model(request)
        cost = self.COSTS[model]

        # Try generation with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"[Scene {request.scene_number}] Attempt {attempt + 1}/{self.max_retries}")

                # Submit generation request
                generation_id = await self._submit_generation(request, model)

                if not generation_id:
                    raise Exception("No generation ID returned")

                # Poll for completion
                video_url = await self._poll_completion(generation_id)

                if not video_url:
                    raise Exception("Generation timed out or failed")

                # Success!
                self.circuit.record_success()
                self.stats.successful += 1
                self.stats.total_cost_usd += cost

                generation_time = time.time() - start_time
                self.stats.avg_generation_time = (
                    (self.stats.avg_generation_time * (self.stats.successful - 1) + generation_time)
                    / self.stats.successful
                ) if self.stats.successful > 0 else generation_time

                logger.info(f"[Scene {request.scene_number}] Generated in {generation_time:.1f}s (${cost:.2f})")

                return VideoResult(
                    video_url=video_url,
                    status=GenerationStatus.COMPLETED,
                    model_used=model.value,
                    cost_usd=cost,
                    generation_time_seconds=generation_time,
                    scene_number=request.scene_number,
                    source="kie"
                )

            except Exception as e:
                logger.warning(f"[Scene {request.scene_number}] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.circuit.record_failure()

        # All retries failed - generate placeholder
        logger.error(f"[Scene {request.scene_number}] All attempts failed - using placeholder")
        self.stats.failed += 1
        return await self._generate_placeholder(request, start_time)

    async def _submit_generation(self, request: VideoRequest, model: VideoModel) -> Optional[str]:
        """Submit generation request to KIE.ai VEO 3.1 API."""
        # Map aspect ratio to KIE.ai format
        aspect_map = {
            "16:9": "16:9",
            "9:16": "9:16",
            "1:1": "1:1",
        }
        aspect_ratio = aspect_map.get(request.aspect_ratio, "16:9")

        payload = {
            "prompt": request.prompt,
            "aspectRatio": aspect_ratio,
            "model": model.value,  # veo3_fast or veo3
            "generationType": "TEXT_2_VIDEO"
        }

        # VERBOSE: Log submission details
        debug_print(f"Submitting to KIE.ai: {self.GENERATE_ENDPOINT}")
        debug_print(f"Payload: model={model.value}, aspectRatio={aspect_ratio}")
        debug_print(f"Prompt: {request.prompt[:100]}...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    self.GENERATE_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                # VERBOSE: Log raw response
                status_code = response.status_code
                text = response.text
                debug_print(f"KIE submit response: status={status_code}")
                debug_print(f"KIE body: {text[:500]}")

                if status_code == 429:
                    raise Exception("KIE.ai rate limited - too many requests")

                # CRITICAL: Detect credits exhausted (402 Payment Required)
                if status_code == 402:
                    error_msg = "KIE.AI CREDITS EXHAUSTED - Please add credits at https://kie.ai"
                    debug_print("=" * 60)
                    debug_print(f"🚨 {error_msg}")
                    debug_print("=" * 60)
                    logger.critical(error_msg)
                    # Set credits exhausted flag for health checks
                    self.credits_exhausted = True
                    self.credits_exhausted_time = time.time()
                    # Mark circuit breaker to prevent repeated failed calls
                    self.circuit.state = CircuitState.OPEN
                    self.circuit.last_failure_time = time.time()
                    raise Exception(error_msg)

                if status_code != 200:
                    raise Exception(f"KIE.ai generation failed: {status_code} - {text}")

                data = response.json()

                # Check KIE.ai response format
                if data.get("code") != 200:
                    error_msg = data.get("msg", "Unknown error")
                    raise Exception(f"KIE.ai error: {error_msg}")

                # Extract task ID from response
                task_id = data.get("data", {}).get("taskId")
                if not task_id:
                    raise Exception(f"No taskId in KIE.ai response: {data}")

                debug_print(f"KIE task started: {task_id}")
                return task_id

            except Exception as e:
                debug_print(f"KIE submit exception: {type(e).__name__}: {e}")
                raise

    async def _poll_completion(self, task_id: str) -> Optional[str]:
        """Poll KIE.ai for task completion and return video URL."""
        if not task_id:
            debug_print("No task_id provided to poll")
            return None

        poll_count = 0
        max_polls = self.max_poll_time // self.poll_interval
        start_time = time.time()

        # VERBOSE: Log poll start
        debug_print(f"Starting KIE poll for task_id={task_id}")
        debug_print(f"Poll URL: {self.STATUS_ENDPOINT}")
        debug_print(f"Timeout: {self.max_poll_time}s, Interval: {self.poll_interval}s, Max polls: {max_polls}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            while poll_count < max_polls:
                await asyncio.sleep(self.poll_interval)
                poll_count += 1
                elapsed = time.time() - start_time

                try:
                    response = await client.get(
                        self.STATUS_ENDPOINT,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        params={"taskId": task_id}
                    )

                    # VERBOSE: Log every poll response
                    status_code = response.status_code
                    text = response.text
                    debug_print(f"KIE Poll #{poll_count} ({elapsed:.1f}s): status={status_code}")
                    debug_print(f"KIE body: {text[:500]}")

                    if status_code != 200:
                        debug_print(f"KIE Poll returned {status_code}, continuing...")
                        continue

                    data = response.json()
                    task_data = data.get("data", {})
                    success_flag = task_data.get("successFlag")

                    debug_print(f"KIE successFlag: {success_flag}")

                    # Check completion status via successFlag
                    if success_flag == 1:  # Completed successfully
                        response_data = task_data.get("response", {})
                        urls = response_data.get("resultUrls", [])
                        if urls:
                            video_url = urls[0]
                            debug_print(f"KIE COMPLETED! video_url={video_url}")
                            return video_url
                        raise Exception("No resultUrls in completed KIE response")

                    elif success_flag == -1:  # Failed
                        error = task_data.get("errorMessage", "Unknown error")
                        debug_print(f"KIE FAILED: {error}")
                        raise Exception(f"KIE.ai generation failed: {error}")

                    # successFlag == 0 means still processing
                    debug_print(f"KIE still processing... attempt {poll_count}/{max_polls}")

                except httpx.TimeoutException:
                    debug_print(f"KIE Poll #{poll_count} timed out, retrying...")
                except httpx.HTTPError as e:
                    debug_print(f"KIE Poll #{poll_count} HTTP error: {e}")

        # Timeout reached
        debug_print(f"KIE TIMEOUT after {self.max_poll_time}s ({poll_count} polls)")
        return None

    async def _generate_placeholder(self, request: VideoRequest, start_time: float) -> VideoResult:
        """Generate a placeholder video clip with FFmpeg or use pre-existing URLs."""
        import tempfile
        import shutil

        generation_time = time.time() - start_time

        # Pre-existing placeholder videos (working AI-generated commercials on catbox)
        # Used when FFmpeg is not available (e.g., on Render without ffmpeg installed)
        FALLBACK_URLS = {
            "16:9": "https://litter.catbox.moe/m8jg2n.mp4",  # 64s landscape commercial
            "9:16": "https://litter.catbox.moe/iwxqiu.mp4",  # 56s portrait commercial
            "1:1": "https://litter.catbox.moe/x2owau.mp4",   # 64s square commercial
        }

        # Check if FFmpeg is available
        ffmpeg_available = shutil.which("ffmpeg") is not None
        if not ffmpeg_available:
            debug_print("FFmpeg not available - using pre-existing placeholder URL")
            fallback_url = FALLBACK_URLS.get(request.aspect_ratio, FALLBACK_URLS["16:9"])
            return VideoResult(
                video_url=fallback_url,
                status=GenerationStatus.COMPLETED,
                model_used="placeholder_fallback",
                cost_usd=0.0,
                generation_time_seconds=generation_time,
                scene_number=request.scene_number,
                source="placeholder"
            )

        # FFmpeg available - generate custom placeholder
        # Color palette based on scene mood
        mood_colors = {
            "intriguing": "0x1a1a2e",
            "frustrated": "0x8b0000",
            "hopeful": "0x2e8b57",
            "confident": "0x4169e1",
            "energetic": "0xff6b35",
            "professional": "0x2c3e50",
            "cinematic": "0x0a0a0f",
        }

        style_lower = request.style.lower()
        color = mood_colors.get(style_lower, "0x0a0a0f")

        # Generate in temp directory
        temp_dir = Path(tempfile.gettempdir()) / "genesis_video"
        temp_dir.mkdir(exist_ok=True)

        output_path = temp_dir / f"placeholder_scene_{request.scene_number}_{int(time.time())}.mp4"

        # Determine resolution
        if request.aspect_ratio == "9:16":
            resolution = "1080x1920"
        elif request.aspect_ratio == "1:1":
            resolution = "1080x1080"
        else:
            resolution = "1920x1080"

        try:
            # Plain color video - no drawtext (fonts not available in Docker)
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={color}:s={resolution}:d={int(request.duration)}:r=30",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and output_path.exists():
                logger.info(f"[Scene {request.scene_number}] Generated placeholder: {output_path}")
                return VideoResult(
                    video_path=str(output_path),
                    status=GenerationStatus.COMPLETED,
                    model_used="placeholder",
                    cost_usd=0.0,
                    generation_time_seconds=generation_time,
                    scene_number=request.scene_number,
                    source="placeholder"
                )
            else:
                error = result.stderr.decode()[:200] if result.stderr else "Unknown error"
                logger.error(f"FFmpeg placeholder failed: {error}")

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg placeholder timed out")
        except FileNotFoundError:
            logger.error("FFmpeg not found - using fallback URL")
        except Exception as e:
            logger.error(f"Placeholder generation failed: {e}")

        # FFmpeg failed - use pre-existing placeholder URL as fallback
        debug_print("FFmpeg failed - using pre-existing placeholder URL")
        fallback_url = FALLBACK_URLS.get(request.aspect_ratio, FALLBACK_URLS["16:9"])
        return VideoResult(
            video_url=fallback_url,
            status=GenerationStatus.COMPLETED,
            model_used="placeholder_fallback",
            cost_usd=0.0,
            generation_time_seconds=generation_time,
            scene_number=request.scene_number,
            source="placeholder"
        )

    async def generate_batch(
        self,
        prompts: List[Dict[str, Any]],
        duration_per_scene: float = 5.0,
        style: str = "cinematic",
        max_concurrent: int = 3
    ) -> List[VideoResult]:
        """
        Generate videos for multiple scenes with limited concurrency.

        Args:
            prompts: List of scene prompts from Agent 3
            duration_per_scene: Duration for each scene
            style: Visual style for all scenes
            max_concurrent: Max concurrent generations (to respect rate limits)

        Returns:
            List of VideoResult for each scene
        """
        # VERBOSE: Log batch start
        debug_print("========== BATCH START ==========")
        debug_print(f"Generating {len(prompts)} videos with style={style}")
        debug_print(f"API configured: {self.is_configured}")
        debug_print(f"API key present: {bool(self.api_key)}")
        debug_print(f"API key prefix: {self.api_key[:10]}..." if self.api_key else "API key: None")
        debug_print(f"Circuit state: {self.circuit.state.value}")
        debug_print(f"Max concurrent: {max_concurrent}")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_with_limit(prompt: Dict[str, Any], scene_num: int) -> VideoResult:
            async with semaphore:
                # Add delay between requests to avoid rate limiting (except first scene)
                if scene_num > 1:
                    debug_print(f"Adding 2s delay before scene {scene_num} to avoid rate limiting")
                    await asyncio.sleep(2.0)
                request = VideoRequest(
                    prompt=prompt.get("prompt", prompt.get("scene_description", "")),
                    duration=prompt.get("duration", duration_per_scene),
                    style=prompt.get("mood", style),
                    scene_number=scene_num
                )
                return await self.generate(request)

        tasks = [
            generate_with_limit(prompt, i + 1)
            for i, prompt in enumerate(prompts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scene {i + 1} failed: {result}")
                final_results.append(VideoResult(
                    status=GenerationStatus.FAILED,
                    scene_number=i + 1,
                    error=str(result),
                    source="error"
                ))
            else:
                final_results.append(result)

        # VERBOSE: Log batch summary
        success_count = sum(1 for r in final_results if r.status == GenerationStatus.COMPLETED)
        total_cost = sum(r.cost_usd for r in final_results)
        debug_print("========== BATCH COMPLETE ==========")
        debug_print(f"Results: {success_count}/{len(prompts)} successful")
        debug_print(f"Total cost: ${total_cost:.2f}")
        debug_print(f"Sources: {[r.source for r in final_results]}")

        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "total_requests": self.stats.total_requests,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "total_cost_usd": self.stats.total_cost_usd,
            "avg_generation_time": self.stats.avg_generation_time,
            "circuit_state": self.circuit.state.value,
            "is_configured": self.is_configured,
            "credits_exhausted": self.credits_exhausted,
        }
        # Add timestamp if credits were exhausted
        if self.credits_exhausted and self.credits_exhausted_time:
            stats["credits_exhausted_at"] = datetime.fromtimestamp(
                self.credits_exhausted_time
            ).isoformat()
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_video_generator(
    api_key: Optional[str] = None
) -> VideoGeneratorAgent:
    """Create VideoGeneratorAgent instance with KIE.ai API."""
    return VideoGeneratorAgent(kie_api_key=api_key)


# =============================================================================
# TESTING
# =============================================================================

async def test_video_generator():
    """Test the video generator agent."""
    agent = create_video_generator()

    print("\n[VideoGenerator] Test Mode")
    print("=" * 60)
    print(f"API Configured: {agent.is_configured}")
    print(f"Stats: {agent.get_stats()}")

    # Test with a single scene
    request = VideoRequest(
        prompt="Cinematic establishing shot of a modern office building at sunrise with dramatic lighting",
        duration=5.0,
        style="cinematic",
        scene_number=1
    )

    print(f"\nTest request: {request.prompt[:50]}...")

    # This will use placeholder since no API key
    result = await agent.generate(request)

    print(f"\nResult:")
    print(f"  Status: {result.status}")
    print(f"  Source: {result.source}")
    print(f"  Model: {result.model_used}")
    print(f"  Cost: ${result.cost_usd:.2f}")
    print(f"  Time: {result.generation_time_seconds:.1f}s")
    if result.video_path:
        print(f"  Path: {result.video_path}")


if __name__ == "__main__":
    asyncio.run(test_video_generator())
