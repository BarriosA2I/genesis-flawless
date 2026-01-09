"""
GENESIS Video Generator Agent (Agent 4)
===============================================================================
AI video generation using Sora 2 and Veo 3.1 via laozhang.ai aggregator.

Features:
- Intelligent routing (Sora 2 for cinematic, Veo 3.1 for cost-effective)
- Circuit breaker pattern for fault tolerance
- Async polling for generation status
- Graceful fallback to placeholders
- Cost tracking per scene

Cost:
- Sora 2: ~$3.00/video (cinematic, complex scenes)
- Veo 3.1: ~$0.15/video (simple, fast generation)

Author: Barrios A2I
Version: 1.0.0 (GENESIS Standalone)
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
    """Supported video generation models"""
    SORA_2 = "sora-2.0"
    VEO_3_1 = "veo-3.1"


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
    source: str = "laozhang"  # laozhang, placeholder, error


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
    failure_threshold: int = 3
    timeout_seconds: int = 60
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
    Agent 4: Video Generator with intelligent model routing.

    Uses laozhang.ai aggregator for cost-effective video generation
    with automatic fallback to placeholders.
    """

    # Model costs via laozhang.ai
    COSTS = {
        VideoModel.SORA_2: 3.00,
        VideoModel.VEO_3_1: 0.15,
    }

    # Routing keywords
    SORA_KEYWORDS = [
        "cinematic", "dramatic", "emotional", "people", "human", "character",
        "tracking shot", "dolly", "zoom", "pan", "realistic", "professional",
        "complex", "motion", "action", "narrative", "story"
    ]

    VEO_KEYWORDS = [
        "product", "logo", "text", "graphics", "simple", "static",
        "animation", "2d", "icon", "minimal", "clean", "modern"
    ]

    def __init__(
        self,
        laozhang_api_key: Optional[str] = None,
        max_retries: int = 3,
        poll_interval: int = 5,
        max_poll_time: int = 300
    ):
        self.api_key = laozhang_api_key or os.getenv("LAOZHANG_API_KEY")
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time

        # Circuit breaker - reset on startup to give fresh code a chance after deploy
        self.circuit = CircuitBreaker(name="laozhang")
        self.circuit.reset()

        # Statistics
        self.stats = GenerationStats()

        # Check if API is configured
        self.is_configured = bool(self.api_key)

        if self.is_configured:
            logger.info("[VideoGenerator] Initialized with laozhang.ai API")
        else:
            logger.warning("[VideoGenerator] No API key - will use placeholders only")

    def _determine_model(self, request: VideoRequest) -> VideoModel:
        """
        Intelligently route to Sora 2 or Veo 3.1 based on prompt content.

        Sora 2: Better for cinematic, complex, human subjects
        Veo 3.1: Better for simple scenes, cost-effective
        """
        if request.model == "sora2":
            return VideoModel.SORA_2
        elif request.model == "veo3.1":
            return VideoModel.VEO_3_1

        # Auto routing
        prompt_lower = request.prompt.lower()

        sora_score = sum(1 for kw in self.SORA_KEYWORDS if kw in prompt_lower)
        veo_score = sum(1 for kw in self.VEO_KEYWORDS if kw in prompt_lower)

        # Long duration favors Sora 2
        if request.duration >= 10:
            sora_score += 1

        # 4K resolution favors Sora 2
        if request.resolution == "4k":
            sora_score += 2

        # Cinematic style favors Sora 2
        if "cinematic" in request.style.lower():
            sora_score += 2

        if sora_score > veo_score + 1:
            logger.info(f"Routing to Sora 2 (score: {sora_score} vs {veo_score})")
            return VideoModel.SORA_2
        else:
            logger.info(f"Routing to Veo 3.1 (score: {veo_score} vs {sora_score})")
            return VideoModel.VEO_3_1

    async def generate(self, request: VideoRequest) -> VideoResult:
        """
        Generate video for a single scene.

        Flow:
        1. Determine optimal model
        2. Submit to laozhang.ai
        3. Poll for completion
        4. Download video
        5. Return result or fallback to placeholder
        """
        start_time = time.time()
        self.stats.total_requests += 1

        # DEBUG: Trace generate() entry
        debug_print(f"generate() called for scene {request.scene_number}")
        debug_print(f"  is_configured: {self.is_configured}")
        debug_print(f"  circuit.state: {self.circuit.state.value}")
        debug_print(f"  circuit.can_attempt(): {self.circuit.can_attempt()}")

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
                    source="laozhang"
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
        """Submit generation request to laozhang.ai using OpenAI-compatible chat completions."""
        url = "https://api.laozhang.ai/v1/chat/completions"

        # Map model enum to laozhang.ai model names
        model_name = "sora-2" if model == VideoModel.SORA_2 else "veo-3.1"

        # Build video generation prompt with all parameters
        video_prompt = (
            f"Generate a {int(request.duration)} second video. "
            f"Aspect ratio: {request.aspect_ratio}. "
            f"Resolution: {request.resolution}. "
            f"Style: {request.style}. "
            f"Description: {request.prompt}"
        )

        payload = {
            "model": model_name,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": video_prompt
                }
            ]
        }

        # VERBOSE: Log submission details
        debug_print(f"Submitting to {url}")
        debug_print(f"Payload: model={model_name}, prompt={video_prompt[:100]}...")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                # VERBOSE: Log raw response
                status_code = response.status_code
                text = response.text
                debug_print(f"Submit response: status={status_code}")
                debug_print(f"Submit body: {text[:500]}")

                if status_code == 429:
                    raise Exception("Rate limited - too many requests")

                response.raise_for_status()
                data = response.json()

                # VERBOSE: Log response structure
                debug_print(f"Full response keys: {list(data.keys())}")

                # OpenAI chat completions format: extract from choices[0].message.content
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    debug_print(f"Chat completion content: {content[:200]}")

                    # Content might be a direct video URL or contain task_id/generation_id
                    # Check if it looks like a URL
                    if content.startswith("http"):
                        debug_print(f"Direct video URL received: {content}")
                        return content  # Direct URL - no polling needed

                    # Try to parse as JSON (might contain video_url or task_id)
                    try:
                        content_data = json.loads(content)
                        video_url = content_data.get("video_url") or content_data.get("url")
                        if video_url:
                            debug_print(f"Video URL from JSON content: {video_url}")
                            return video_url
                        task_id = content_data.get("task_id") or content_data.get("generation_id") or content_data.get("id")
                        if task_id:
                            debug_print(f"Task ID from JSON content: {task_id}")
                            return f"task:{task_id}"  # Prefix to indicate polling needed
                    except (json.JSONDecodeError, TypeError):
                        pass

                    # Return raw content as potential task ID
                    debug_print(f"Returning raw content as task_id: {content[:100]}")
                    return content

                # Fallback: try legacy response format
                task_id = data.get("generation_id") or data.get("id") or data.get("task_id")
                debug_print(f"Fallback task_id from response: {task_id}")
                return task_id

            except Exception as e:
                debug_print(f"Submit exception: {type(e).__name__}: {e}")
                raise

    async def _poll_completion(self, generation_id: str) -> Optional[str]:
        """Poll for generation completion or return direct URL."""
        if not generation_id:
            debug_print("No generation_id provided to poll")
            return None

        # Check if this is already a direct video URL (no polling needed)
        if generation_id.startswith("http"):
            debug_print(f"Direct video URL - no polling needed: {generation_id}")
            return generation_id

        # Check if this is a task ID that needs polling
        actual_id = generation_id
        if generation_id.startswith("task:"):
            actual_id = generation_id[5:]  # Remove "task:" prefix
            debug_print(f"Task ID detected, will poll: {actual_id}")

        poll_count = 0
        max_polls = self.max_poll_time // self.poll_interval
        # Use chat completions endpoint for status check as well
        url = f"https://api.laozhang.ai/v1/chat/completions/status/{actual_id}"
        start_time = time.time()

        # VERBOSE: Log poll start
        debug_print(f"Starting poll for task_id={actual_id}")
        debug_print(f"Poll URL: {url}")
        debug_print(f"Timeout: {self.max_poll_time}s, Interval: {self.poll_interval}s, Max polls: {max_polls}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            while poll_count < max_polls:
                await asyncio.sleep(self.poll_interval)
                poll_count += 1
                elapsed = time.time() - start_time

                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {self.api_key}"}
                    )

                    # VERBOSE: Log every poll response
                    status_code = response.status_code
                    text = response.text
                    debug_print(f"Poll #{poll_count} ({elapsed:.1f}s): status={status_code}")
                    debug_print(f"Poll body: {text[:500]}")

                    if status_code != 200:
                        debug_print(f"Poll returned {status_code}, continuing...")
                        continue

                    data = response.json()

                    # VERBOSE: Log parsed data structure
                    debug_print(f"Parsed keys: {list(data.keys())}")

                    # Check multiple possible status field names
                    gen_status = (
                        data.get("status") or
                        data.get("state") or
                        data.get("generation_status") or
                        ""
                    ).lower()

                    debug_print(f"Generation status: '{gen_status}'")

                    # Check for completion (multiple possible values)
                    if gen_status in ["completed", "succeeded", "success", "done", "finished"]:
                        video_url = (
                            data.get("video_url") or
                            data.get("url") or
                            data.get("output_url") or
                            data.get("result", {}).get("url") if isinstance(data.get("result"), dict) else None
                        )
                        debug_print(f"COMPLETED! video_url={video_url}")
                        return video_url

                    # Check for failure
                    if gen_status in ["failed", "error", "cancelled", "timeout"]:
                        error = data.get("error") or data.get("message") or "Unknown error"
                        debug_print(f"FAILED: {error}")
                        raise Exception(f"Generation failed: {error}")

                    # Still processing
                    progress = data.get("progress") or data.get("percent") or "unknown"
                    debug_print(f"Still processing... progress={progress}")

                except httpx.TimeoutException:
                    debug_print(f"Poll #{poll_count} timed out, retrying...")
                except httpx.HTTPError as e:
                    debug_print(f"Poll #{poll_count} HTTP error: {e}")

        # Timeout reached
        debug_print(f"TIMEOUT after {self.max_poll_time}s ({poll_count} polls)")
        return None

    async def _generate_placeholder(self, request: VideoRequest, start_time: float) -> VideoResult:
        """Generate a placeholder video clip with FFmpeg."""
        import tempfile

        generation_time = time.time() - start_time

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
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c={color}:s={resolution}:d={int(request.duration)}:r=30",
                "-vf", f"drawtext=text='Scene {request.scene_number}':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
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
            logger.error("FFmpeg not found")
        except Exception as e:
            logger.error(f"Placeholder generation failed: {e}")

        # Total failure
        return VideoResult(
            status=GenerationStatus.FAILED,
            cost_usd=0.0,
            generation_time_seconds=generation_time,
            scene_number=request.scene_number,
            error="Failed to generate video or placeholder",
            source="error"
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
        return {
            "total_requests": self.stats.total_requests,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "total_cost_usd": self.stats.total_cost_usd,
            "avg_generation_time": self.stats.avg_generation_time,
            "circuit_state": self.circuit.state.value,
            "is_configured": self.is_configured
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_video_generator(
    api_key: Optional[str] = None
) -> VideoGeneratorAgent:
    """Create VideoGeneratorAgent instance."""
    return VideoGeneratorAgent(laozhang_api_key=api_key)


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
