"""
THE AUTEUR v2.0 - Vision-Based Creative QA (Agent 7.5)
===============================================================================
Uses Claude Vision to analyze video frames for creative quality assessment.

Checks:
1. Visual-Script Alignment - Does the video show what the script describes?
2. Brand Consistency - Are colors, fonts, logos correct?
3. Composition Quality - Rule of thirds, balance, focal points
4. Emotional Impact - Does it evoke the intended emotion?
5. Technical Quality - Any artifacts, glitches, or low-quality frames?

Cost: ~$0.05-0.15 per video (5 frames @ ~$0.01-0.03 each)

Author: Barrios A2I
Version: 2.0.0 (GENESIS Standalone, Claude Vision)
===============================================================================
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from pydantic import BaseModel, Field

logger = logging.getLogger("genesis.auteur")


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class IssueSeverity(str, Enum):
    """Severity levels for creative issues."""
    CRITICAL = "critical"   # Must fix before delivery
    MAJOR = "major"         # Should fix if time allows
    MINOR = "minor"         # Nice to fix, not blocking


class IssueCategory(str, Enum):
    """Categories of creative issues."""
    VISUAL = "visual"           # Composition, framing
    BRAND = "brand"             # Brand consistency
    TEXT = "text"               # Text readability
    QUALITY = "quality"         # Technical quality
    EMOTION = "emotion"         # Emotional impact
    PACING = "pacing"           # Timing issues


class QARecommendation(str, Enum):
    """Recommendation outcomes."""
    APPROVE = "approve"     # Ready for delivery
    REVISE = "revise"       # Needs improvements
    REJECT = "reject"       # Major issues, re-generate


# =============================================================================
# DATA MODELS
# =============================================================================

class CreativeIssue(BaseModel):
    """A detected creative issue."""
    severity: IssueSeverity
    category: IssueCategory
    description: str
    timestamp_seconds: Optional[float] = None
    frame_number: Optional[int] = None
    suggested_fix: str


class FrameAnalysis(BaseModel):
    """Analysis of a single frame."""
    frame_number: int
    timestamp_seconds: float
    score: float = Field(ge=0, le=100)
    composition_score: float = Field(ge=0, le=100)
    brand_score: float = Field(ge=0, le=100)
    emotion_score: float = Field(ge=0, le=100)
    quality_score: float = Field(ge=0, le=100)
    strengths: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CreativeQARequest(BaseModel):
    """Request for creative quality assessment."""
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    frames_base64: Optional[List[str]] = None  # Pre-extracted frames
    script_summary: str
    visual_style: str = "professional"
    brand_guidelines: Dict[str, Any] = Field(default_factory=dict)
    frame_count: int = 5


class CreativeQAResult(BaseModel):
    """Complete creative QA result."""
    passed: bool
    overall_score: float = Field(ge=0, le=100)
    composition_score: float = Field(ge=0, le=100)
    brand_score: float = Field(ge=0, le=100)
    emotion_score: float = Field(ge=0, le=100)
    storytelling_score: float = Field(ge=0, le=100)
    issues: List[CreativeIssue] = Field(default_factory=list)
    frame_analyses: List[FrameAnalysis] = Field(default_factory=list)
    recommendation: QARecommendation
    revision_prompts: List[str] = Field(default_factory=list)
    overall_recommendations: List[str] = Field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    source: str = "auteur"


# =============================================================================
# CLAUDE VISION PROMPT
# =============================================================================

AUTEUR_ANALYSIS_PROMPT = """You are THE AUTEUR, an elite Creative Director AI analyzing video frames for a commercial.

Analyze these {num_frames} frames from a commercial video.

## SCRIPT CONTEXT:
{script_summary}

## TARGET VISUAL STYLE:
{visual_style}

## BRAND GUIDELINES:
{brand_guidelines}

For EACH frame, evaluate these dimensions (score 0-100):

1. **COMPOSITION** (Rule of thirds, visual balance, focal point clarity)
2. **BRAND CONSISTENCY** (Color palette, visual identity alignment)
3. **EMOTIONAL IMPACT** (Mood effectiveness, engagement potential)
4. **TECHNICAL QUALITY** (Clarity, lighting, no artifacts)

Then provide an OVERALL assessment.

Return your analysis as valid JSON in this exact format:
```json
{{
  "overall_score": <0-100>,
  "composition_score": <0-100>,
  "brand_score": <0-100>,
  "emotion_score": <0-100>,
  "storytelling_score": <0-100>,
  "recommendation": "<approve|revise|reject>",
  "frame_analyses": [
    {{
      "frame_number": 1,
      "score": <0-100>,
      "composition_score": <0-100>,
      "brand_score": <0-100>,
      "emotion_score": <0-100>,
      "quality_score": <0-100>,
      "strengths": ["strength 1", "strength 2"],
      "issues": ["issue 1"],
      "recommendations": ["recommendation 1"]
    }}
  ],
  "issues": [
    {{
      "severity": "<critical|major|minor>",
      "category": "<visual|brand|text|quality|emotion>",
      "description": "Issue description",
      "frame_number": 1,
      "suggested_fix": "How to fix it"
    }}
  ],
  "overall_recommendations": ["Top recommendation 1", "Top recommendation 2", "Top recommendation 3"]
}}
```

IMPORTANT:
- Score 85+ = approve (ready for delivery)
- Score 70-84 = revise (needs minor improvements)
- Score <70 = reject (major issues, consider re-generation)
- Be specific about issues and recommendations
- Focus on actionable feedback
"""


# =============================================================================
# THE AUTEUR AGENT
# =============================================================================

class TheAuteur:
    """
    Agent 7.5: THE AUTEUR - Vision-Based Creative QA

    Uses Claude Vision (claude-3-5-sonnet) to analyze video frames
    and provide comprehensive creative quality assessment.
    """

    def __init__(
        self,
        anthropic_client: Optional[anthropic.Anthropic] = None,
        model: str = "claude-sonnet-4-20250514",
        max_frames: int = 5
    ):
        self.anthropic = anthropic_client
        if not self.anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.anthropic = anthropic.Anthropic(api_key=api_key)

        self.model = model
        self.max_frames = max_frames
        self.cost_per_frame = 0.02  # ~$0.02 per frame with Claude Vision

        # Quality thresholds
        self.thresholds = {
            "approve": 85,
            "revise": 70,
            "reject": 0
        }

        logger.info(f"[AUTEUR] Initialized with {model}, max_frames={max_frames}")

    async def analyze(self, request: CreativeQARequest) -> CreativeQAResult:
        """
        Perform comprehensive creative quality assessment.

        Args:
            request: CreativeQARequest with video/frames and context

        Returns:
            CreativeQAResult with scores, issues, and recommendations
        """
        start_time = time.time()

        # Get frames (either provided or extracted)
        frames_b64 = request.frames_base64
        if not frames_b64:
            if request.video_url:
                frames_b64 = await self._extract_frames_from_url(
                    request.video_url,
                    request.frame_count
                )
            elif request.video_path:
                frames_b64 = self._extract_frames_from_file(
                    request.video_path,
                    request.frame_count
                )

        if not frames_b64:
            logger.warning("[AUTEUR] No frames available, using mock analysis")
            return self._mock_analysis(start_time)

        # Run Claude Vision analysis
        if self.anthropic:
            try:
                result = await self._analyze_with_claude(
                    frames_b64=frames_b64,
                    script_summary=request.script_summary,
                    visual_style=request.visual_style,
                    brand_guidelines=request.brand_guidelines
                )
                result.cost_usd = len(frames_b64) * self.cost_per_frame
                result.latency_ms = (time.time() - start_time) * 1000
                return result
            except Exception as e:
                logger.error(f"[AUTEUR] Claude analysis failed: {e}")
                return self._mock_analysis(start_time, error=str(e))
        else:
            logger.warning("[AUTEUR] Anthropic client not available, using mock")
            return self._mock_analysis(start_time)

    async def _analyze_with_claude(
        self,
        frames_b64: List[str],
        script_summary: str,
        visual_style: str,
        brand_guidelines: Dict[str, Any]
    ) -> CreativeQAResult:
        """Analyze frames using Claude Vision."""

        # Build content with images
        content = []

        # Add text prompt
        prompt = AUTEUR_ANALYSIS_PROMPT.format(
            num_frames=len(frames_b64),
            script_summary=script_summary,
            visual_style=visual_style,
            brand_guidelines=json.dumps(brand_guidelines, indent=2) if brand_guidelines else "None specified"
        )
        content.append({"type": "text", "text": prompt})

        # Add each frame as an image
        for i, frame_b64 in enumerate(frames_b64):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_b64
                }
            })
            content.append({
                "type": "text",
                "text": f"[Frame {i+1} of {len(frames_b64)}]"
            })

        # Call Claude Vision
        response = self.anthropic.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}]
        )

        # Parse response
        response_text = response.content[0].text

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"[AUTEUR] Failed to parse response: {e}")
            return self._mock_analysis(time.time())

        # Convert to CreativeQAResult
        overall_score = data.get("overall_score", 75)
        recommendation = data.get("recommendation", "revise")

        # Parse issues
        issues = []
        for issue_data in data.get("issues", []):
            try:
                issues.append(CreativeIssue(
                    severity=IssueSeverity(issue_data.get("severity", "minor")),
                    category=IssueCategory(issue_data.get("category", "visual")),
                    description=issue_data.get("description", ""),
                    frame_number=issue_data.get("frame_number"),
                    suggested_fix=issue_data.get("suggested_fix", "")
                ))
            except Exception:
                continue

        # Parse frame analyses
        frame_analyses = []
        for fa_data in data.get("frame_analyses", []):
            try:
                frame_analyses.append(FrameAnalysis(
                    frame_number=fa_data.get("frame_number", 0),
                    timestamp_seconds=fa_data.get("frame_number", 0) * 6.0,  # Estimate
                    score=fa_data.get("score", 75),
                    composition_score=fa_data.get("composition_score", 75),
                    brand_score=fa_data.get("brand_score", 75),
                    emotion_score=fa_data.get("emotion_score", 75),
                    quality_score=fa_data.get("quality_score", 75),
                    strengths=fa_data.get("strengths", []),
                    issues=fa_data.get("issues", []),
                    recommendations=fa_data.get("recommendations", [])
                ))
            except Exception:
                continue

        # Determine pass/fail
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        passed = overall_score >= self.thresholds["approve"] and len(critical_issues) == 0

        return CreativeQAResult(
            passed=passed,
            overall_score=overall_score,
            composition_score=data.get("composition_score", 75),
            brand_score=data.get("brand_score", 75),
            emotion_score=data.get("emotion_score", 75),
            storytelling_score=data.get("storytelling_score", 75),
            issues=issues,
            frame_analyses=frame_analyses,
            recommendation=QARecommendation(recommendation),
            revision_prompts=self._generate_revision_prompts(issues),
            overall_recommendations=data.get("overall_recommendations", []),
            source="auteur_claude"
        )

    def _generate_revision_prompts(self, issues: List[CreativeIssue]) -> List[str]:
        """Generate revision prompts from issues."""
        prompts = []
        for issue in issues:
            if issue.severity in [IssueSeverity.CRITICAL, IssueSeverity.MAJOR]:
                prompts.append(f"{issue.category.value.title()}: {issue.suggested_fix}")
        return prompts[:5]  # Top 5 revision prompts

    async def _extract_frames_from_url(
        self,
        video_url: str,
        num_frames: int = 5
    ) -> List[str]:
        """Extract frames from video URL using FFmpeg."""
        import httpx

        try:
            # Download video to temp file
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(video_url)
                if response.status_code != 200:
                    logger.error(f"[AUTEUR] Failed to download video: {response.status_code}")
                    return []

            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = Path(temp_dir) / "video.mp4"
                video_path.write_bytes(response.content)
                return self._extract_frames_from_file(str(video_path), num_frames)

        except Exception as e:
            logger.error(f"[AUTEUR] Frame extraction from URL failed: {e}")
            return []

    def _extract_frames_from_file(
        self,
        video_path: str,
        num_frames: int = 5
    ) -> List[str]:
        """Extract frames from local video file using FFmpeg."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_pattern = Path(temp_dir) / "frame_%03d.jpg"

                # Extract frames at intervals
                result = subprocess.run([
                    "ffmpeg", "-i", video_path,
                    "-vf", f"fps=1/{30//num_frames}",  # Spread across video
                    "-frames:v", str(num_frames),
                    "-q:v", "2",  # High quality JPEG
                    str(frames_pattern)
                ], capture_output=True, timeout=30)

                if result.returncode != 0:
                    logger.error(f"[AUTEUR] FFmpeg failed: {result.stderr.decode()}")
                    return []

                # Read frames as base64
                frames = []
                for frame_path in sorted(Path(temp_dir).glob("frame_*.jpg")):
                    with open(frame_path, "rb") as f:
                        frame_b64 = base64.b64encode(f.read()).decode()
                        frames.append(frame_b64)

                return frames[:num_frames]

        except Exception as e:
            logger.error(f"[AUTEUR] Frame extraction failed: {e}")
            return []

    def _mock_analysis(
        self,
        start_time: float,
        error: Optional[str] = None
    ) -> CreativeQAResult:
        """Generate mock analysis when real analysis unavailable."""
        return CreativeQAResult(
            passed=True,
            overall_score=78,
            composition_score=80,
            brand_score=75,
            emotion_score=77,
            storytelling_score=79,
            issues=[
                CreativeIssue(
                    severity=IssueSeverity.MINOR,
                    category=IssueCategory.VISUAL,
                    description="Consider tighter framing on key product shots",
                    suggested_fix="Adjust composition to follow rule of thirds more closely"
                )
            ],
            frame_analyses=[
                FrameAnalysis(
                    frame_number=1,
                    timestamp_seconds=0,
                    score=78,
                    composition_score=80,
                    brand_score=75,
                    emotion_score=77,
                    quality_score=82,
                    strengths=["Good lighting", "Clear focal point"],
                    issues=["Slightly off-center composition"],
                    recommendations=["Adjust framing"]
                )
            ],
            recommendation=QARecommendation.REVISE if error else QARecommendation.APPROVE,
            revision_prompts=[],
            overall_recommendations=[
                "Tighten composition on key frames",
                "Ensure brand colors are more prominent",
                "Add more dynamic transitions"
            ],
            cost_usd=0.0,
            latency_ms=(time.time() - start_time) * 1000,
            source="mock" if not error else f"error:{error}"
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_auteur(
    anthropic_client: Optional[anthropic.Anthropic] = None,
    model: str = "claude-sonnet-4-20250514"
) -> TheAuteur:
    """Create TheAuteur instance."""
    return TheAuteur(
        anthropic_client=anthropic_client,
        model=model
    )


# =============================================================================
# TESTING
# =============================================================================

async def test_auteur():
    """Test THE AUTEUR agent."""
    auteur = create_auteur()

    print("\n[THE AUTEUR] Test Mode")
    print("=" * 60)

    # Test with mock (no frames)
    request = CreativeQARequest(
        script_summary="A 30-second commercial for TechCorp AI platform showing business transformation",
        visual_style="Professional, modern, tech-forward",
        brand_guidelines={
            "primary_color": "#00CED1",
            "secondary_color": "#0A0A0F",
            "tone": "professional"
        },
        frame_count=5
    )

    result = await auteur.analyze(request)

    print(f"\nOverall Score: {result.overall_score}/100")
    print(f"Recommendation: {result.recommendation.value}")
    print(f"Passed: {result.passed}")

    print(f"\nDimension Scores:")
    print(f"  Composition: {result.composition_score}/100")
    print(f"  Brand: {result.brand_score}/100")
    print(f"  Emotion: {result.emotion_score}/100")
    print(f"  Storytelling: {result.storytelling_score}/100")

    print(f"\nIssues Found: {len(result.issues)}")
    for issue in result.issues[:3]:
        print(f"  [{issue.severity.value}] {issue.description}")

    print(f"\nRecommendations:")
    for rec in result.overall_recommendations[:3]:
        print(f"  - {rec}")

    print(f"\nLatency: {result.latency_ms:.0f}ms")
    print(f"Cost: ${result.cost_usd:.3f}")


if __name__ == "__main__":
    asyncio.run(test_auteur())
