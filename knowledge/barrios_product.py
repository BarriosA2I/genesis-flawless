"""
============================================================================
BARRIOS A2I PRODUCT KNOWLEDGE
============================================================================
Central source of truth for commercial specifications.
Used by RAGNAROK agents, script writer, and frontend.
============================================================================
"""

BARRIOS_PRODUCT_KNOWLEDGE = """
## Barrios A2I Commercial Specifications

**Standard Commercial Package:**
- Total Duration: 64 seconds
- Scene Count: 4 Primary Scenes
- Pacing: 16 seconds per scene (High-retention cinematic pacing)

**Standard 4-Scene Structure:**
1. HOOK (0:00-0:16): Disrupt the scroll with a high-impact visual "Pattern Interrupt."
2. PROBLEM (0:16-0:32): Agitate the core pain point identified in Trinity research.
3. SOLUTION (0:32-0:48): Show the product/service as the only logical resolution.
4. CTA (0:48-1:04): Direct call to action with logo placement and contact info.

**Production Standards:**
- Voiceover: Premium OpenAI TTS HD
- Visuals: 8K Photorealistic or Stylized Cinematic (based on tone)
- Logo: Mandatory placement in Scene 4 (CTA)
- Music: Royalty-free background track matching tone
- Transitions: Professional cross-dissolves or dynamic cuts

**Pricing Tiers:**
- Starter: $449/mo (8 tokens = 1 commercial)
- Creator: $899/mo (16 tokens = 2 commercials)
- Growth: $1,699/mo (32 tokens = 4 commercials)
- Scale: $3,199/mo (64 tokens = 8 commercials)
- Single Test: $500 one-time

**Token System:**
- 1 token = 1 x 8-second scene
- 8 tokens = 1 x 64-second commercial
- Tokens reset monthly (no rollover)
"""

COMMERCIAL_CONFIG = {
    "duration_seconds": 64,
    "scene_count": 4,
    "scene_duration_seconds": 16,
    "scene_structure": [
        {
            "scene": 1,
            "type": "HOOK",
            "timestamp": "0:00-0:16",
            "duration_sec": 16,
            "purpose": "Pattern interrupt - stop the scroll",
            "voiceover_sentences": "2-3 max"
        },
        {
            "scene": 2,
            "type": "PROBLEM",
            "timestamp": "0:16-0:32",
            "duration_sec": 16,
            "purpose": "Agitate the pain point",
            "voiceover_sentences": "3-4"
        },
        {
            "scene": 3,
            "type": "SOLUTION",
            "timestamp": "0:32-0:48",
            "duration_sec": 16,
            "purpose": "Product/service as THE answer",
            "voiceover_sentences": "3-4"
        },
        {
            "scene": 4,
            "type": "CTA",
            "timestamp": "0:48-1:04",
            "duration_sec": 16,
            "purpose": "Drive action with logo + contact",
            "voiceover_sentences": "2-3"
        }
    ],
    "production_standards": {
        "voiceover": "OpenAI TTS HD",
        "visual_quality": "8K Photorealistic/Cinematic",
        "logo_placement": "Scene 4 (CTA)",
        "music": "Royalty-free, tone-matched"
    }
}

# Scene type constants for validation
SCENE_TYPES = ["HOOK", "PROBLEM", "SOLUTION", "CTA"]

# Valid tone options
TONE_OPTIONS = [
    "professional",
    "friendly",
    "luxurious",
    "energetic",
    "playful",
    "serious",
    "warm",
    "casual",
    "sophisticated",
    "bold",
    "minimalist"
]
