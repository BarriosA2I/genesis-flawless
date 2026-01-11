"""
Voice listing endpoint for ElevenLabs integration.
Add this to flawless_api.py after the voice_preview endpoint.
"""

# Add this import at the top of flawless_api.py if not already present:
# import httpx

VOICES_ENDPOINT_CODE = '''
@app.get("/api/voices", tags=["Voice"])
async def list_voices():
    """
    List available ElevenLabs voices for the voice selector UI.

    Returns a curated list of professional voices suitable for commercials.
    Voices are categorized by gender and style for easy selection.
    """
    import httpx

    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    # Default curated voices (fallback if API fails)
    default_voices = [
        {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "female", "description": "Warm, professional American female", "preview_url": None},
        {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi", "category": "female", "description": "Strong, confident female", "preview_url": None},
        {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Bella", "category": "female", "description": "Soft, gentle female", "preview_url": None},
        {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni", "category": "male", "description": "Well-rounded, warm male", "preview_url": None},
        {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold", "category": "male", "description": "Crisp, authoritative male", "preview_url": None},
        {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam", "category": "male", "description": "Deep, narrative male", "preview_url": None},
        {"voice_id": "yoZ06aMxZJJ28mfd3POQ", "name": "Sam", "category": "male", "description": "Raspy, dynamic male", "preview_url": None},
        {"voice_id": "jBpfuIE2acCO8z3wKNLl", "name": "Gigi", "category": "female", "description": "Childlike, animated female", "preview_url": None},
    ]

    if not elevenlabs_api_key:
        logger.warning("ElevenLabs API key not configured, returning default voices")
        return {"voices": default_voices, "source": "default"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": elevenlabs_api_key}
            )

            if response.status_code == 200:
                data = response.json()
                voices = []
                for voice in data.get("voices", []):
                    voices.append({
                        "voice_id": voice.get("voice_id"),
                        "name": voice.get("name"),
                        "category": voice.get("labels", {}).get("gender", "unknown"),
                        "description": voice.get("labels", {}).get("description", voice.get("description", "")),
                        "preview_url": voice.get("preview_url"),
                        "labels": voice.get("labels", {})
                    })
                return {"voices": voices, "source": "elevenlabs_api", "count": len(voices)}
            else:
                logger.warning(f"ElevenLabs voices API returned {response.status_code}, using defaults")
                return {"voices": default_voices, "source": "default"}

    except httpx.TimeoutException:
        logger.warning("ElevenLabs voices API timeout, using defaults")
        return {"voices": default_voices, "source": "default"}
    except Exception as e:
        logger.error(f"Error fetching voices: {e}")
        return {"voices": default_voices, "source": "default", "error": str(e)}
'''

print("Copy the endpoint code from VOICES_ENDPOINT_CODE and add it to flawless_api.py")
print("Location: After the voice_preview endpoint, before ROOT & DOCS section")
