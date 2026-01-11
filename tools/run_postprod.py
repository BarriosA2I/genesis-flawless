"""
================================================================================
RUN POST-PRODUCTION LOCALLY
================================================================================
Runs the RAGNAROK post-production pipeline on local video files.

Requires a .env file with:
- ANTHROPIC_API_KEY
- ELEVENLABS_API_KEY

Usage:
    python run_postprod.py --source "C:\\Users\\gary\\Desktop\\BARRIOS A2I LAUNCH" --output "C:\\Users\\gary\\Desktop\\RAGNAROK_OUTPUT"

Author: Barrios A2I | 2026-01-11
================================================================================
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
try:
    from dotenv import load_dotenv
    # Try multiple locations for .env
    env_locations = [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
        Path.home() / ".env",
        Path("C:/Users/gary/.env"),
    ]
    for env_path in env_locations:
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded env from: {env_path}")
            break
except ImportError:
    print("Warning: python-dotenv not installed, using system environment")

# Check keys
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

if not anthropic_key:
    print("\nAPI KEYS NOT FOUND!")
    print("=" * 60)
    print("Create a .env file with your API keys:")
    print()
    print("    ANTHROPIC_API_KEY=sk-ant-api03-...")
    print("    ELEVENLABS_API_KEY=...")
    print()
    print("Put it in one of these locations:")
    print("  - C:\\Users\\gary\\python-genesis-flawless\\.env")
    print("  - C:\\Users\\gary\\.env")
    print("=" * 60)
    sys.exit(1)

print(f"Anthropic API Key: {anthropic_key[:15]}...")
print(f"ElevenLabs API Key: {'SET' if elevenlabs_key else 'NOT SET'}")

# Now run the main script
if __name__ == "__main__":
    import asyncio
    from ragnarok_postprod import main
    asyncio.run(main())
