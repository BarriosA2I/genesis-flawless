"""
WORDSMITH v2.0 LEGENDARY - Test Script
=======================================

Tests the new WORDSMITH OCR + spell-checking system against a video
with known spelling errors:

- "Action trigcted"   -> "Action triggered"   (~00:05)
- "Websiite"          -> "Website"            (~00:40)
- "Wew question"      -> "New question"       (~00:40)
- "New quesiton"      -> "New question"       (~00:40)
- "Analystics"        -> "Analytics"          (~00:40)

Expected: WORDSMITH should BLOCK (return success=False) and list all errors.
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Test video path
TEST_VIDEO = r"C:\Users\gary\Downloads\_BARROSA2I\MEDIA\LAUNCH_VIDEOS\barrios_a2i_3min_WITH_VO_20260120_035046.mp4"


async def main():
    print("=" * 70)
    print("WORDSMITH v2.0 LEGENDARY TEST")
    print("=" * 70)
    print()

    # Check if video exists
    if not os.path.exists(TEST_VIDEO):
        print(f"[ERROR] Test video not found: {TEST_VIDEO}")
        print()
        print("Please ensure the video file exists at the specified path.")
        return

    print(f"[INFO] Test video: {TEST_VIDEO}")
    print(f"[INFO] File size: {os.path.getsize(TEST_VIDEO) / (1024*1024):.1f} MB")
    print()

    # Import WordsmithV2
    print("[INFO] Importing WordsmithV2...")
    try:
        from agents.vortex_postprod.the_wordsmith import WordsmithV2
        print("[OK] WordsmithV2 imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import WordsmithV2: {e}")
        print()
        print("Make sure you're running from the project root:")
        print("  cd C:\\Users\\gary\\python-genesis-flawless")
        print("  python test_wordsmith_legendary.py")
        return

    print()
    print("-" * 70)
    print("INITIALIZING WORDSMITH v2.0")
    print("-" * 70)

    # Initialize WordsmithV2
    wordsmith = WordsmithV2(
        fps_sample_rate=1.0,      # 1 frame per second
        confidence_threshold=0.5,
        blocking_mode=True,
        auto_fix_mode=True
    )

    print(f"[CONFIG] FPS sample rate: {wordsmith.fps_sample_rate}")
    print(f"[CONFIG] Confidence threshold: {wordsmith.confidence_threshold}")
    print(f"[CONFIG] Blocking mode: {wordsmith.blocking_mode}")
    print(f"[CONFIG] Auto-fix mode: {wordsmith.auto_fix_mode}")

    print()
    print("-" * 70)
    print("ANALYZING VIDEO")
    print("-" * 70)
    print()

    # Run analysis
    result = await wordsmith.analyze_video(TEST_VIDEO)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    print(f"Signal:  {result.signal}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")

    if not result.success:
        print()
        print(f"SPELLING ERRORS FOUND: {len(result.errors)}")
        print("-" * 40)

        for i, err in enumerate(result.errors, 1):
            timestamp_sec = err.get('timestamp_ms', 0) / 1000
            print(f"  {i}. '{err.get('original', 'N/A')}' -> '{err.get('correction', 'N/A')}'")
            print(f"     @ {timestamp_sec:.1f}s (frame {err.get('frame', 'N/A')})")
            print()

        if result.corrected_video_path:
            print("-" * 40)
            print(f"CORRECTED VIDEO: {result.corrected_video_path}")
            if os.path.exists(result.corrected_video_path):
                print(f"File size: {os.path.getsize(result.corrected_video_path) / (1024*1024):.1f} MB")
            else:
                print("[WARNING] Corrected video file not found")
    else:
        print()
        print("NO SPELLING ERRORS DETECTED")
        print("[WARNING] This may indicate the OCR is not working correctly,")
        print("          as this video is known to contain spelling errors.")

    print()
    print("=" * 70)
    print("EXPECTED ERRORS (should be detected):")
    print("=" * 70)
    expected_errors = [
        ("Action trigcted", "Action triggered", "~00:05"),
        ("Websiite", "Website", "~00:40"),
        ("Wew question", "New question", "~00:40"),
        ("New quesiton", "New question", "~00:40"),
        ("Analystics", "Analytics", "~00:40"),
    ]

    for orig, corr, ts in expected_errors:
        print(f"  '{orig}' -> '{corr}' at {ts}")

    print()

    # Verify expected errors were caught
    if not result.success and result.errors:
        detected_originals = set(e.get('original', '').lower() for e in result.errors)
        expected_originals = set(e[0].lower() for e in expected_errors)

        caught = detected_originals & expected_originals
        missed = expected_originals - detected_originals

        print("-" * 70)
        print("VERIFICATION")
        print("-" * 70)
        print(f"  Expected errors: {len(expected_originals)}")
        print(f"  Caught:          {len(caught)}")
        print(f"  Missed:          {len(missed)}")

        if missed:
            print()
            print("  MISSED ERRORS:")
            for m in missed:
                print(f"    - {m}")

        if len(caught) == len(expected_originals):
            print()
            print("  [SUCCESS] All expected spelling errors were detected!")
        elif len(caught) > 0:
            print()
            print(f"  [PARTIAL] Detected {len(caught)}/{len(expected_originals)} expected errors")
        else:
            print()
            print("  [FAIL] No expected errors were detected")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
