"""
Regression tests for Creative Director V2
Run with: python -m pytest tests/test_creative_director.py -v
Or run standalone: python tests/test_creative_director.py
"""
import requests
import time
import json
import sys

API_BASE = "https://barrios-genesis-flawless.onrender.com"
# For local testing:
# API_BASE = "http://localhost:8000"


class TestIntakeFlow:
    """Test the 5-field intake process"""

    def setup_method(self):
        self.session_id = f"test-{int(time.time())}"

    def send_message(self, message: str) -> dict:
        """Send a message to the V2 chat endpoint"""
        try:
            r = requests.post(
                f"{API_BASE}/api/chat/v2",
                json={
                    "session_id": self.session_id,
                    "message": message
                },
                timeout=30
            )
            return r.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "response": ""}

    def test_hello_asks_for_business(self):
        """
        CRITICAL TEST: Hello should ask for business name, NOT logo

        This tests the bug where intake skips all 5 fields and jumps to assets.
        """
        result = self.send_message("Hello")
        response = result.get("response", "").lower()

        # Should NOT mention logo/assets yet
        assert "logo" not in response, f"BUG: Asked for logo too early! Response: {response}"
        assert "image" not in response, f"BUG: Asked for images too early! Response: {response}"
        assert "asset" not in response, f"BUG: Asked for assets too early! Response: {response}"
        assert "upload" not in response, f"BUG: Asked for upload too early! Response: {response}"

        # Should ask for business name (or be a welcome message that asks for it)
        expected_words = ["business", "company", "name", "welcome"]
        assert any(word in response for word in expected_words), \
            f"Should ask for business name, got: {response}"

        print("✅ test_hello_asks_for_business PASSED")

    def test_full_intake_sequence(self):
        """Complete intake should collect all 5 fields before asking for assets"""

        # Step 1: Hello
        r1 = self.send_message("Hello")
        response1 = r1.get("response", "").lower()
        assert "logo" not in response1, f"Step 1: Asked for logo too early"
        print(f"  Step 1 (Hello): {r1.get('response', '')[:100]}...")

        # Step 2: Business name
        r2 = self.send_message("TechFlow Solutions")
        response2 = r2.get("response", "").lower()
        assert "logo" not in response2, f"Step 2: Asked for logo too early"
        print(f"  Step 2 (Business): {r2.get('response', '')[:100]}...")

        # Step 3: Product
        r3 = self.send_message("AI automation software for small businesses")
        response3 = r3.get("response", "").lower()
        assert "logo" not in response3, f"Step 3: Asked for logo too early"
        print(f"  Step 3 (Product): {r3.get('response', '')[:100]}...")

        # Step 4: Audience
        r4 = self.send_message("Small business owners and entrepreneurs")
        response4 = r4.get("response", "").lower()
        assert "logo" not in response4, f"Step 4: Asked for logo too early"
        print(f"  Step 4 (Audience): {r4.get('response', '')[:100]}...")

        # Step 5: CTA
        r5 = self.send_message("Start your free trial")
        response5 = r5.get("response", "").lower()
        assert "logo" not in response5, f"Step 5: Asked for logo too early"
        print(f"  Step 5 (CTA): {r5.get('response', '')[:100]}...")

        # Step 6: Tone - NOW it should ask for logo/assets
        r6 = self.send_message("Professional but friendly")
        response6 = r6.get("response", "").lower()

        # After all fields, it SHOULD ask for assets
        asset_words = ["logo", "image", "asset", "upload"]
        has_asset_prompt = any(word in response6 for word in asset_words)
        print(f"  Step 6 (Tone): {r6.get('response', '')[:100]}...")

        assert has_asset_prompt, \
            f"Should ask for assets after all fields, got: {r6.get('response', '')}"

        print("✅ test_full_intake_sequence PASSED")

    def test_session_isolation(self):
        """Different sessions should have independent state"""
        session_a = f"test-a-{int(time.time())}"
        session_b = f"test-b-{int(time.time())}"

        # Session A: Complete business name
        r_a1 = requests.post(
            f"{API_BASE}/api/chat/v2",
            json={"session_id": session_a, "message": "Hello"},
            timeout=30
        ).json()

        r_a2 = requests.post(
            f"{API_BASE}/api/chat/v2",
            json={"session_id": session_a, "message": "My business is Alpha Corp"},
            timeout=30
        ).json()

        # Session B: Should start fresh, not inherit Alpha Corp
        r_b1 = requests.post(
            f"{API_BASE}/api/chat/v2",
            json={"session_id": session_b, "message": "Hello"},
            timeout=30
        ).json()

        # Session B should not know about Alpha Corp
        assert "alpha" not in r_b1.get("response", "").lower(), \
            "Session B leaked data from Session A!"

        print("✅ test_session_isolation PASSED")


class TestVoiceSelector:
    """Test voice selection feature"""

    def test_voices_endpoint_exists(self):
        """API should have a voices endpoint"""
        try:
            r = requests.get(f"{API_BASE}/api/voices", timeout=10)
            assert r.status_code == 200, f"Voices endpoint returned {r.status_code}"

            data = r.json()
            voices = data.get("voices", [])
            assert len(voices) > 0, "No voices returned"
            print(f"✅ Found {len(voices)} voices (source: {data.get('source', 'unknown')})")

        except requests.exceptions.RequestException as e:
            print(f"❌ test_voices_endpoint_exists FAILED: {e}")
            raise

    def test_voice_preview_endpoint_exists(self):
        """API should have a voice preview endpoint"""
        # Just check it exists (don't actually call it without a valid voice_id)
        try:
            r = requests.post(
                f"{API_BASE}/api/voice/preview",
                json={"voice_id": "test", "text": "test"},
                timeout=10
            )
            # Expect either 200 (success) or 4xx (bad request due to invalid voice_id)
            # but NOT 404 (endpoint doesn't exist)
            assert r.status_code != 404, "Voice preview endpoint not found"
            print(f"✅ Voice preview endpoint exists (returned {r.status_code})")

        except requests.exceptions.RequestException as e:
            print(f"❌ test_voice_preview_endpoint_exists FAILED: {e}")
            raise


class TestScriptGeneration:
    """Test script output format"""

    def test_script_format_64_seconds(self):
        """Script should be 64 seconds with 4 scenes"""
        # This test requires completing the full flow
        # TODO: Implement after intake is fixed
        print("⏳ test_script_format_64_seconds SKIPPED (requires full flow)")


def run_smoke_tests():
    """Run quick smoke tests for immediate validation"""
    print("=" * 60)
    print("Creative Director V2 Regression Tests")
    print(f"API: {API_BASE}")
    print("=" * 60)

    results = {"passed": 0, "failed": 0}

    # Test 1: Intake - Hello should ask for business
    print("\n[TEST 1] test_hello_asks_for_business")
    try:
        test = TestIntakeFlow()
        test.setup_method()
        test.test_hello_asks_for_business()
        results["passed"] += 1
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        results["failed"] += 1
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results["failed"] += 1

    # Test 2: Voice selector endpoint
    print("\n[TEST 2] test_voices_endpoint_exists")
    try:
        test = TestVoiceSelector()
        test.test_voices_endpoint_exists()
        results["passed"] += 1
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        results["failed"] += 1
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results["failed"] += 1

    # Test 3: Full intake sequence (longer test)
    print("\n[TEST 3] test_full_intake_sequence")
    try:
        test = TestIntakeFlow()
        test.setup_method()
        test.test_full_intake_sequence()
        results["passed"] += 1
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        results["failed"] += 1
    except Exception as e:
        print(f"❌ ERROR: {e}")
        results["failed"] += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
    print("=" * 60)

    return results["failed"] == 0


if __name__ == "__main__":
    # Allow command line override of API base
    if len(sys.argv) > 1:
        API_BASE = sys.argv[1]

    success = run_smoke_tests()
    sys.exit(0 if success else 1)
