#!/bin/bash
# ============================================================================
# ASSET UPLOAD E2E TEST
# ============================================================================
# Tests the complete V2 flow with asset collection
# Expected: System asks for logo after gathering 5 fields

API="https://barrios-genesis-flawless.onrender.com"
SESSION="asset-test-$(date +%s)"

echo "========================================"
echo "ASSET UPLOAD E2E TEST"
echo "========================================"
echo "Session: $SESSION"
echo "API: $API/api/chat/v2"
echo ""

# 1. Business name
echo "[1/7] Business name..."
RESP=$(curl -s -X POST "$API/api/chat/v2" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"Barrios A2I\"}")
echo "Response: $(echo $RESP | python3 -c 'import sys,json; print(json.load(sys.stdin).get("response","")[:80])' 2>/dev/null || echo 'OK')"
sleep 2

# 2. Product/Service
echo "[2/7] Product..."
RESP=$(curl -s -X POST "$API/api/chat/v2" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"We create AI-powered commercial videos\"}")
echo "Response: $(echo $RESP | python3 -c 'import sys,json; print(json.load(sys.stdin).get("response","")[:80])' 2>/dev/null || echo 'OK')"
sleep 2

# 3. Target audience
echo "[3/7] Target audience..."
RESP=$(curl -s -X POST "$API/api/chat/v2" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"B2B tech companies and marketing agencies\"}")
echo "Response: $(echo $RESP | python3 -c 'import sys,json; print(json.load(sys.stdin).get("response","")[:80])' 2>/dev/null || echo 'OK')"
sleep 2

# 4. Call to action
echo "[4/7] Call to action..."
RESP=$(curl -s -X POST "$API/api/chat/v2" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"Visit barriosa2i.com to get started\"}")
echo "Response: $(echo $RESP | python3 -c 'import sys,json; print(json.load(sys.stdin).get("response","")[:80])' 2>/dev/null || echo 'OK')"
sleep 2

# 5. Tone (THIS SHOULD TRIGGER ASSET REQUEST)
echo "[5/7] Tone..."
RESP=$(curl -s -X POST "$API/api/chat/v2" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"Professional and innovative\"}")

echo ""
echo "========================================"
echo "CRITICAL CHECK: Did AI ask for assets?"
echo "========================================"
echo "$RESP" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    response = d.get('response', '')
    print(f'Response: {response[:300]}...')
    print('')
    print('Assets Reviewed:', d.get('assets_reviewed', False))
    print('Is Complete:', d.get('is_complete', False))
    print('')
    if 'logo' in response.lower() or 'assets' in response.lower() or 'images' in response.lower():
        print('✅ SUCCESS: AI asked for assets!')
    else:
        print('❌ FAIL: AI did NOT ask for assets')
except:
    print('Parse error')
"
echo "========================================"
echo ""
sleep 2

# 6. Upload Logo (via URL)
echo "[6/7] Uploading logo URL..."
LOGO_RESP=$(curl -s -X POST "$API/api/chat/v2" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"message\": \"Here is our logo: https://barriosa2i.com/images/logo.png\"}")

echo "$LOGO_RESP" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(f'Response: {d.get(\"response\", \"\")[:200]}')
    print('')
    print('Assets:', d.get('uploaded_assets', []))
    print('Is Complete:', d.get('is_complete', False))
    print('Current Phase:', d.get('current_phase', 'unknown'))
    print('')
    assets = d.get('uploaded_assets', [])
    if assets and any('logo.png' in str(a) for a in assets):
        print('✅ SUCCESS: Logo detected and stored!')
    else:
        print('⚠️  WARNING: Logo not detected in state')
except:
    print('Parse error')
"
echo ""
sleep 2

# 7. Check session state
echo "[7/7] Verifying session state..."
SESSION_STATE=$(curl -s "$API/api/chat/v2/session/$SESSION")

echo "$SESSION_STATE" | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print('=== FINAL STATE ===')
    print(f'Business: {d.get(\"business_name\", \"N/A\")}')
    print(f'Product: {d.get(\"primary_offering\", \"N/A\")}')
    print(f'Assets: {d.get(\"uploaded_assets\", [])}')
    print(f'Phase: {d.get(\"current_phase\", \"N/A\")}')
    print(f'Complete: {d.get(\"is_complete\", False)}')
except Exception as e:
    print(f'Session state unavailable: {e}')
"

echo ""
echo "========================================"
echo "TEST COMPLETE"
echo "========================================"
echo ""
echo "Expected Behavior:"
echo "1. After 5th field (tone), AI asks: 'Do you have a logo?'"
echo "2. After logo URL, AI acknowledges asset"
echo "3. Session state includes uploaded_assets array"
echo "4. System proceeds to research phase"
echo ""
