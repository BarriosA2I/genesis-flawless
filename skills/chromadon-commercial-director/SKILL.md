# CHROMADON SKILL v2: Commercial Director AI Testing
## WITH CONVERSATION STATE MACHINE

## Overview
This skill enables automated testing of the Barrios A2I Commercial Director AI.
**CRITICAL**: This version includes explicit conversation handling - Claude Code must follow the state machine exactly.

## Target URL
```
https://barriosa2i.com/creative-director
```

---

## CONVERSATION STATE MACHINE

### THE CORE PROBLEM THIS SOLVES
Previous versions failed because Claude Code would:
- Type all answers at once instead of waiting for questions
- Not detect which question was being asked
- Click random elements instead of the chat input
- Not wait for AI responses before continuing

### THE SOLUTION: Explicit Turn-Taking Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSATION FLOW                            │
├───────┘
```

---

## QUESTION DETECTION PATTERNS

Claude Code MUST detect questions by matching these patterns in the AI's message:

```javascript
const QUESTION_PATTERNS = {
  "business_name": [
    "business name",
    "company name",
    "what.*business",
    "what.*company",
    "who are you",
    "tell me about your business"
  ],

  "product_service": [
    "product",
    "service",
    "what do you offer",
    "what does.*do",
    "primary offering",
    "what.*sell"
  ],

  "target_audience": [
    "target audience",
    "who are your customers",
    "who do you serve",
    "ideal customer",
    "target market",
    "who.*trying to reach"
  ],

  "campaign_goal": [
    "goal",
    "objective",
    "what.*trying to achieve",
    "campaign.*about",
    "purpose"
  ],

  "cta": [
    "call to action",
    "cta",
    "what.*want.*do",
    "action.*take"
  ],

  "tone_style": [
    "tone",
    "style",
    "vibe",
    "aesthetic",
    "look and feel",
    "mood"
  ],

  "logo_assets": [
    "logo",
    "image",
    "asset",
    "brand.*material",
    "upload",
    "visual"
  ],

  "confirmation": [
    "ready to create",
    "start production",
    "looks good",
    "confirm",
    "approve",
    "proceed"
  ],

  "voice_selection": [
    "voice",
    "narrator",
    "voiceover",
    "select.*voice",
    "choose.*voice"
  ]
};
```

---

## ANSWER MAP

When a question pattern is detected, respond with EXACTLY this answer:

```javascript
const ANSWER_MAP = {
  "business_name": "Barrios A2I",

  "product_service": "Commercial Director AI - an autonomous system that generates production-ready video commercials for any business using AI agents and the RAGNAROK pipeline. It handles intake, research, scriptwriting, storyboarding, video generation, QC, and export with zero human intervention.",

  "target_audience": "Marketing directors at B2B SaaS companies, agency owners who want to scale video production, e-commerce managers needing constant ad creative, and enterprise teams seeking AI automation infrastructure.",

  "campaign_goal": "Launch video announcing Commercial Director AI to position Barrios A2I as enterprise-grade AI infrastructure - not an agency, but a product company building autonomous systems.",

  "cta": "Two CTAs: BEGIN INTAKE (primary) and BOOK A DEMO (secondary). Final tagline: Barrios A2I — Alienation → Innovation.",

  "tone_style": "Enterprise tech giant launch aesthetic - confident, inevitable, autonomous. Like Apple keynote meets Bloomberg terminal. Obsidian/deep navy gradients, neon cyan particle fields, glassmorphism UI panels, amber highlights. Smooth camera movements only. NO talking heads - all cinematic B-roll.",

  "logo_assets": ">>> SPECIAL CASE: DO NOT TYPE - USE upload_file COMMAND <<<",

  "confirmation": "Yes, let's start production!",

  "voice_selection": "Daniel"
};
```

---

## LOGO UPLOAD PROCEDURE (CRITICAL - READ THIS)

### THE PROBLEM THIS SOLVES
Claude Code keeps typing "YES" or "No logo needed" instead of actually uploading the logo file.

### WHAT NOT TO DO
```
WRONG: Type "yes" in chat
WRONG: Type "Yes, I have a logo"
WRONG: Type "No logo needed"
WRONG: Skip the logo upload
WRONG: Describe the logo in text
```

### WHAT TO DO - EXACT STEPS

**When AI asks about logo/images (matches: "logo", "image", "asset", "upload"):**

```
STEP 1: DO NOT TYPE ANYTHING IN CHAT YET

STEP 2: Take a screenshot to see the page state
        mcp__chrome-devtools__take_screenshot()

STEP 3: Take a snapshot to find the file input element
        mcp__chrome-devtools__take_snapshot()

STEP 4: Look for file upload element in snapshot:
        - input[type="file"]
        - button with paperclip icon
        - button with "attach" or "upload" label
        - The UID will look like: uid=XXX_YY

STEP 5: Use upload_file with the EXACT logo path:
        mcp__chrome-devtools__upload_file(
            uid="<the file input uid from snapshot>",
            filePath="C:\\Users\\gary\\python-genesis-flawless\\skills\\chromadon-commercial-director\\barrios-a2i-logo.png"
        )

STEP 6: Wait 2 seconds for upload to process

STEP 7: THEN type the logo placement instructions in chat:
        "This is the Barrios A2I logo. Use it in every scene:
        - Scene 1: Logo fades in inside holographic orb, centered
        - Scene 2: Logo docks top-left as system insignia
        - Scene 3: Logo etched into system grid beneath agents
        - Scene 4: Logo locks above output as certification seal
        - Scene 5: Logo as watermark on preview screens
        - Scene 6: Logo pulses once at center
        - Scene 7: Network converges INTO the logo
        - Scene 8: Logo centered and dominant on end card"

STEP 8: Press Enter to send
```

### LOGO FILE LOCATION
```
ABSOLUTE PATH: C:\Users\gary\python-genesis-flawless\skills\chromadon-commercial-director\barrios-a2i-logo.png

ESCAPED FOR JSON: C:\\Users\\gary\\python-genesis-flawless\\skills\\chromadon-commercial-director\\barrios-a2i-logo.png
```

### IF FILE INPUT NOT FOUND
If you cannot find a file input element:
1. Look for a button that might trigger file selection
2. Try clicking it first, then look for the file input
3. If still not found, take a screenshot and report the issue
4. DO NOT fall back to typing "no logo" - the logo upload is REQUIRED

---

## EXPLICIT STEP-BY-STEP INSTRUCTIONS

### STEP 1: Navigate and Wait
```
ACTION: Navigate to https://barriosa2i.com/creative-director
WAIT: Until page fully loads (networkidle)
WAIT: 3 additional seconds for JavaScript initialization
SCREENSHOT: Capture initial state
```

### STEP 2: Wait for Welcome Message
```
WAIT: Look for AI welcome message containing "Welcome" or "Creative Director" or "Let's create"
DO NOT: Type anything until you see the AI's first message
SCREENSHOT: Capture welcome message
```

### STEP 3: Conversation Loop
```
REPEAT UNTIL production starts:

  A. READ the most recent AI message
     - Find element: [class*="sys_override"], [class*="message"], [class*="assistant"]
     - Extract the text content
     - Log: "AI said: [message text]"

  B. DETECT the question type
     - Match message text against QUESTION_PATTERNS
     - Log: "Detected question type: [type]"
     - If no match, log warning and wait 5 seconds

  C. GET the answer
     - Lookup from ANSWER_MAP
     - Log: "Will answer: [answer]"
     - **IF question type is "logo_assets": STOP - Go to LOGO UPLOAD PROCEDURE section**

  D. TYPE the answer (SKIP THIS FOR LOGO - use upload_file instead)
     - Find chat input: input[placeholder*="command"], textarea, input[type="text"]
     - Clear any existing text
     - Type the answer
     - WAIT 500ms

  E. SEND the message
     - Press Enter key
     - OR click send button if Enter doesn't work
     - Log: "Sent answer"

  F. WAIT for AI response
     - Wait for new message to appear (different from last)
     - Timeout: 30 seconds
     - If timeout, screenshot and log error

  G. CHECK for special UI elements
     - Voice selector: If visible, click "Daniel" option
     - Production status: If "Production complete" visible, exit loop
     - Error message: If visible, screenshot and report
```

### STEP 4: Voice Selection (if applicable)
```
DETECT: Voice selector UI visible (cards with voice names)
ACTION: Click on "Daniel" voice card
WAIT: For "Continue" button to appear
ACTION: Click "Continue with Daniel" button
WAIT: For confirmation
```

### STEP 5: Production Monitoring
```
WAIT: For production to complete
- Look for "Production complete" text
- Look for progress bar at 100%
- Timeout: 5 minutes

SCREENSHOT: Final state
LOG: Production status
```

### STEP 6: Verify Gallery
```
ACTION: Navigate to https://video-preview-theta.vercel.app/gallery.html
WAIT: Page load
SCREENSHOT: Gallery page
VERIFY: New video appears in gallery
```

---

## CRITICAL RULES FOR CLAUDE CODE

1. **NEVER type multiple answers at once** - One answer per turn
2. **ALWAYS wait for AI response** before typing next answer
3. **ALWAYS detect which question** is being asked before answering
4. **NEVER click random elements** - Only interact with chat input and specific buttons
5. **ALWAYS screenshot** at each major step for debugging
6. **LOG everything** - Question detected, answer being sent, response received
7. **If confused, WAIT** - Don't guess, wait 5 seconds and re-read the page
8. **Voice selector is a special case** - It's a card grid, not chat input

---

## SELECTOR REFERENCE

### Chat Input (try in order)
```javascript
const chatInputSelectors = [
  'input[placeholder*="command"]',
  'input[placeholder*="Enter"]',
  'textarea',
  'input[type="text"]:not([type="hidden"])',
  '[contenteditable="true"]'
];
```

### AI Messages
```javascript
const aiMessageSelectors = [
  '[class*="sys_override"]',
  '[class*="SYS_OVERRIDE"]',
  '[class*="assistant"]',
  '[class*="response"]',
  '[class*="message"]:not([class*="user"])'
];
```

### Voice Selector
```javascript
const voiceSelectors = {
  voiceCard: '[class*="voice-card"], [class*="VoiceCard"]',
  danielOption: 'text=Daniel, button:has-text("Daniel")',
  continueButton: 'button:has-text("Continue"), button:has-text("Select")'
};
```

### Production Status
```javascript
const productionSelectors = {
  complete: 'text=Production complete, text=complete, [class*="complete"]',
  progress: '[class*="progress"], progress',
  error: '[class*="error"], text=error'
};
```

---

## EXAMPLE CONVERSATION TRACE

```
[CHROMADON] Navigated to https://barriosa2i.com/creative-director
[CHROMADON] Waiting for page load...
[CHROMADON] Page loaded, waiting for welcome message...
[CHROMADON] AI MESSAGE: "Welcome to the A2I Commercial Lab! I'm your AI Creative Director..."
[CHROMADON] Detected: welcome/intro message (no question yet)
[CHROMADON] Waiting for first question...
[CHROMADON] AI MESSAGE: "What's your business name?"
[CHROMADON] DETECTED: business_name
[CHROMADON] ANSWERING: "Barrios A2I"
[CHROMADON] Typed in chat input, pressing Enter...
[CHROMADON] Waiting for AI response...
[CHROMADON] AI MESSAGE: "Great! What product or service does Barrios A2I offer?"
[CHROMADON] DETECTED: product_service
[CHROMADON] ANSWERING: "Commercial Director AI - an autonomous system..."
[CHROMADON] Typed in chat input, pressing Enter...
... continues through tone_style ...
[CHROMADON] AI MESSAGE: "Do you have a logo or product images to include?"
[CHROMADON] DETECTED: logo_assets >>> SPECIAL CASE - FILE UPLOAD <<<
[CHROMADON] NOT typing in chat - using upload_file instead
[CHROMADON] Taking snapshot to find file input...
[CHROMADON] Found file input: uid=123_45
[CHROMADON] Uploading: C:\Users\gary\python-genesis-flawless\skills\chromadon-commercial-director\barrios-a2i-logo.png
[CHROMADON] Upload complete, now typing placement instructions...
[CHROMADON] Typed logo instructions, pressing Enter...
[CHROMADON] Waiting for AI response...
[CHROMADON] AI MESSAGE: "Select a voice for your commercial"
[CHROMADON] DETECTED: voice_selection (SPECIAL UI)
[CHROMADON] Looking for voice selector cards...
[CHROMADON] Found voice grid, clicking "Daniel"...
[CHROMADON] Clicked Continue button...
[CHROMADON] AI MESSAGE: "Starting production..."
[CHROMADON] Production in progress, monitoring...
[CHROMADON] Production complete! Verifying gallery...
[CHROMADON] SUCCESS: Video appears in gallery
```

---

## TROUBLESHOOTING

### "Claude Code types YES instead of uploading logo"
CAUSE: Following old instructions that said "No logo needed"
FIX: When logo question detected, DO NOT TYPE - use mcp__chrome-devtools__upload_file command
See: LOGO UPLOAD PROCEDURE section above

### "Claude Code types everything at once"
CAUSE: Not waiting for AI responses
FIX: After every Enter press, WAIT for new AI message before continuing

### "Claude Code answers wrong question"
CAUSE: Not detecting question patterns
FIX: Log the AI message text, match against QUESTION_PATTERNS

### "Voice selector not working"
CAUSE: Trying to type in chat instead of clicking cards
FIX: Voice selection is a click interaction, not a type interaction

### "Production never completes"
CAUSE: May have answered questions wrong or skipped confirmation
FIX: Check screenshots at each step to verify flow

---

## USAGE FOR CLAUDE CODE

```
CHROMADON TEST INSTRUCTIONS:

1. Read this skill file completely
2. Navigate to https://barriosa2i.com/creative-director
3. Follow the EXPLICIT STEP-BY-STEP INSTRUCTIONS exactly
4. Use QUESTION_PATTERNS to detect what's being asked
5. Use ANSWER_MAP to get the correct response
6. Take screenshots at every step
7. Report success or failure with evidence
```
