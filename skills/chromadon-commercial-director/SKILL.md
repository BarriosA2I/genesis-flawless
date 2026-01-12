# CHROMADON SKILL: Commercial Director AI Testing

## Overview
This skill enables automated testing of the Barrios A2I Commercial Director AI assistant located at https://barriosa2i.com/creative-director

## Target URL
```
https://barriosa2i.com/creative-director
```

## Purpose
Test the AI assistant as a client would - answering questions naturally, uploading assets, reviewing scripts, and approving video production.

---

## UI ELEMENT SELECTORS

### Chat Interface
```javascript
// Chat input field
const chatInput = 'input[placeholder*="command"], textarea, [contenteditable="true"]';

// Send button
const sendButton = 'button[type="submit"], button svg[class*="send"], button[aria-label*="send"]';

// File upload button (attachment/paperclip icon)
const fileUpload = 'input[type="file"], button[aria-label*="attach"], button[aria-label*="upload"], .attachment-btn';

// AI response messages
const aiMessage = '[class*="assistant"], [class*="response"], [class*="sys_override"], [class*="message"]';

// Progress indicator
const progressBar = '[class*="progress"], [class*="complete"]';
```

### Navigation
```javascript
// Main nav items
const navItems = {
  pricing: 'a[href*="pricing"]',
  nexus: 'a[href*="nexus"]',
  commercialLab: 'a[href*="commercial"]',
  founder: 'a[href*="founder"]',
  command: 'a[href*="command"]'
};
```

---

## INTERACTION FLOW

### Phase 1: Navigate & Initialize
```
1. Navigate to https://barriosa2i.com/creative-director
2. Wait for page load (networkidle)
3. Screenshot to verify Neural Interface is visible
4. Look for chat input field
```

### Phase 2: Answer AI Questions
The AI assistant asks questions in this order. Answer ONLY what is asked:

| Question | Answer |
|----------|--------|
| Business name? | `Barrios A2I` |
| Product/service? | `Commercial Director AI - an autonomous system that generates production-ready video commercials for any business using AI agents and the RAGNAROK pipeline` |
| Target audience? | `Marketing directors at B2B SaaS companies, agency owners who want to scale video production, e-commerce managers needing constant ad creative, enterprise teams seeking AI automation infrastructure` |
| Campaign goal? | `Launch video announcing Commercial Director AI - positioning Barrios A2I as enterprise-grade AI infrastructure for autonomous video generation` |
| CTA? | `BEGIN INTAKE (primary) and BOOK A DEMO (secondary). Tagline: Barrios A2I — Alienation → Innovation` |
| Tone/style? | `Enterprise tech giant launch - confident, inevitable, autonomous. Like Apple keynote meets Bloomberg terminal. Obsidian/deep navy gradients, neon cyan particles, glassmorphism UI, amber highlights. NO talking heads - all B-roll.` |
| Logo/images? | `YES` - then UPLOAD the logo file |

### Phase 3: Upload Logo
**CRITICAL: Always upload the logo when asked**

```
1. When AI asks about logo/images, answer: "Yes, I have the Barrios A2I logo"
2. Click file upload button
3. Upload: Barrios_a2i_logo-removebg-preview__1_.png
4. After upload, provide logo placement instructions (see LOGO_INSTRUCTIONS below)
```

### Phase 4: Review Script
```
1. Wait for script generation (30-60 seconds)
2. Read the generated script
3. Verify it matches the creative brief
4. If revisions needed, specify exact changes
5. If approved, say: "Looks good" or "Approved"
```

### Phase 5: Production
```
1. Wait for video production to start
2. Monitor progress
3. Download/retrieve final video when complete
```

---

## LOGO_INSTRUCTIONS

When uploading the logo, provide these placement rules:

```
This is the Barrios A2I logo. It MUST appear in EVERY scene:

Scene 1 (0-8s): Logo fades in crisply inside holographic orb, centered
Scene 2 (8-16s): Logo docks top-left as system insignia
Scene 3 (16-24s): Logo etched faintly into system grid beneath agents
Scene 4 (24-32s): Logo locks above output frame as certification seal
Scene 5 (32-40s): Logo as subtle watermark on preview screens
Scene 6 (40-48s): Logo pulses once at center as chaos disappears
Scene 7 (48-56s): Network converges INTO the logo at center
Scene 8 (56-64s): Logo centered and dominant on final end card

LOGO RULES:
- Always flat, readable, undistorted
- Never warp, bend, melt, glitch, or spin
- Treat as system authority mark
- Use soft cyan rim light with faint amber highlights
- Never overpower the scene
```

---

## CREATIVE BRIEF REFERENCE

### Video Specs
- **Duration:** 64 seconds
- **Structure:** 8 scenes × 8 seconds each
- **Aspect:** 9:16 (primary)
- **Model:** Veo 3.1

### Global Style (ALL scenes)
- Obsidian / deep navy gradient environment
- Neon cyan particle fields (controlled, elegant)
- Glassmorphism UI panels
- Subtle amber highlights
- Smooth dolly or gimbal motion only (NO handheld)
- High-end studio lighting with volumetric glow

### Negative Prompt (inject into every scene)
```
no misspelled text, no gibberish UI, no warped logos, no jitter, no cheap glitch, no cartoon style, no messy typography, no creepy faces, no extra fingers, no low-res artifacts, no talking heads, no presenters
```

### Scene Breakdown

**Scene 1 (0-8s) - SYSTEM ONLINE**
Dark obsidian command-center. Neon cyan particles drift. Holographic orb stabilizes. Barrios A2I logo fades in inside orb. Glass UI panels lock into place. Slow camera push-in.
TEXT: "COMMERCIAL DIRECTOR AI | 0% HUMAN INTERVENTION | Any business. Any offer."

**Scene 2 (8-16s) - RAGNAROK PIPELINE**
Logo docks top-left. Glowing glass pipeline animates: INTAKE → RESEARCH → SCRIPT → STORYBOARD → GENERATE → QC → EXPORT. Green confirmations.
TEXT: "RAGNAROK PIPELINE | Industry-agnostic automation"

**Scene 3 (16-24s) - AGENTIC SYSTEM**
Multiple autonomous agents operate in parallel. Logo etched into system grid.
TEXT: "Autonomous agents | No handoffs. No delays."

**Scene 4 (24-32s) - ANY BUSINESS PROOF**
Glass tiles representing industries collapse into Commercial Output. Logo as certification seal.
TEXT: "Works for any business | Templates that auto-adapt"

**Scene 5 (32-40s) - OUTPUTS**
Three floating glass screens with format chips: 9:16 • 16:9 • 1:1. Logo as watermark.
TEXT: "Launch • Offer • UGC • Testimonial"

**Scene 6 (40-48s) - HUMAN REMOVAL**
Manual workflows dissolve. Automated pipeline completes. Logo pulses.
TEXT: "From weeks → minutes | Machine-consistent output"

**Scene 7 (48-56s) - ECOSYSTEM SCALE**
Systems interconnect: Commercial Director AI, RAGNAROK Core, Ad Forge, NEXUS Brain. Network converges into logo.
TEXT: "One ecosystem | Built to scale"

**Scene 8 (56-64s) - CTA LOCK-IN**
Final glassmorphism end card. Logo centered. Two glass buttons.
TEXT: "SYSTEM_CONNECTED | BEGIN INTAKE | BOOK A DEMO | Barrios A2I — Alienation → Innovation"

---

## CHROMADON COMMAND SEQUENCE

```javascript
// 1. Navigate
chrome-devtools: navigate
  url: "https://barriosa2i.com/creative-director"

// 2. Wait for load
chrome-devtools: wait_for_load_state
  state: "networkidle"

// 3. Screenshot initial state
chrome-devtools: screenshot
  name: "commercial-director-loaded"

// 4. Find chat input and type
chrome-devtools: type_text
  selector: "input, textarea"
  text: "[answer to current question]"

// 5. Press Enter to send
chrome-devtools: press_key
  key: "Enter"

// 6. Wait for response
chrome-devtools: wait_for_selector
  selector: "[class*='sys_override'], [class*='response']"
  timeout: 30000

// 7. For file upload
chrome-devtools: click
  selector: "input[type='file'], button[aria-label*='attach']"

chrome-devtools: upload_file
  selector: "input[type='file']"
  path: "/path/to/Barrios_a2i_logo.png"

// 8. Screenshot progress
chrome-devtools: screenshot
  name: "script-generated"
```

---

## ERROR HANDLING

### Chat Input Not Found
```javascript
// Try alternative selectors
const alternatives = [
  'textarea',
  'input[type="text"]',
  '[contenteditable="true"]',
  '.chat-input',
  '[data-testid="chat-input"]'
];
```

### File Upload Not Working
If file upload fails, describe the logo in chat:
```
The Barrios A2I logo features crystalline teal/cyan wings with gold accents, spelling out BARRIOS above A2I. It should be treated as a system authority mark - always flat, never warped.
```

### Response Timeout
```javascript
// Increase wait time for script generation
chrome-devtools: wait_for_selector
  timeout: 90000  // 90 seconds for complex operations
```

---

## VALIDATION CHECKLIST

Before approving any generated script, verify:

- [ ] Logo appears in ALL 8 scenes
- [ ] NO talking heads or presenters
- [ ] Duration is 64 seconds (8 × 8s)
- [ ] Visual style matches brief (obsidian, cyan, glassmorphism)
- [ ] Text overlays match specified copy
- [ ] CTAs are correct (BEGIN INTAKE, BOOK A DEMO)
- [ ] Tagline included (Alienation → Innovation)
- [ ] Camera motion is smooth (no handheld)

---

## QUICK REFERENCE

### Essential Files
- Logo: `Barrios_a2i_logo-removebg-preview__1_.png`
- Creative Brief: See CREATIVE BRIEF REFERENCE above

### Key URLs
- Test Page: https://barriosa2i.com/creative-director
- Production API: https://barrios-genesis-flawless.onrender.com

### AI Question Flow
1. Business name → Barrios A2I
2. Product → Commercial Director AI (autonomous video generation)
3. Audience → Marketing directors, agency owners, e-commerce managers
4. Goal → Launch video for enterprise positioning
5. CTA → BEGIN INTAKE / BOOK A DEMO
6. Tone → Enterprise tech giant launch
7. Logo → YES + UPLOAD FILE + placement instructions

---

## USAGE

To use this skill in Claude Code:

```
Read the SKILL.md at /mnt/skills/user/chromadon-commercial-director/SKILL.md
Then navigate to https://barriosa2i.com/creative-director and test the AI assistant.
Answer questions as specified in the skill. Always upload the logo when asked.
```
