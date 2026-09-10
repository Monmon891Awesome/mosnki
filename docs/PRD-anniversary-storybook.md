# PRD — "Our Story" Anniversary Storybook Web App

| | |
|---|---|
| **Owner** | Monskie |
| **Recipient / Audience of 1** | Camille |
| **Occasion** | Anniversary — **September 22, 2026, 7:00 PM** |
| **Status** | Draft v1 |
| **Author** | Claude (Sonnet 5) |
| **Repo** | `Monmon891Awesome/mosnki` |

---

## 1. Summary

Upgrade the existing single-page "Will you be my Valentine?" gag page into a **phone/tablet-only, scroll-driven digital storybook** that walks Camille chronologically through the relationship — photo by photo, chapter by chapter — culminating in a "Chapter: Today" moment that unlocks at **7:00 PM on September 22, 2026** (countdown before, celebration after).

This is not a generic Tailwind landing page. The brief specifically asks for:
1. A **restrained, editorial, almost bookish aesthetic** — closer to Anthropic's "printed essay" feel — combined with **Apple's product-marketing craft**: large type, deliberate pacing, scroll-triggered reveals, no visual noise.
2. **Mobile/tablet-first, not responsive-as-an-afterthought.** No desktop layout is in scope. Anything wider than tablet just centers/caps the experience.
3. **Real UX-law grounding**, not decoration for its own sake.
4. A way for Monskie to **add his own photos** without touching code each time.

---

## 2. Design Language Research

### 2.1 What we're borrowing from Anthropic

Anthropic's site (anthropic.com) explicitly avoids the "AI product" visual cliché — no gradients, no glassmorphism, no drop shadows. Findings:

- **Palette:** near-black ink (`#141413`) on warm ivory (`#faf9f5`), with a small set of muted accent tones (clay, fig, cactus, sky, heather, olive) reserved for section-level accents rather than buttons/CTAs everywhere.
- **Typography:** a serif for headlines (literary, essay-like) paired with a clean sans for body copy — type carries the emotion instead of imagery/effects.
- **Depth via layering & color blocking**, not shadows or blur.
- **Whitespace as the primary design tool.** Sections breathe; nothing is crammed.

**Application to this project:** each "chapter" of the storybook reads like a page in a printed book — a pull-quote-style date/headline in a serif display face, a few lines of sans body copy, one hero photo. Accent colors shift subtly per chapter (a "season" palette) instead of one loud brand color everywhere.

### 2.2 What we're borrowing from Apple marketing pages

Apple product pages (and the open-source clones of them) are built on a small set of repeatable techniques:

- **Scroll-pinned hero sections** where a photo/video is pinned in the viewport while text crossfades over it.
- **One idea per screen** — a single sentence, a single image, generous negative space, then the next screen.
- **Parallax at low amplitude** (never more than ~10–15% offset) — enough to feel alive, never gimmicky.
- **Deliberate, physically-plausible easing** (`cubic-bezier` ease-out curves, spring-like entrances) rather than linear/robotic motion.
- **Sticky progress indicators** (a slim dot/line rail) so the user always knows how far into the story they are.

**Application:** Each chapter = one full-height scroll section. The current photo pins briefly, caption text fades/slides in, then releases into the next chapter. A slim vertical "chapter rail" on the right edge (thumb-reachable on a phone) shows progress and lets Camille jump to a chapter she's already seen.

### 2.3 Candidate tech references (GitHub)

Used as **references for technique**, not dependencies to install wholesale — the brief asks to avoid boilerplate Tailwind templates, so we hand-roll CSS using these as inspiration:

- `basementstudio/scrollytelling` — React + GSAP scrollytelling primitives (timeline waypoints, pinning, image-sequence patterns) — reference for how to structure a scroll timeline, not a drop-in library.
- `zhengdechang/awesome-gsap` — catalog of ScrollTrigger patterns (pin transitions, reveal choreography) to borrow easing curves and sequencing ideas from.
- `emanuelefavero/apple-scroll-animation` — vanilla JS, frame-synced-to-scroll technique for Apple-style hero sections, useful for the opening/closing chapters without pulling in Three.js/video-frame overhead we don't need on a phone.
- Anthropic-style token references (`designmd.cc/benchmarks/anthropic`, `design-extractor.com/gallery/anthropic`) — used only to validate our color/type token choices, not copied verbatim (Anthropic's actual brand assets are proprietary; we build an **inspired-by, not identical-to** palette).

### 2.4 Explicit non-goals for the visual system

- No Tailwind utility soup, no generic "glassmorphism card" template, no stock-photo gradients.
- No desktop breakpoint work — this is a **phone/tablet-only artifact**, viewed by exactly one person, most likely on her phone.
- No third-party font-loading bloat — 1 serif display + 1 sans body, self-hosted or a single Google Fonts request, subset if possible.

---

## 3. UX Laws Applied (grounding, not decoration)

| Law | Definition | Where it's applied here |
|---|---|---|
| **Aesthetic-Usability Effect** | Users perceive beautiful things as more usable/trustworthy. | The entire premise — restrained, book-like visual craft is the product, not a wrapper around it. |
| **Peak-End Rule** | People judge an experience mainly by its peak moment and its ending. | Two intentional peaks: (1) the countdown unlocking at exactly 7:00 PM 9/22/26, (2) a final "letter" chapter as the emotional close — both get extra animation/motion budget vs. the middle chapters. |
| **Serial Position Effect** | People best remember the first and last items in a sequence. | First chapter (how we met) and last chapter (today/the letter) get the most visual weight; middle chapters stay simpler by design, not by neglect. |
| **Hick's Law** | More choices = slower decisions. | Navigation is reduced to one gesture: scroll. The only extra control is the chapter-progress rail — no menus, no settings screen. |
| **Fitts's Law** | Time-to-target depends on size and distance. | All tap targets (chapter rail dots, "continue" affordance, photo-upload UI for Monskie) sized ≥44×44px and placed within one-thumb reach at the bottom third / right edge of a phone screen. |
| **Jakob's Law** | Users expect your product to behave like products they already know. | Scroll-to-progress is a pattern everyone already knows from Instagram Stories/Reels-adjacent long-form scrollers — we don't invent new gesture vocabulary. |
| **Von Restorff Effect** | A visually distinct item is remembered best. | The countdown chapter and the "today" unlock chapter intentionally break the established chapter rhythm (different background treatment) so they stand out against the otherwise-consistent chapter template. |
| **Zeigarnik Effect** | Unfinished tasks stay top-of-mind. | Before 9/22/26 7PM, the final chapter is visible-but-locked (a wax-seal / closed-letter visual), creating anticipation instead of hiding it entirely. |
| **Doherty Threshold** | Keep response times < ~400ms to maintain flow. | Photo assets are pre-optimized (responsive `srcset`, AVIF/WebP with JPEG fallback) and animations are GPU-composited (`transform`/`opacity` only) to avoid scroll jank on mid-range phones. |
| **Law of Prägnanz / Proximity** | Users perceive grouped/simple shapes as a single unit. | Each chapter's date, headline, photo, and caption are visually grouped tightly with generous space *between* chapters, so the eye never has to work to parse what belongs together. |

---

## 4. Goals & Non-Goals

### 4.1 Goals
- Recreate the emotional impact of the original Valentine's gag page, but as a **multi-chapter story** instead of a single Yes/No screen.
- Let Monskie **add/reorder photos and captions** per chapter without rewriting HTML/CSS by hand.
- Ship something that feels **premium and editorial**, not templated.
- Build for **phone-first, tablet-compatible**; desktop is explicitly out of scope.
- Time-gate the finale to **September 22, 2026, 7:00 PM** (device-local time, see §7.4 for edge cases).

### 4.2 Non-Goals (v1)
- No user accounts, no backend database, no multi-user support (this is a gift for one recipient).
- No native app / App Store distribution — web app (PWA-installable) only.
- No desktop-optimized layout.
- No CMS/admin backend with auth — "adding pictures" happens via a lightweight local content file + build step (see §6.3), not a hosted admin panel, to keep hosting on GitHub Pages (free, matches current repo setup) trivial.

---

## 5. User & Use Case

**Primary user:** Camille, opening a link on her phone (most likely via text message) on or before 9/22/26.

**Primary flow:**
1. Opens link → sees a closed-book / envelope-style cover screen with her name.
2. Taps/swipes to open → scrolls through chronological chapters (met, first date, milestones, inside jokes, trips, etc.), each with 1 hero photo + a few lines of text.
3. Reaches the final chapter:
   - **Before 9/22/26 7:00 PM:** sees a locked/sealed chapter with a live countdown.
   - **At/after 9/22/26 7:00 PM:** the seal breaks open (animated), revealing the anniversary letter, a closing photo, and a celebratory moment (confetti/hearts, reusing and elevating the existing celebration logic from `index.html`).
4. Can re-visit anytime after unlock; the app remembers (via `localStorage`) that it's already been unlocked so it doesn't need to re-check time-lock logic once past the date.

**Secondary user:** Monskie, as content editor — needs a low-friction way to add new chapters/photos over time (before *and* after the anniversary, since this can keep growing).

---

## 6. Functional Requirements

### 6.1 Cover / Intro
- Full-bleed cover screen, her name, a single tap/swipe-to-begin affordance (evolution of the current envelope interaction in `index.html`).
- Respects `prefers-reduced-motion`.

### 6.2 Chapter Structure
- Each chapter is data-driven: `{ id, date, title, body, photo(s), accentTheme }`.
- Chapter template: full-viewport-height section, pinned photo, text reveal on scroll-into-view (IntersectionObserver-driven, not scroll-jacking the whole page — see §7.2 on performance/accessibility).
- Progress rail (right edge, thumb reachable) shows current chapter / total, tappable to jump.
- Support for 1–3 photos per chapter (single hero, or a short swipeable mini-gallery within the chapter for multi-photo moments).

### 6.3 Content Authoring ("adding pictures")
Given no backend, the lowest-friction approach that still lets Monskie add photos without redeplobegging code review each time:
- **Option A (recommended for v1):** A single `content/chapters.json` file + `content/photos/` folder. Adding a memory = drop a photo in the folder + append a JSON entry. Committed via git (already Monskie's existing workflow in this repo).
- **Option B (stretch goal):** A small local-only "editor mode" (`?edit=true` query flag, password-less since this never leaves Monskie's own device) that lets him preview new chapters in-browser and export the updated JSON — nice-to-have, not a blocker for v1.
- Photos are responsibility of Monskie to source at reasonable resolution (2000px longest edge recommended); build step generates responsive sizes (see §7.3).

### 6.4 Time-Lock / Countdown
- Final chapter is locked until **2026-09-22T19:00:00** in **Camille's local device time** by default (simplest, matches "surprise opens when it's evening for her" intent) — flagged as an open question in §9 since a fixed timezone (e.g., Philippines time) may be more correct if they're not in the same timezone.
- Countdown shows days/hours/minutes live.
- On unlock: one-time animated "seal breaks" sequence, then reveals the letter + closing photo + celebration effect.
- Unlock state persisted in `localStorage` so a revisit after the date doesn't replay the countdown.

### 6.5 Reused/Elevated Assets from Current `index.html`
- Floating hearts / sparkle background → becomes a **subtle, chapter-accent-colored** ambient layer (toned down from the Valentine version to match the more editorial tone) rather than the primary decoration.
- Confetti celebration logic → reused for the finale unlock moment, restyled to match the new palette.
- The playful "runaway No button" mechanic does **not** carry over — tonally wrong for an anniversary storybook; noted explicitly as descoped.

### 6.6 Installability
- Basic PWA manifest + icon so Camille can "Add to Home Screen" and treat it like an app rather than a browser tab.

---

## 7. Non-Functional Requirements

### 7.1 Platform Scope
- **Target viewports:** 360–430px width (phones), 744–1024px width (tablets/iPad-class), portrait-first, landscape-tolerant.
- No desktop breakpoints. If opened on desktop, content simply centers in a max-width column (fallback, not a designed experience).

### 7.2 Accessibility & Motion
- `prefers-reduced-motion: reduce` disables parallax/pin effects, falls back to simple fade-ins.
- All interactive elements meet 44×44px minimum touch target (Fitts's Law, §3).
- Sufficient color contrast even with the muted/editorial palette (WCAG AA minimum for body text).
- Scroll-based reveals use IntersectionObserver + CSS transitions, **not** full scroll-jacking, so native scroll physics/momentum on iOS/Android are preserved (avoids the "janky hijacked scroll" failure mode common in cheap Apple-style clones).

### 7.3 Performance
- Doherty Threshold target: interactions feel instant (<400ms perceived response).
- Images: responsive `srcset`/`sizes`, modern formats with fallback, lazy-loaded below the fold.
- Animations restricted to GPU-friendly properties (`transform`, `opacity`).
- Total initial payload budget: keep first-chapter-visible load lean (target <1.5MB before any below-fold photos load) given this will likely be opened on mobile data.

### 7.4 Timezone Edge Case
- Flagged as open question (§9): "7:00 PM" needs a defined reference — device-local time is simplest to implement but could unlock "early" or "late" for Camille depending on where she is relative to Monskie when it's opened.

### 7.5 Hosting
- Stays on GitHub Pages (matches current repo setup, zero hosting cost, already configured per prior conversation) — static site, no server required, consistent with the no-backend constraint in §4.2.

---

## 8. Tech Approach (high level, not prescriptive)

- **No framework required** — vanilla HTML/CSS/JS is sufficient at this scope and keeps the "no generic Tailwind template" spirit; a tiny build step (e.g. a simple Node script or even hand-maintained) generates the chapter list from `content/chapters.json` and produces responsive image sizes.
- **Scroll choreography:** IntersectionObserver for reveal triggers + CSS `scroll-timeline`/`animation-timeline` where supported, with a JS-driven fallback (progress rail, pin behavior) for broader mobile browser support — modeled on the GSAP ScrollTrigger patterns referenced in §2.3, but hand-rolled to avoid an unnecessary animation-library dependency on a page this scoped.
- **Design tokens:** a small CSS custom-properties file (ink, ivory, 4–6 muted "season" accents, 1 serif + 1 sans type scale) — the Anthropic-inspired system from §2.1, made original rather than copied.
- **State:** `localStorage` only (unlock status, last chapter viewed) — no backend, matches §4.2.

---

## 9. Open Questions

1. **Timezone reference for the 7:00 PM unlock** — device-local vs. a fixed timezone? Needs Monskie's input based on where Camille will most likely be that evening.
2. **Number of chapters / photos** — need the actual content (dates, milestone list, photo set) from Monskie before final IA can be locked.
3. **Editor mode (§6.3 Option B)** — worth building, or is manual JSON editing by Monskie acceptable long-term?
4. **Does the original Valentine's page (`index.html`) stay live/linked**, or does this new storybook replace it as the repo's primary page?
5. **Post-unlock lifecycle** — is this a one-time reveal, or does Monskie want to keep adding chapters after 9/22/26 (i.e., a living storybook vs. a single finished gift)?

---

## 10. Success Criteria

- Camille opens it on her phone and the experience feels **crafted**, not templated — no recognizable "Tailwind starter kit" look.
- Countdown unlocks correctly at the specified moment with no manual intervention needed from Monskie.
- Adding a new photo/chapter takes Monskie **under 5 minutes** (drop image + one JSON entry + git push).
- Fully usable one-handed on a phone, no janky/hijacked scrolling, no motion-sickness-inducing parallax.
