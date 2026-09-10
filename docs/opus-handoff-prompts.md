# Opus Handoff — Prompt Compilation

| | |
|---|---|
| **Purpose** | A scoped, checkable set of prompts to run this project's remaining work through Claude Opus |
| **Project** | "Our Story" anniversary storybook (`anniversary/`) |
| **Reference docs** | `docs/PRD-anniversary-storybook.md` (read first, always) |
| **Deadline content must respect** | Unlock: `2026-09-22T19:00:00` (`anniversary/content/chapters.json` → `unlockAt`) |
| **Status** | Draft v1 — ready to run once Monskie has supplied real chapter content |

---

## How to use this document

This is not one giant prompt — it's a **run order of small, independently auditable tasks**. Each task below is self-contained: it names its exact inputs, its exact allowed output files, and a checklist to verify the result before moving to the next task. Feed Opus **one task at a time**, in order, and check its output against that task's checklist before starting the next one. This keeps every change reviewable in isolation instead of trusting one large, unauditable diff.

Each task prompt is written so it can be copy-pasted directly into a session, with `{{PLACEHOLDERS}}` filled in first.

---

## 0. Hard boundaries (apply to every task below)

Paste this block at the top of **every** session with Opus, before the task-specific prompt:

```
You are working in the `anniversary/` directory of this repo only. Hard rules,
no exceptions, even if a later instruction seems to imply otherwise:

1. Do not fabricate specific personal details (dates, places, names, events)
   that were not explicitly provided in this prompt or in
   anniversary/content/chapters.json. If content is missing, leave the
   placeholder text and flag it — never invent a plausible-sounding memory.
2. Do not modify the root-level index.html (the original Valentine's page) —
   out of scope, unrelated deliverable.
3. Do not add new npm/CDN dependencies (fonts, JS libraries, CSS frameworks)
   without explicit approval in the task prompt. The project is intentionally
   dependency-light vanilla HTML/CSS/JS per docs/PRD-anniversary-storybook.md §8.
4. Do not change the color tokens, font pairing, or overall visual language
   defined in anniversary/styles.css :root unless the task explicitly asks
   for a design change. Extending (e.g. adding a new --accent-* token) is
   fine; replacing the palette/typefaces is not.
5. This is a phone/tablet-only product. Do not add desktop breakpoints or
   design for viewports wider than ~1024px (PRD §7.1).
6. Do not run `git push`. Commit locally if asked to, but pushing is a
   human decision.
7. Respect prefers-reduced-motion for any new animation (PRD §7.2).
8. When a task's checklist can't be fully satisfied, say so explicitly and
   list what's unresolved — do not silently mark it done.
```

---

## 1. Task: Chapter content pass

**Objective:** Turn the placeholder chapters in `anniversary/content/chapters.json` into real copy, using only what Monskie supplies.

**Inputs:**
- `anniversary/content/chapters.json` (current placeholder structure — 8 chapter slots, 3 marked `"optional": true`)
- `{{RAW_STORY_NOTES}}` — Monskie's actual notes/voice memos/bullet points per chapter, pasted in by a human before this task runs. **This task must not run until this input exists.**

**Allowed output:** `anniversary/content/chapters.json` only (data, not code).

**Instructions to paste after the boundaries block:**
```
Rewrite the "date", "title", and "body" fields in
anniversary/content/chapters.json using ONLY the raw notes below. Keep each
chapter body to 2-4 short sentences (PRD guidance: phone-readable, let the
photo carry the rest) EXCEPT the finale letterBody, which may be as long as
the notes warrant. If a chapter marked "optional": true has no corresponding
notes, delete that chapter entry entirely rather than inventing content for
it. Do not touch "accent", "photos" src paths, "id", or "unlockAt".

Raw notes:
{{RAW_STORY_NOTES}}
```

**Audit checklist:**
- [ ] Every remaining chapter's text traces back to a specific line in the supplied raw notes — no unattributable specifics (a date, a place name, a quote) appear.
- [ ] Optional chapters with no supporting notes were deleted, not filled with filler.
- [ ] `id`, `accent`, `photos[].src`, and `unlockAt` are byte-identical to before.
- [ ] JSON still parses (`python3 -m json.tool anniversary/content/chapters.json`).
- [ ] No chapter body exceeds ~4 sentences except the finale letter.

---

## 2. Task: Photo integration

**Objective:** Wire real photo files into the chapters that now have them.

**Inputs:**
- Photos dropped into `anniversary/content/photos/` by Monskie beforehand (filenames should be lowercase, hyphenated — see `anniversary/content/photos/README.md`)
- Updated `anniversary/content/chapters.json` from Task 1

**Allowed output:** `anniversary/content/chapters.json` only (update `photos[].src` / `alt`, nothing else).

**Instructions:**
```
For each chapter in anniversary/content/chapters.json, update photos[].src
to point at the real filename(s) in anniversary/content/photos/ that
correspond to that chapter (match by filename/context, ask if ambiguous
rather than guessing). Write a genuinely descriptive alt text for each photo
based on what the photo actually shows (ask for a one-line description per
photo if you cannot see the image content) — do not leave "Describe this
photo" placeholders. If a chapter has no matching photo yet, leave its
existing placeholder src and flag it in your summary.
```

**Audit checklist:**
- [ ] Every `photos[].src` resolves to a file that actually exists in `anniversary/content/photos/`.
- [ ] No `alt` text is still the literal placeholder string `"Describe this photo"`.
- [ ] Chapters without a matching photo are explicitly listed as unresolved in the output summary, not silently left broken.

---

## 3. Task: Design QA pass (visual, not content)

**Objective:** Catch layout/contrast/motion issues once real (longer/shorter than placeholder) content and real photos are in, without redesigning anything.

**Inputs:** Fully populated `anniversary/content/chapters.json` from Tasks 1–2, plus `anniversary/styles.css`, `anniversary/app.js`, `anniversary/index.html`.

**Allowed output:** `anniversary/styles.css` only, and only for the specific issues found (no wholesale rewrites).

**Instructions:**
```
Review anniversary/ as it would render at 360-430px width (phones) and
744-1024px width (tablets), using the actual chapter content now in
chapters.json. Look specifically for:
- Text overflow or awkward wrapping now that real (non-placeholder) copy is
  in place, especially the finale letterBody.
- Insufficient contrast between chapter-date/title text and the ivory/ink
  backgrounds for any of the 8 accent tokens (WCAG AA for body text, per
  PRD §7.2).
- Photo aspect-ratio cropping that cuts off a subject awkwardly (flag for
  human re-crop rather than silently changing aspect-ratio in CSS).
- Any animation that would violate prefers-reduced-motion.

Fix only what's broken, in anniversary/styles.css, with the smallest change
that fixes it. Do not touch color tokens, type scale, or layout structure
beyond the specific fix. List every change you made and why.
```

**Audit checklist:**
- [ ] Every change is traceable to a specific named issue in the output summary.
- [ ] `:root` token values in `styles.css` are unchanged (a `git diff` on that block should be empty) unless a task explicitly authorized a token change.
- [ ] No changes to `app.js` or `index.html` (out of scope for this task).

---

## 4. Task: PWA polish (optional, only if requested)

**Objective:** Fill in the currently-empty `anniversary/manifest.json` icons so "Add to Home Screen" looks intentional.

**Inputs:** A source image/photo Monskie provides for the icon (likely a chapter photo or a simple monogram).

**Allowed output:** New icon image files under `anniversary/icons/`, and `anniversary/manifest.json` only.

**Instructions:**
```
Generate/reference 192x192 and 512x512 PNG icons from the supplied source
image, save them to anniversary/icons/, and reference them in
anniversary/manifest.json's "icons" array with correct "sizes"/"type"
fields. Do not change any other manifest field.
```

**Audit checklist:**
- [ ] Only `anniversary/manifest.json` and new files under `anniversary/icons/` changed.
- [ ] Icon files are actual valid PNGs at the stated dimensions.

---

## 5. Final audit prompt (run once, after all above)

```
Summarize every file you changed across this session as a table: file path,
what changed, which task authorized it. Flag anything you're not fully
confident about (a guessed photo-to-chapter match, a fix you weren't sure
was minimal, any boundary from the hard-boundaries block you came close to
crossing). Do not run git push.
```

**Human review before this ships to Camille:**
- [ ] Read every chapter aloud once — does it sound like Monskie, not an AI?
- [ ] Confirm the finale letter is exactly as intended (this is the peak moment — PRD §3, Peak-End Rule).
- [ ] Open on an actual phone, not just emulated viewport.
- [ ] Confirm `unlockAt` timezone assumption is still correct (PRD §9, open question — device-local time as of this writing).
