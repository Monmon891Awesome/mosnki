# Adding photos

Every chapter currently points at a `placeholder-*.svg` in this folder. Those exist
only so the layout is viewable — replace them with real photos.

1. Drop your photo in this folder, e.g. `first-date.jpg` (2000px on the longest
   edge is plenty — bigger just slows down loading on her phone).
2. In `../chapters.json`, point that chapter's `photos[].src` at
   `photos/first-date.jpg` and write a short, real `alt` description.
3. Refresh — no build step.

Paths are written **relative to `content/`**, so they always start with `photos/`.
The app resolves them from there.

Photos are shown at a **4:5 portrait crop**, centered (`object-fit: cover`), so
anything important near the top or bottom edge of a landscape photo may get cut.

Keep filenames lowercase, hyphenated, no spaces.

## Deleting a chapter

Three chapters are marked `"optional": true` in `chapters.json`
(`meeting-the-family`, `through-something-hard`, `a-random-tuesday`). If one
doesn't apply, delete that whole entry from the `chapters` array — the progress
rail and chapter count adjust automatically.
