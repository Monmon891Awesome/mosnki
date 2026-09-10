# Adding photos and videos

Every chapter currently points at a `placeholder-*.svg` in this folder. Those exist
only so the layout is viewable — replace them with real media.

1. Drop your file in this folder, e.g. `first-date.jpg` or `first-date.mp4`.
2. In `../chapters.json`, add it to that chapter's `media` array with a `src` of
   `photos/first-date.jpg` and a short, real `alt` description.
3. Refresh — no build step.

Paths are written **relative to `content/`**, so they always start with `photos/`.

## Photos vs videos

Both go in the same `media` array. The **file extension** decides how it renders —
`.mp4`, `.mov`, `.m4v`, `.webm`, `.ogv` become video players, anything else is an
image. You never state the type.

- **Photos** are shown at a **4:5 portrait crop**, centered, so anything important
  near the top or bottom edge of a landscape photo may get cut.
- **Videos** keep their own aspect ratio (no crop) and get playback controls. They
  do not autoplay, and only metadata is preloaded so opening the page doesn't pull
  whole clips over mobile data.
- Optional: add `"poster": "photos/still.jpg"` to a video entry for the frame shown
  before it plays.

## Video file size matters

Phone video is big — a minute of 4K can exceed 300 MB. Before committing:

- GitHub **rejects any file over 100 MB** outright, so the push will fail.
- Anything over ~10 MB is slow to load on mobile data.
- Compress first. `ffmpeg -i in.mov -vcodec h264 -crf 28 -vf scale=-2:1080 out.mp4`
  usually cuts a clip by 80–90% with no visible loss on a phone.
- `.mov` from iPhone often works, but `.mp4` (H.264) is the safest for browsers.

Keep filenames lowercase, hyphenated, no spaces.

## Deleting a chapter

Three chapters are marked `"optional": true` in `chapters.json`
(`meeting-the-family`, `through-something-hard`, `a-random-tuesday`). If one
doesn't apply, delete that whole entry from the `chapters` array — the progress
rail and chapter count adjust automatically.
