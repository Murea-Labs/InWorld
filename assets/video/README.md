# Story previews

The page contains 333 clips from 35 worlds, arranged into one parent-connected
sequence per world. `story-order.json` records the source gallery, world and node IDs,
parent relationships, display order, video URLs, and local poster paths. Adjacent
clips must satisfy `next.parent === previous.node`; do not sort by timestamp or
combine branches. A retained sequence may start after excluded source clips;
the first parent ID remains the original ID even when that parent is not shown.
`source_order`, when present, records the original turn number before display
renumbering. Worlds with one available opening clip are labelled as opening scenes.

The 2026-10-14 review covered all 337 previously displayed videos, decoding
37,858 frames and visually reviewing six evenly spaced frames per clip plus
flagged frames. Two Stellar academy clips with baked-in white side panels and
the island of mist opening containing a white-background character reference
were removed. Stellar academy retains its connected original turns 3–8, shown
as turns 1–6. `quality-review.json` records the exclusions; do not restore them
from the source gallery without a fresh visual review. Original media remains
available at the source; removed posters are recoverable from Git history.

A follow-up review removed the first of The star-sea island's two opening
timer shots because their displayed times conflict. The original second clip
is retained, followed by the original connected turns 3–37; the sequence is
displayed as turns 1–36. This brings the current total to 333 clips.

The video files are hosted at:

`https://aura-data-training.vicoo.ai/Musee/InWorld/story-library-20261013-v1/`

Previews use H.264, a maximum encoded width of 768 pixels, YUV 4:2:0, and MP4
fast start. Audio is omitted for the silent looping previews. The source clips
remain unchanged. Posters live in `assets/img/stories/` so a scene remains visible
while its video buffers. The original twelve MP4s remain available for old links.

`assets/js/motion.js` starts previews only after they settle in the viewport,
pauses them outside it or when the document is hidden, and releases distant
media after 20 seconds. It respects reduced motion and explicit pause controls.
Cards stay 240–288 CSS pixels wide and use `object-fit: contain`.

Worlds with one to three clips share compact grids within their existing story
collections. Each world keeps its heading, anchor, and connected clip order;
whole worlds are packed into four-card rows on wide screens. The grids adapt to
three or two columns on smaller screens and return to individual story tracks
on phones. Navigation arrows appear only when a track overflows.
