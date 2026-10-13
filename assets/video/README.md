# Story previews

The page contains 337 clips from 36 worlds, arranged into one parent-connected
path per world. `story-order.json` records the source gallery, world and node IDs,
parent relationships, display order, video URLs, and local poster paths. Adjacent
clips must satisfy `next.parent === previous.node`; do not sort by timestamp or
combine branches. Worlds with one available clip are labelled as opening scenes.

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
