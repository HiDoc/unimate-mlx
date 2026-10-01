# UniMate Studio design system

## Surface

The local studio is an **operating light table**: the user's character is the artifact, 2-second motions are strips of material to arrange, and the interface provides precise controls around a dark viewing stage. It is a work tool, not a marketing page.

## Layout

- Desktop: a narrow preparation rail for upload and prompt queue; a wide viewing area for the character, playback, timeline, and generated assets.
- The timeline is a single ordered strip of 2-second clips. It remains visible while reviewing results.
- On small screens, preparation and viewer stack without hiding the queue or downloads behind tabs.

## Visual tokens

- Warm neutral canvas `#F2F0E9`, sheet `#FBFAF6`, deep ink `#252A29`, quiet ink `#646B66`, fine rule `#D9DCD5`.
- A single signal orange `#D9643A` marks primary action, selected prompt, and playhead. Forest `#53776D` marks completed work; error is dark red `#A64338`.
- Viewing stage `#161D1C` with a controlled grid and soft light so varied uploaded rigs remain legible.
- DM Sans for interface text; IBM Plex Mono only for timecodes, counts, and numerical run data.
- Flat planes and 1px separators define hierarchy; shadows appear only on lifted interactive surfaces.

## Behavior

- Upload gives an immediate local model preview while the server prepares the skeleton after submission.
- Presets add editable prompts to a reorderable queue; the custom field uses the same queue.
- During generation, one persistent progress area names the current stage and clip, shows elapsed time and an updating ETA, and keeps completed clips visible.
- Individual and chained animations use one viewer and one playback vocabulary. Clicking a timeline clip seeks to its 2-second slot.
- Failures name the stage and offer the log; no silent loader stalls.

## Responsive and accessibility

- All controls have visible keyboard focus, clear labels, and text alongside icons.
- Progress has a live text status and numeric estimate; it does not rely on color alone.
- Reduced-motion preference removes ornamental easing without hiding state transitions.
- Uploaded 3D content is supplementary to downloadable artifacts and descriptive clip text.
