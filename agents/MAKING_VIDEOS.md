# Making Videos

The primary purpose of aldyparen-py is to create videos of fractals. This document tells an AI agent how to
autonomously design and save an interesting Aldyparen project from a short user prompt—or from no prompt at all.

The required deliverable is a project file, not a rendered video. Save it as
`aldyparen-video-projects/aldyparen_XXX.json`, where `XXX` is the next unused three-digit number. Never overwrite an
existing project. Reserve this number before exploration begins, and store every temporary exploratory render in
the matching `tmp/aldyparen_XXX/` directory. The project can later be opened in the GUI or rendered at any
resolution and frame rate.

## Public API

Aldyparen is designed for interactive use in the GUI, but its Python API can create, preview, animate, and save
projects programmatically. The public rendering types are available from `aldyparen`:

```python
from aldyparen import ColorPalette, Frame, StaticRenderer, Transform, VideoRenderer
from aldyparen.gui.presets import PRESET_NAMES, load_preset
from aldyparen.mixing import make_animation
from aldyparen.painters import MandelbroidPainter
from aldyparen.project import AldyparenProject
```

See `examples/example.ipynb` for a small library example. The most important APIs are:

```python
frame = Frame(painter=painter, transform=transform, palette=palette)
image = StaticRenderer(width, height).render(frame)  # uint8 RGB NumPy array

segment = make_animation(frame1, frame2, length)

project = AldyparenProject.create(
    frames,
    work_frame=frames[0],
    selected_frame_idx=0,
    description=description,
)
project.save(file_name)
```

`make_animation(frame1, frame2, length)` treats `length` as the number of frame intervals and returns
`length + 1` frames, including both key frames. The two key frames must use the same painter class. For example,
at 12 FPS, `length=36` adds three seconds to a multi-key-frame timeline when the shared endpoint is deduplicated.
As a standalone list, those 37 rendered frames last $37/12$ seconds.

Pass `work_frame=frames[0]` to `AldyparenProject.create`, or omit it to use the first frame automatically. The
resulting project stores the frames, selected frame, and description; `project.save(file_name)` adds the project
version and timestamp and writes the repository's JSON format. Do not construct that JSON by hand.

## Core Concepts

An Aldyparen project is an ordered list of `Frame` objects. Each frame contains:

- a `Painter`, which defines the mathematical image and its parameters;
- a `Transform`, which defines the center, scale, and rotation; and
- a `ColorPalette`, which maps the painter's non-negative integer outputs to RGB colors.

Painter color indices wrap around when they exceed the palette length. This makes palette length as important as
the colors themselves.

Create transforms with `Transform.create(...)`. Useful arguments are:

```python
Transform.create(
    center=-0.75 + 0.1j,  # or center_x=... and center_y=...
    scale=4,              # visible frame width in mathematical units
    # scale_log10=-2,     # alternatively, log10 of the visible width
    rotation_deg=15,
)
```

For high-precision deep zooms, provide `center_x` and `center_y` as decimal strings so that Python does not first
round them to ordinary floating-point values.

Useful palette constructors include:

```python
ColorPalette.gradient("navy", "gold", size=64)
ColorPalette.categorical(["black", "red", "orange", "white"])
ColorPalette.grayscale(size=64)
```

Use `load_preset(name)` to obtain a known-good `(painter, transform, palette)` tuple. Current preset names are
listed in `PRESET_NAMES`; they include Mandelbrot, high-precision Mandelbrot, Burning Ship, high-precision Burning
Ship, Lyapunov, and magnetic-pendulum presets.

## Creative Goal

Produce a coherent visual journey rather than a random sequence of fractals. A good video has:

- a clear visual theme or progression;
- strong, intentional colors with enough contrast to reveal structure;
- smooth motion that gives the viewer time to understand each scene;
- a mixture of broad establishing views and detailed close-ups;
- meaningful parameter changes, not motion for its own sake; and
- concise cuts or transitions between incompatible painters.

Transform animation can zoom, pan, and rotate simultaneously. Painter animation can make the mathematical system
itself appear alive. Palette animation can shift mood or emphasize a transition. Prefer one or two simultaneous
changes; changing every property at once often looks noisy and arbitrary.

Unless the user requests otherwise, plan at 12 FPS. Most moving segments should last 2–5 seconds (24–60 frame
intervals). A brief hold can be made by repeating a frame. Avoid long static shots, abrupt accidental jumps, and
rapid motion that makes fine detail impossible to follow.

## Audio-Synchronized Videos

When the user supplies an audio track, analyze it before storyboarding and make the visual timeline from measured
musical events rather than from an assumed constant tempo. Keep the original audio file unchanged. Aldyparen
project files do not contain audio, so save the visual project normally and, only if a final movie is requested,
mux the original track into the rendered video afterward.

### Audio-analysis tools

Use the following local tools; audio-analysis packages are working dependencies and do not need to be added to
Aldyparen's runtime dependencies:

- `ffprobe` for exact duration, codec, channel count, and sample rate;
- `ffmpeg` to decode a temporary mono floating-point or PCM WAV at a consistent sample rate, and later to mux or
  trim audio without changing the source file;
- Python with `librosa` and NumPy for onset strength, onset times, beat tracking, local tempo, RMS energy, spectral
  centroid/rolloff/contrast, and chroma; use `scipy.signal.find_peaks` when explicit peak filtering is useful; and
- Matplotlib for a waveform/spectrogram/feature plot used to inspect the automatic analysis.

Install analysis-only packages in the selected Python environment if they are unavailable. Save generated WAVs,
plots, feature tables, and scripts under the reserved `tmp/aldyparen_XXX/` directory. Export a machine-readable
analysis table (CSV, TSV, or JSON) containing at least event time, event type, strength, nearest video frame, local
tempo, and local energy. Do not rely on one global BPM: electronic tracks can contain tempo changes, half-time
sections, breakdowns, pickups, and syncopated onsets. Compare beat-track results with onset peaks and the plotted
waveform, and correct obvious octave, phase, or section-boundary errors before designing visuals.

### Turn the analysis into a 12 FPS schedule

1. Determine the exact visual duration from the user's request and track duration. At 12 FPS, three minutes is
  exactly 2,160 frames. Record any trim, fade, or duration mismatch explicitly rather than silently drifting out
  of sync.
2. Divide the track into musical sections using sustained changes in energy, onset density, timbre, and harmonic
  content. Refine those boundaries to nearby strong onsets or downbeats. Give each section a mood, painter,
  movement vocabulary, palette family, and rendering-cost budget.
3. Convert event time to the nearest frame with `round(time_seconds * FPS)`. Resolve collisions deliberately when
  several events quantize to one frame: preserve the strongest/downbeat event, combine compatible accents, or
  move a weaker accent by one frame only when that improves the rhythm. Keep all schedule calculations in
  floating-point time and quantize only the final visual events so rounding error cannot accumulate.
4. Use strong beats and downbeats for major palette pulses, cuts, direction changes, or parameter accents. Use
  weaker beats for small luminance/hue pulses. An effective palette pulse has a fast one-frame attack followed by
  a two-to-five-frame decay toward its baseline; scale its amplitude by normalized onset strength and preserve
  enough luminance contrast to keep the fractal readable.
5. In breakdowns and low-energy passages, favor longer transform interpolation, restrained colors, and breathing
  room. In high-energy passages, cut among preselected strong images on beats or subdivisions and use shorter
  motion phrases. Do not make every detected onset a cut, and do not animate expensive painter parameters merely
  to acknowledge a beat when a palette or transform accent will read more clearly.
6. Render contact sheets or low-resolution previews with timecodes for every section, then preview representative
  beat pulses and all rapid-cut passages at 12 FPS. Check sync against the actual audio, not only against the
  feature plot. Revise the schedule if an automatically detected event does not feel musically important.

It is sometimes desirable to put several exactly identical frames in a row. During a rapid-cut section, for
example, repeating each selected image for two or three frames makes it readable for $2/12$ or $3/12$ seconds while
retaining a crisp cut. Identical consecutive frames should reuse one rendered image (or be coalesced into one clip
with the combined duration), so such holds can also save substantial rendering time. Do not replace a deliberate
hold with near-identical interpolated frames, because those still require separate renders.

### Keep audio-driven projects practical to render

Benchmark promising frames after Numba's first-call compilation at the low preview resolution and reject costly
parameter paths early. Start with the following conservative ranges, increasing them only where a preview proves
that the extra work adds visible detail:

- `MandelbroidPainter`: simple formulas and `max_iter` around 60–140; reserve roughly 180–250 for a small number of
  detailed key frames rather than an entire rapid sequence.
- `JuliaPainter`: simple Newton/rational formulas, `iters` around 20–60, and `max_colors` around 8–24.
- `LyapunovFractalPainter`: `warmup` around 30–70 and `iterations` around 40–100; prefer `color_scale` and transform
  changes over unnecessarily raising both iteration counts.
- `MagneticPendulumPainter`: three or four magnets and `max_steps` around 300–700. Tune damping, time step,
  `settle_distance`, and `settle_speed` so most points settle early; use larger step counts only for short scenes
  whose added basin detail is visibly worthwhile.

These are starting budgets, not fixed artistic limits. For fast passages, render a modest bank of visually distinct
frames and hold/cut between them instead of calculating a unique expensive fractal for every one of the 12 frames
per second. Record section boundaries, beat-frame mappings, pulse rules, painter parameters, and any audio trim in
the reproducible build script and project description.

## Autonomous Workflow

### 1. Interpret the prompt

Honor explicit requests for painters, colors, duration, mood, formulas, or subject matter. Treat unspecified details
as creative freedom. If there is no prompt, use the default brief below instead of asking the user for direction.

Before writing code, read the painter descriptions in `README.md`, inspect `aldyparen/gui/presets.py`, and inspect
the relevant painter class and its interpolation branch in `aldyparen/mixing.py`. Use only formulas and parameter
shapes accepted by the current implementation.

### 2. Explore candidates visually

Start from presets and variations inspired by them. Write a temporary exploration script and render previews at a
small 16:9 resolution such as 320×180 or 480×270. Before rendering, determine the next project number and create
`tmp/aldyparen_XXX/`, where `aldyparen_XXX` exactly matches the planned project filename. Save **all** exploratory
renders there—never directly under `tmp/` or elsewhere—so that the user can inspect the agent's exploration in real
time. Use descriptive or sequential filenames, and keep the renders until the task is complete. Do not add these
exploration artifacts to the final project directory.

Explore 20–50 meaningfully different candidates across centers, scales, rotations, formulas or painter parameters,
and palettes. This range is a target, not a reason to generate near-duplicates. Inspect the rendered images rather
than selecting frames from numeric parameters alone. Reject candidates that are mostly flat, noisy, badly framed,
or too expensive to animate.

At minimum, evaluate each promising candidate for:

- composition at the eventual 16:9 aspect ratio;
- visible detail at both preview and likely video resolution;
- palette contrast and banding;
- numerical warnings or large uninformative regions;
- rendering cost; and
- whether nearby transforms or parameters provide a worthwhile path for animation.

### 3. Storyboard key frames

Choose the strongest candidates and organize them into scenes. For each scene, decide what changes between its key
frames and why: reveal a structure by zooming out, enter a detailed boundary by zooming in, orbit a focal feature,
morph compatible formula coefficients, move magnets along a deliberate trajectory, or change colors to mark a
visual phase.

Render every key frame before generating the full animation. Also render several intermediate frames—especially
the midpoint—to catch dull passages, invalid mixed formulas, palette problems, or paths that cross empty space.

Estimate the final duration before generating all frames:

```text
duration in seconds = number of frames / rendering FPS
```

Because adjacent animation segments share a key frame, concatenate them without duplicating that endpoint:

```python
frames = make_animation(key_frames[0], key_frames[1], 36)
for start, end in zip(key_frames[1:-1], key_frames[2:]):
    frames.extend(make_animation(start, end, 36)[1:])
```

Keep deliberate duplicate frames only when a hold is desired.

### 4. Handle painter compatibility correctly

`make_animation` can interpolate transforms and palettes, but painter interpolation has painter-specific rules:

- `MandelbroidPainter`: supported. Numeric literals in `gen_function` may change only when both formulas have the
  same token structure and all non-numeric tokens are identical. `max_iter` and `radius` are interpolated.
- `JuliaPainter`: supported under the same formula-token restriction. `iters`, `tolerance`, and `max_colors` are
  interpolated.
- `LyapunovFractalPainter`: supported only when both key frames use the same `sequence`; other numeric parameters
  are interpolated.
- `MagneticPendulumPainter`: supported only when both key frames contain the same number of magnets. Magnets are
  paired by list position, and their positions and strengths are interpolated.
- `MandelbrotHighPrecisionPainter`: supported; `max_iter` is interpolated.
- Other differing painter configurations, including `MandelbroidHighPrecisionPainter` and
  `SierpinskiCarpetPainter`, cannot currently be interpolated. Equal painter instances can still be animated by
  changing only their transforms and palettes.

Use the same palette size at both ends of a scene when practical. If sizes differ, Aldyparen repeats the shorter
palette to the longer length before interpolating colors; this is valid but can produce surprising intermediate
palettes.

Do not call `make_animation` across different painter classes. Join such scenes with a direct cut. A more polished
option is to end the first scene and begin the next on nearly uniform frames of the same color, but the painter
switch itself is still a cut and the uniform interval should remain under one second.

To make magnets appear or disappear, keep the same maximum magnet count and ordering in every key frame. Give an
inactive magnet a very small positive strength at the transition and animate its strength up or down. Strength must
remain greater than zero. Do not actually add or remove a magnet between key frames passed to `make_animation`.

### 5. Assemble and save the project

Write a reproducible script that constructs all key frames, calls `make_animation`, concatenates the scenes, and
saves the final project. It is acceptable to keep this script temporarily while working; the required artifact is
the JSON project.

Use the project number reserved before exploration. It must have been selected by inspecting both
`aldyparen-video-projects/` and `examples/`, taking the greatest existing `aldyparen_XXX.json` number plus one, and
formatting it with three digits. Confirm again that the destination is still unused, and refuse to overwrite it.

The description passed to `AldyparenProject.create` should be ready to adapt for YouTube. Include:

- a short title and an engaging one- or two-sentence summary;
- timestamped scene descriptions based on 12 FPS, unless another planning FPS was requested;
- every painter used, with its defining formula or mathematical process;
- important formulas, sequences, or changing parameters;
- the track name when an audio track was used for analysis or synchronization;
- a brief credit to aldyparen-py; and
- no claims about resolution, FPS, audio, or rendered quality that the project file cannot guarantee.

Compute timestamps from frame indices instead of estimating them:

```text
timestamp in seconds = frame index / FPS
```

For example, frame 144 begins at `00:12` at 12 FPS. Use `MM:SS` timestamps, or `HH:MM:SS` for videos longer than
an hour.

### 6. Validate the result

Before finishing:

1. Confirm that the frame list is non-empty and that the duration matches the plan.
2. Preview all key frames and representative intermediate frames at 16:9.
3. Check each painter's `warning` after rendering and investigate warnings rather than silently accepting them.
4. Construct the project with `AldyparenProject.create`, then save it with `project.save`.
5. Load the JSON and deserialize every frame in sequence with `Frame.deserialize(..., prev=previous_frame)`, or
   open it through the application, to verify that the project is readable.
6. Confirm that the output name is the next sequential name and that no existing file was modified.
7. Report the output path, frame count, planned FPS, expected duration, painters used, and a short scene summary.

Rendering the final MP4 is optional unless the user explicitly requests it. If requested, use
`VideoRenderer(width, height, fps).render_movie_from_file(project_path, output_path)`. Rendering FPS is not stored
in the project, so use the same FPS used for storyboard timing and description timestamps.

## Default Brief When No Prompt Is Given

Create a 45–75 second, 16:9 visual journey titled **Boundaries of Order and Chaos**, planned at 12 FPS.

1. Open with a recognizable wide Mandelbrot or Burning Ship view.
2. Move into a detailed boundary using a controlled pan, zoom, and subtle rotation.
3. Introduce a second visual system—prefer Lyapunov or magnetic pendulum—to vary texture and mathematical meaning.
4. Animate a genuine system parameter in at least one scene, such as Lyapunov `color_scale` or magnetic positions
   and strengths.
5. Reuse a small, coherent color family across scenes, while adapting palette structure to each painter's output.
6. End on a balanced, memorable frame rather than stopping in the middle of a zoom.

This is a creative starting point, not a mandatory formula. Replace any scene that does not produce strong previews.

## Additional Creative Hints

- Presets are reliable starting points, not finished compositions.
- Favor motion along boundaries, filaments, and basin intersections; large uniform interiors rarely sustain a shot.
- Slow down near the most intricate structure and speed up only while crossing visually simple regions.
- Use rotation sparingly during deep zooms so that the viewer retains spatial orientation.
- For magnetic pendulums, try symmetric-to-asymmetric magnet arrangements, orbiting magnets, or one magnet fading in
  while another fades out. Keep magnet ordering fixed.
- For Lyapunov fractals, keep the sequence fixed within a scene and explore stable/chaotic boundaries by transform,
  `color_scale`, and carefully chosen iteration counts.
- For Mandelbroid or Julia formula morphs, test every intermediate formula. Token-compatible interpolation does not
  guarantee that the intermediate images are visually useful.
- Increase iteration counts during a zoom when needed to preserve detail, but avoid unnecessary values that make
  every frame expensive.
- A palette can evolve gradually within a scene, but preserve luminance contrast around important boundaries.
- Do not assume randomness creates variety. Use deterministic parameters in the final generation script so the
  project is reproducible.

If a new technique proves consistently useful while creating a project, add a concise, generally applicable hint to
this section. Keep project-specific notes in the project's description rather than in this guide.
