# New Painter Ideas

This document proposes five ordinary-precision painters that are visually and mathematically different from the painters currently available in Aldyparen. High-precision painters are deliberately out of scope.

Aldyparen's current ordinary painter API is point based: each input complex coordinate is classified independently and converted to a palette index. Proposals that fit this model are preferable because they retain interactive, chunked, and video rendering without requiring a second rendering architecture.

## 2. MagneticPendulumPainter

Treat every image coordinate as the initial horizontal position of a damped pendulum above three or more magnets. Numerically integrate the pendulum's motion until it settles near a magnet or reaches a step limit. Color primarily by the magnet reached and secondarily by settling time.

The boundaries between attraction basins become fractal. This is a physical dynamical system rather than a complex escape-time formula, and moving or changing the magnets can produce compelling videos.

Suggested parameters:

- `magnets`: JSON-compatible list of magnet positions and strengths.
- `height`: vertical distance between the pendulum and magnet plane.
- `damping`: velocity damping coefficient.
- `gravity`: restoring-force coefficient.
- `time_step`: numerical integration step.
- `max_steps`: integration limit.
- `settle_distance` and `settle_speed`: convergence criteria.

This painter fits the point-independent API because each pixel is one initial condition. It will be computationally expensive, so the integration loop should be Numba compiled and use a stable fixed-step method. Parameter interpolation should preserve the number and ordering of magnets; incompatible magnet configurations should be rejected when creating an animation.

Related background: [basins of attraction and strange attractors](https://en.wikipedia.org/wiki/Attractor) and [Wada basins](https://en.wikipedia.org/wiki/Wada_basin).

## 3. KleinianLimitSetPainter

Kleinian groups act on the Riemann sphere using Mobius transformations. Their limit sets can form nested circles, filaments, lace-like curves, and structures commonly associated with *Indra's Pearls*.

A practical first implementation can use a constrained family of paired Schottky circles. For each image point, repeatedly apply the circle inversion or inverse Mobius transformation that moves it toward a fundamental domain. Color by iteration count, the last generator used, or a combination of both.

Suggested parameters:

- `generators`: JSON-compatible Mobius-transform coefficients or a constrained circle-pair description.
- `max_iter`: maximum number of reductions or inversions.
- `tolerance`: boundary or convergence tolerance.
- `color_mode`: iteration count, generator identity, or both.

This can fit the point-independent API, but arbitrary transform coefficients may not define a useful discrete group. Constructor validation and a few known-good presets are important. Smooth video interpolation should initially be limited to a parameterized family known to remain valid between keyframes.

References: [Kleinian group](https://en.wikipedia.org/wiki/Kleinian_group) and [Indra's Pearls](http://klein.math.okstate.edu/IndrasPearls/).

## 4. DomainColoringPainter

Domain coloring visualizes a complex-valued function `w = f(z)`. The argument of `w` is represented by cyclic hue, while its magnitude controls brightness or contour bands. Zeros, poles, branch points, and conformal distortion all become visible.

Suggested parameters:

- `function`: an expression in `z`, using the existing safe function preparation machinery where possible.
- `phase_bins`: number of hue divisions.
- `magnitude_bins`: number of brightness or contour divisions.
- `magnitude_scale`: spacing of logarithmic magnitude contours.
- `grid_strength`: optional emphasis for phase and magnitude contour lines.

The painter can encode two dimensions into one palette index:

```text
index = phase_bin + phase_bins * magnitude_bin
```

This works with the current API but requires a matching structured palette to represent hue and brightness properly. The initial implementation may document the expected palette dimensions; a later improvement could let a painter provide a recommended palette or allow direct RGB output.

Functions with compatible token structure can use the existing numeric-expression interpolation approach, producing smooth coefficient-morphing videos.

Reference: [Domain coloring](https://en.wikipedia.org/wiki/Domain_coloring).

## 5. ApollonianGasketPainter

An Apollonian gasket begins with three mutually tangent circles and recursively fills every curved triangular gap with another tangent circle. Descartes' circle theorem determines the curvature of each new circle. The result is an intricate packing of nested circles with a fractional-dimensional residual set.

Suggested parameters:

- A constrained description of the initial three circles or an initial curvature triple.
- `depth`: recursion depth.
- `line_width`: circle-boundary width in world coordinates.
- `color_mode`: recursion depth, curvature band, circle identity, or filled/outline mode.

For efficiency, generate the finite circle list once in the constructor or lazily on the first render, then classify points against that list. The generated data must not appear in `to_object()`; only the constructor parameters should be serialized.

This is geometric rather than escape-time based. Zooming reveals successively smaller tangent circles, while animation can vary a constrained valid initial configuration, recursion depth, line width, or coloring. If smooth interpolation between arbitrary initial triples is unreliable, allow animation only when the structural parameters match.

Reference: [Apollonian gasket](https://en.wikipedia.org/wiki/Apollonian_gasket).

# Implementing a New Painter

## Painter Class

1. Add a module under `aldyparen/painters/` and define a class derived from `Painter` in `aldyparen/painters/base.py`. These proposals must use ordinary `numpy.complex128` coordinates and must not derive from `HighPrecisionPainter`.
2. Provide a no-argument constructor. Every constructor argument must be accepted by keyword, have a JSON-serializable default, and be validated with a clear error message.
3. Implement `paint(points, ans) -> None`. `points` and `ans` are one-dimensional arrays of equal length. Fill every element of the `numpy.uint32` output array in place with a non-negative palette index.
4. Keep expensive per-point loops in Numba-compiled functions where practical. Do not retain references to renderer-owned input or output arrays after `paint` returns.
5. Catch recoverable evaluation or numerical errors, set `self.warning` to a useful user-facing message, and leave `ans` fully initialized.
6. Implement `to_object()` so it returns exactly the JSON-compatible constructor arguments needed to recreate the painter. Do not serialize compiled functions, caches, NumPy arrays, or other derived state.

## Application Integration

1. Export the class from `aldyparen/painters/__init__.py` and append it to `ALL_PAINTERS`. This also adds it to `PAINTERS_INDEX`, project deserialization, the GUI painter selector, and default painter configuration handling.
2. Verify that selecting the painter in the GUI shows valid default JSON and that editing this JSON reconstructs and rerenders the painter. The GUI passes the object returned by `to_object()` back to the constructor as keyword arguments.
3. Decide explicitly how animation should behave. If different configurations can be mixed, add a painter-specific branch and interpolation function in `aldyparen/mixing.py`. Interpolate continuous numeric values, round discrete numeric values deliberately, and reject incompatible structural parameters with `ValueError`. If mixing is not supported, equal painters will still work, while differing painters must continue to fail clearly.
4. Add one or more useful defaults or presets when the bare constructor does not demonstrate the painter well. Presets must use a suitable transform and palette and must remain JSON serializable.
4.1. Add menu item in "Presets" menu in main.xml for added preset and link it to 
showing new preset, like it's done for other presets.
5. Ensure the painter works through `StaticRenderer`, `ChunkingRenderer`, and `InteractiveRenderer`; these renderers may split the coordinate array into chunks, so output must not depend on chunk boundaries, processing order, or mutable global state.
6. Add painter description to README.md, inlcuding references.

## Required Tests

Every new painter must include tests. At minimum, add the following coverage:

1. **Construction and serialization:** the existing `test_defaults` test exercises every class in `ALL_PAINTERS`. Confirm that the new painter can be constructed with no arguments, that `json.dumps(painter.to_object())` succeeds, and that `PainterClass(**painter.to_object()).to_object()` is identical.
2. **Validation:** test invalid parameter values and malformed expressions or structures. Assert useful error messages, not only the exception type.
3. **Focused algorithm tests:** call `paint` on a small array containing points with known classifications. Assert the exact `numpy.uint32` output, including boundary, convergence, divergence, or step-limit cases relevant to the algorithm.
4. **Failure handling:** where runtime numerical or expression failures are possible, verify that `warning` is set and the entire output array receives deterministic fallback values.
5. **Animation:** if interpolation is supported, add tests to `aldyparen/mixing_test.py` for both a known midpoint and rejected incompatible configurations. If it is intentionally unsupported, test that differing configurations raise `ValueError`.
6. **Renderer compatibility:** for algorithms that use caches or preprocessing, verify that chunked and unchunked rendering produce the same image.
7. **Golden image test:** render a small, deterministic representative picture with `StaticRenderer` and compare it with a stored BMP image by calling `_assert_picture` from `aldyparen/test_util.py`. Commit the approved image as `goldens/<descriptive_name>.bmp`. The test must fail when the golden is absent or pixels differ; use `max_mismatched_pixels` only when a small, justified platform-dependent tolerance is unavoidable. Generate or update a golden deliberately with `_assert_picture(..., overwrite=True)`, inspect the resulting image, then restore the test to normal comparison mode before committing.

A typical golden test has this shape:

```python
def test_renders_new_painter():
    renderer = StaticRenderer(200, 200)
    painter = NewPainter(...)
    transform = Transform.create(...)
    palette = ColorPalette.gradient("white", "black", size=...)
    frame = Frame(painter, transform, palette)

    _assert_picture(renderer.render(frame), "new_painter")
```

Run the focused painter and mixing tests while developing, then run the complete validation commands before considering the integration finished:

```sh
./lint.sh
python3 -m pytest .
```
