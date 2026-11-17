# Demo plan

Goal: a thing in a browser, at a URL, that makes the technique obvious and makes people want to
share it. Written 2026-11-17. Background in [IDEAS.md](IDEAS.md).

The demo has to earn two different verdicts at once — *is the technique sound* and *does it look
good* — so each stage below has a success criterion for both, and an off-ramp if it fails.

## The hero moment: fly into it and it keeps going

A shape refines as you approach, you fly inside, and its sub-branches turn out to be whole
structures of their own, forever. No mesh system can do that, and it is exactly the property this
method has for free — the LOD cut and the generator are the same mechanism. One continuous shot,
no cuts, no loading.

Morphing the maps over time is the second hero moment and goes in later (stage 3). It is the
better *beauty* demo and the weaker *technique* demo: continuous morphing is striking, but it
proves coherent animation rather than the LOD thesis. It also stresses the ordering harder than
anything else, since both camera and geometry move, so it is a good final exam rather than a
first test.

## Stage 0 — does approximate order actually look wrong? (half a day, offline)

Pure Python, extends `render_check.py`. No GPU, no browser.

1. Render a frame with an exact per-splat depth sort (ground truth).
2. Render the same frame with ordered expansion (per-node permutation from `M^T d`).
3. Diff the images; report max and mean per-pixel error.
4. Count inversions between the two orders, weighted by the product of the two splats' opacities —
   an inversion between two near-transparent splats does not matter, one between two opaque ones
   does.
5. Sweep camera direction over a few dozen angles so the number is not a lucky view.

**Success:** weighted inversion rate low enough that the A/B images are hard to tell apart, with a
number attached rather than an impression.

**If it fails:** fall back to the lazy k-way merge — descend further only where sibling
`[dmin, dmax]` intervals overlap. Costs more on ambiguous nodes only. Re-measure.

This stage decides whether the whole ordering idea survives, costs almost nothing, and needs no new
technology. It goes first for that reason.

## Stage 1 — one object, real time, in a browser (2–3 days)

WebGL2. Static object, orbit camera, 150–250k splats.

- Expand once on the CPU at load.
- Order per frame with ordered expansion: ~N/3 nodes, 9 compare-swaps each, fast enough in JS.
  Upload the result as an index buffer; the splat attributes never move.
- Instanced quads, covariance projected in the vertex shader, blending in primitive order with
  depth writes off.
- **Mip-Splatting-style covariance dilation by the pixel footprint from day one.** Not polish —
  without it, distant splats shimmer violently and every later judgement about how it looks is
  made against a broken image.

**Success:** 60 fps at 200k splats on a mid-range laptop, no shimmer while the camera moves.

**If it fails:** the budget is lower than hoped. Reduce depth, keep going; this changes the scale
of the demo, not its viability.

## Stage 2 — adaptive LOD and the infinite flight (3–5 days)

The real test, and where the hero moment comes from.

- Expand adaptively: recurse while projected size exceeds a threshold, with a hysteresis band so
  nodes do not thrash split/merge at the boundary.
- Moment-matched parents (mass, mean, covariance from the maps' fixed-point equations), plus an
  opacity cross-fade across the transition band.
- **Re-rooting, for precision.** Accumulated affines at depth 30+ destroy float32. Track which node
  the camera is inside and express everything relative to it, factoring that transform out — the
  same trick deep Mandelbrot zooms use. This is the main technical risk in the whole plan.
- **Address-hashed jitter.** `-j` currently draws from a sequential RNG, so a node's perturbation
  depends on how many nodes were generated before it. Under adaptive LOD, subtrees are generated
  independently and at different times, so this must become `hash(address)` or the world shimmers
  as things cross the LOD cut. Fix before this stage, not during it.

**Success:** a continuous zoom over six or more orders of magnitude with no popping and no
precision breakdown.

**If re-rooting proves hard:** cap the zoom at whatever depth is stable and make the demo a flight
*between* objects instead of *into* one. Still good; less singular.

## Stage 3 — a space of shapes, morphing (1–2 weeks)

Port to WebGPU for compute-shader expansion (WebGL2 has no compute; transform feedback can fake it
but is awkward). Then:

- Six to eight attractors floating in a dark void, each a different map set: branching, spiral,
  shell, lattice, something nearly flat, something nearly dust.
- Free flight between them, with the zoom from stage 2 available on any of them.
- Slow continuous morphing of each map set. Interpolating between two map sets morphs the whole
  attractor coherently, every splat moving smoothly, because everything derives from the maps.
  Nothing else in real-time graphics can do this.
- Depth fog so distant objects fade — looks good and hides LOD transitions at the same time.
- Colour by recursion depth. Free, and it makes the fractal structure legible rather than decorative.
- Geometric palette, not a forest: cold spectral, deep blue through cyan to white, or iridescent by
  address. Per [IDEAS.md](IDEAS.md), this is a crystalline constructed nature, not an attempt at one.

**Success:** someone shares it without being asked to.

## Hidden in the demo: the proof panel

A debug toggle that splits the screen — exact sort on one side, ordered expansion on the other,
with the live weighted-inversion count and both frame times. Means nothing to a general viewer,
means everything to a graphics person, and costs an afternoon once stage 0 exists.

## Publishing

Static page plus data; the site at `/srv/sites/mikaelcromsjo/splat/` already serves this way with
no service and no root. Note the pending CORS task in `root.md` if the page and the data ever end
up on different origins — same-origin here, so it is not blocking.

## Stretch, once morphing works

Drive the map parameters from audio. Cheap to add on top of stage 3 and very shareable.

## Rough total

Three to four weeks part-time to something worth showing. Stages 0 and 1 are the ones that decide
whether to continue, and together they are under a week.
