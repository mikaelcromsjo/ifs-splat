# Where this got to — 2026-11-17

Against [PLAN.md](PLAN.md). Everything below was measured, not estimated. **No GPU code has ever
been run**: installing a headless browser was denied, so there is no WebGL/WebGPU execution,
screenshot or frame time from a real device anywhere in this work. Every image in `web/out/` comes
from `web/js/cpu_raster.js`, a CPU transcription of the shader.

## Stage 0 — done, verdict: ordered expansion is good enough

- Weighted inversion rate ~14.7% mean over 64 camera directions (`py/stage0.py`).
- **That number is a poor predictor of image error.** The honest verdict came from motion
  (`py/stage0_motion.py`): ordering flicker is ~1/3 of the camera's own frame-to-frame change at
  every percentile including the single worst pixel. Mean error 0.36/255.
- Two IDEAS.md claims tested and **disproved**: inversions are *not* buried under opacity
  (visible-weighted 16.0% ≥ opacity-weighted 15.5%); hierarchical ordering is *not* more temporally
  stable (it is ~5× less stable by pair-churn).

## Stage 1 — done

WebGL2, instanced quads, covariance in the vertex shader, Mip-Splatting dilation from the start.
Ordering 3.00 ms/frame at 152k splats in Node. Python↔JS parity is bit-exact for geometry, colour
and draw order (`py/parity_check.py`); the GLSL math matches the Python reference at 64 dB.

## Stage 2 — done, all three criteria met

- Depth 47, **10.5 orders of magnitude**, 47 re-roots (`web/tests/flight.mjs`).
- Re-rooted float32 positions **exact**; world-coordinate float32 carries 3.9% of a splat radius of
  error at that depth.
- Cross-fade band gives **~30× fewer visible pops** than a hard cut (`web/tests/pop.mjs`).
- **LOD fidelity measured against ground truth** (`web/tests/fidelity.mjs`): 26–44 dB at 2–12×
  fewer splats. Sparse sharp structures (lattice, plane, spine) sit at ~26 dB — a Gaussian cannot
  represent a sharp-edged void. Dense ones reach 35–44 dB.
- Blob opacity 0.95 was checked against ground truth rather than guessed: a sweep showed PSNR rises
  monotonically to 0.95, so the murky look of the tree's canopy is the *object*, not the
  approximation — the 900k-splat ground truth is just as murky.

## Stage 3 — mostly done

- **Seven map sets** (`web/js/maps.js`), parametric so any pair interpolates validly:
  tree, spiral, shell, lattice, plane, dust, spine. All validated by `web/tests/mapset.mjs`
  including all 21 morph pairs × 11 steps.
- **The void** (`web/js/world.js`): placements, a camera that keeps its *address* rather than a
  world position, and a coordinate chain verified exact to 2e-16 at depth 0 and depth 8
  (`web/tests/void.mjs`).
- **Morphing**: `Attractor.reparam()` rebuilds moments + child table in 0.97 ms (was 1.67 ms before
  making the fixed points allocation-free and convergence-based).
- **Palettes**: cold spectral, ice, ember, iridescent, violet — colour by recursion depth, cyclic.
  `paletteBleach()` measures what the blob fixed point costs: 8% for a 12-entry ramp, 17% for a
  24-entry hue wheel. (The tree's olive canopy is *not* this effect — that palette keeps 92%.)
- **Autopilot**: cruise → dive → climb out backwards. Three real bugs found and fixed by
  `web/tests/tour.mjs`, which flies the demo headless and photographs it:
  1. `cruise` moved a fraction of the remaining distance along whatever way the camera faced, so
     after a climb it flew *backwards* and the void ended up 380,000 units behind. 2875 blank
     frames of 3600.
  2. `dive` aimed at the child straight ahead rather than the one with mass, so in sparse
     attractors it steered into gaps.
  3. The climb turned to face outward (black screen) and ascended ~16 frames/level while the dive
     descended in 2, so depth ran away to 125 and never came back. Now it reverses without turning
     — which also makes the ascent the hero zoom played backwards.
  After: all 7 attractors visited, depth to 65, 127 blank frames of 4500 (2.8%, turn-arounds).
- **A latent trap, found and closed.** (Raised by the agent that designed the map sets, which
  stopped mid-fix believing its `mu0` was inconsistent with the moments. Investigated: all seven
  sets are in fact consistent -- every trunk map has zero tilt/roll/twist and every `mu0` lies on
  z -- so nothing needed fixing in `maps.js` and the agent does not need resuming. What the
  concern was right about is that nothing was *enforcing* it.) The traversal computes a rod's position as
  `A * (0, 0, trunkA[8] * mu0[2]) + b` and its covariance from `trunkA`'s diagonal alone -- three
  multiplies per node instead of a 3x3 compose -- while `invariantMoments` uses the full `trunkA`
  and the full `mu0`. The two agree only while the trunk map is diagonal and `mu0` lies on z. That
  is true of all seven sets, so nothing is wrong today, but a future set with a tilted trunk would
  have rendered something quietly different from what its own moments describe. `Attractor.reparam`
  now refuses such a set with an error that names both remedies.
- **`tests/reroot.mjs` was measuring the wrong thing.** Its thresholds were absolute, in local
  units, so it became a different test at every depth. Once `dive` started aiming at mass it
  reached depth 52 instead of ~30 and the test "failed" at 2.9e-4 -- which, multiplied by the local
  scale, is 8.4e-16: one float64 ulp *in world units*. That is not an error, it is the argument for
  re-rooting, stated as a number. The test now normalises by the local scale and says so.
- **Proof panel**: split screen, both frame times, and a live weighted-inversion count
  (`web/js/inversions.js`), checked against the offline numpy one to 3e-4 percentage points
  (`py/inv_parity.py`; not bit-exact because the browser's ground truth is a 16-bit counting sort
  over float32 keys).
- **WebGPU compute expansion**: WGSL kernel written (`web/js/gpu_kernel.js`) with a line-for-line
  JS twin (`web/js/gpu_ref.js`). `web/tests/gpu_transcribe.mjs` confirms it reproduces the CPU
  expansion exactly — **0 wrong draw slots out of 114,687**, colours exact, geometry to one float32
  ulp. One pass, no atomics, no compaction: the order falls out of the same walk, so the per-frame
  CPU cost of the object is zero. A pleasant find: because every branch map is a similarity, the
  foliage disc's eigen-decomposition has a closed form, so the GPU needs no Jacobi solver —
  verified equal to the eigensolver to 6e-8.

## Stage 4 — an endless field, generated on the fly

The void was seven attractors on a ring. It is now a field with no edges: space is cut into cells
of side `L = 10`, and a cell's whole contents — whether it is occupied at all, where inside the
cell, how big, which way up, which pool shape, its colour tint, its root hash — are a pure
function of its integer coordinates under the same `h32`/`mix`/`stream` discipline that already
made splats reproducible. Nothing is stored, so eviction is free; fly back and the same objects
are there, bit-for-bit (a SHA-256 over every field of every record in an 18³ box matches between
two fresh processes).

Three things made it affordable, and each was a measured problem before it was a design:

- **A pool of eight shapes, not one per object.** `Attractor.reparam` costs ~1.2 ms, and the old
  `World` rebuilt every item — fine for seven, forty milliseconds a frame for a field. The pool
  makes that cost fixed: at most one rebuild per frame under a 1.5 ms budget, whatever is in view.
  Two cells on the same slot are not copies, because each supplies its own **root hash**, which is
  the root of the address-hashed jitter — the same map set walked from a different hash is a
  different realisation of the shape, different branch angles the whole way down, for no memory
  and no rebuild. Scale, orientation and a per-cell colour tint do the rest.
- **Non-overlap by arithmetic, not by rejection.** Draw order *between* two attractors comes only
  from the order they are streamed, which is valid while their bounding spheres are disjoint. With
  reach capped at `ρL` and the position jittered within `L(½−ρ)`, two objects in different cells
  are provably separated. The cap uses the worst `|C|+R` over *all* map sets and all morph pairs
  (`REACH_MAX = 3.7537`), not the current one, because `R` changes as a shape morphs. Swept over
  125,000 cells: 190,995 neighbouring pairs, zero intersections, worst margin 0.48 world units.
- **Allocation discipline.** Written the obvious way, `frames()` allocated a matrix, four vectors
  and an object per visible object per frame. At seven objects that is noise; at forty it is two
  thousand allocations a second and the collector answers with pauses — a settled frame measured
  at 33 ms. Making `frames()` and `rootFrame()` allocation-free took the mean rebuild cost from
  1.99 to 1.18 ms/frame.

Measured, 40 s flown headless at 260×200 (`node tests/tour.mjs 40 out/f.png 260 200 field`):

| | field | the old ring |
|---|---|---|
| objects in view | mean 29.5, min 15 (of ~340 resident) | 7 |
| blank frames | **0** | 32 |
| morph rebuild | 1.14 ms/frame, p99 2.51 | 2.07 ms/frame, p99 5.39 |
| worst of the first 30 frames | 2.50 ms | 61.53 ms |
| whole frame, CPU | 41.3 ms | 42.9 ms |

The last row is the point: **a field of thirty objects costs the same as the ring of seven.** That
40 ms is expanding ~90k splats under a splat budget, not the object count — the per-node frustum
and occlusion culling in `visit()` was already doing the work, and an off-screen object costs one
early-returning call. In the viewer the budget is time-based, so it trades splats for frame rate
instead.

**Ordering in a crowd was the one that could have failed honestly.** Within an attractor the order
is exact; between attractors it is a sort by nearest point of the bounding sphere, a heuristic
that had never been measured with more than seven objects. `tests/order_field.mjs` measures the
whole frame against an exact per-splat sort: **7.68% mean opacity-weighted inversion rate with
25–38 objects in view, against 9.60% for the seven-object ring in the same run** — better, not
merely acceptable — and the image difference from a true global sort is 0.015/255, invisible.
(Run to run the pair moves around 8–9% either way; the ratio is the stable part.) The separation guarantee
is what buys that.

**The biggest remaining gap is aesthetic, and it is older than this stage.** Measured over a
900-frame autopilot flight, **28.2% of frames are a near-uniform wash of one colour** — the camera
is inside a structure, and at that range a Gaussian field has no silhouette to read. The old ring
is *worse* at 42.0%, so the field improved it rather than caused it (more to look at, less time
spent deep inside one thing), but a quarter of the flight still being featureless colour is the
thing standing between this and the plan's stage-3 criterion. The fix is to the autopilot, not the
renderer: the hero shot is the approach, and the dive currently keeps going well past the range
where structure reads. See `IDEAS.md`.

Two further caveats recorded rather than smoothed over. `REACH_MAX` has 1.03e-5 of headroom and the
measured support sits at 0.99638 of reach — both thinner than comfortable, so a new map set could
break the separation guarantee; `tests/field.mjs` checks 2 and 5 catch it. And the 3σ hull of the
Gaussians is ~16% larger than the support they stand for, so in the very worst pair the faint
tails can graze; that was already true of the ring and is not something the field introduced.

## Stage 5 — a breathing universe

Four complaints, all fair: the morphing stopped and started instead of running continuously; the
rate was constant where it should breathe; every object was a variation on a tree; and the world
was an evenly spaced lattice rather than a universe. Steps 1-3 below are done and gated.

### Jitter became a parameter, and the table shrank

A node's jitter rotation used to be baked into a lookup table at construction. That made `jitter`
the one map parameter that could not morph, so a rigid crystal and an organic tangle could never
morph into one another. It is now assembled during `reparam` from a cached axis and standard
normal, and the table went from 4096 entries to 1024.

- **Rebuild cost 1.388 ms -> 0.44 ms**, which is what pays for morphing continuously rather than
  in bursts. Two rebuilds now fit the 1.5 ms frame budget where one did before.
- The assembled table is **bit-exact** against the shared reference `jitterRotInto()`, checked
  over 193,536 entries across 3 table sizes and 4 jitter amounts (`tests/jitterbits.mjs`).
- `py/parity_check.py` is untouched and still bit-exact, as predicted: Python computes the
  rotation per node with no table at all, so the table was only ever an adaptive-path optimisation.
- 1024 entries show no visible repetition. Even 256 (x127 reuse) does not — see
  `web/out/jitterbits.png`. The images are different *realisations*, not better and worse ones, so
  the test asserts only that brightness and contrast do not move, and leaves the judgement to a
  person.

### The morph breathes, and never stops

The chain of hold-then-crossfade waypoints is gone. The path is now a closed Catmull-Rom spline
through the map sets, driven by a breathing **rate** rather than a breathing position: a raised
cosine between 0.18x and 1.90x, period 26 s. The distinction matters — the holds were the only
thing making a shape legible, and a constant rate along a smooth path would be perpetual mush. A
rate that falls almost to rest gives the legibility back without putting the corner back.

- **Velocity across a waypoint: 0.053% discontinuous, against 44.5% for the old chain** — 843x
  smoother at the joins (`tests/breath.mjs`).
- Quantisation is in distance travelled, not time elapsed, so the breath modulates rebuild demand
  for free: **0.46 rebuilds/frame while slow against 1.48 while quick**. The legible moments are
  the cheap ones.

Three real defects surfaced here, none of which would have been visible as a crash:

1. **`sig0` went negative** (-0.021) partway through a morph — a negative variance is a degenerate
   splat, not a small one. The first guard, `Math.max(1e-9, ...)`, would have hidden it. Strictly
   positive quantities now spline in log space: positive by construction, C1, and the right
   geometry anyway (halfway between 0.0016 and 0.39 should be their geometric mean).
2. **An exact 180-degree tie.** `roll` on map 3 steps exactly 180 twice in the loop, where "the
   short way round" is genuinely ambiguous — and the helper returned -180 for *both* directions,
   so the tangents disagreed across the loop's wrap. A visible lurch, twice per cycle. Worse, the
   first fix compared raw values, which the unwrap chain shifts by multiples of 360; the tie-break
   has to depend only on the residue.
3. **Clamps put corners back.** A clamp is flat while active and sloped the instant it is not, so
   any set sitting exactly *on* one gets a velocity jump. The scale clamp's bounds also changed at
   every waypoint. Scale needed no clamp at all once measured (0.807 over the whole loop, against
   a contraction limit of 1); `omega` and `jitter`, which are legitimately *zero*, spline as the
   square of a splined square root — non-negative by construction, with no clamp anywhere.

### Eleven sets, and four of them are not trees

`omega` is the trunk's mass fraction — the "how tree-like is this" knob — and it had never been
turned: all seven original sets sat between 0.15 and 0.38, every one with the same jitter of 0.16.
That is why the field read as a wood whichever set a cell drew.

Added, all with `omega = 0` and little or no jitter: **sierpinski** (the tetrahedron, exact —
contraction 0.5, R = sqrt(3)), **whirl** (the same skeleton turned, chiral), **fold** (a dragon-
family folded curve), **cage** (four maps closing into a shell of struts). No engine change was
needed for any of them; the knobs already existed. See `web/out/catalogue.png`, and
`web/out/spline.png` for a tree dissolving into a gasket.

The loop order alternates organic and crystalline, which is not only aesthetic: an earlier
ordering put the two highest-contraction sets adjacent, and because R grows like
1/(1 - contraction), the spline's overshoot between them reached **5.12 against a REACH_MAX of
3.75** — enough to make neighbouring objects intersect and put a sorting seam through both.
Interleaving brought it to 3.92. `REACH_MAX` is 3.95, and `tests/field.mjs` re-measures it over
the spline so that a new set *or a new order* fails loudly.

### A hierarchical universe

The flat grid is gone. A cell now either holds one object, **subdivides** into 8 children of half
the side, or is **empty** — chosen by its own hash, to a depth of 3 levels. A cell that holds an
object never subdivides, and that single rule makes the separation proof *shorter* than the flat
one: cells at any two levels are nested or disjoint, no object's cell contains another's, so every
pair of object-holding cells is disjoint, and each object fits inside its own cell by the same
`hw + rho*L = L/2` arithmetic applied at that level's own side length.

- **66,818 objects over 64,000 root cells, 983,142 pair-tests, 619,519 of them cross-level,
  120,599 inside a single root cell: 0 intersecting.** Worst margin 0.139 world units.
- Objects per root cell matches the branching process (`density + 8*P_SUB*E[below]`) to **1.10%**.
- **51.2% of root cells yield nothing** — real voids, which is what the flat field never had.
- Each level is exactly 2× smaller than the one above (measured ratios 2.000, 2.000), giving an
  **11.4× size range** where the flat field had essentially one size.
- Draw order **improved**: 7.52% mean weighted inversions against 8.07% flat. Clustering puts
  objects closer together and did not hurt it.
- **0 blank frames** on the tour despite real voids, 57 objects in view against ~30, and whole-frame
  CPU went *down* to 36.4 ms from 37.1.

One bug worth recording, because it looked like a design failure rather than a defect: the
per-root-cell cap was tested against the length of the caller's output array, so a caller that
accumulated into one array had the entire field truncated at 24 objects — 24 objects across 24,389
root cells, which reads as an empty universe, not as an off-by-one.

### Placement, size and orientation as functions of time

Objects drift, breathe and turn continuously. This is free of any separation risk, and by
construction rather than by luck: the proof bounds the **envelope** — every position and size an
object can ever take — not its instantaneous value. The drift amplitude is drawn from what is
*left* after the static offset (`|base| + amplitude <= 1` in units of the cell's wander
half-width) and the size breathes *downward* from the cap, never up. So the envelope is exactly
the sphere the static field already fitted inside the cell. Rotation is free: reach is measured
about `t` and is rotation-invariant.

- **1,556,516 pair-tests across 481 instants spanning 240 s: 0 intersecting.** Worst margin 0.418.
- Objects drift up to 1.65 world units and breathe 30% of their size.
- `Placed.animate()` writes `t`, `s` and `R` in place and allocates nothing — stage 4's lesson was
  a 33 ms frame caused by allocating per object per frame.

### Still open in this stage

The hierarchical universe (clusters, voids, a power-law of sizes), placement and size as functions
of time, and an autopilot that turns at peak legibility rather than diving to `maxDepth` — 28% of
stage-4 frames were a washed-out fog bank. `tests/legible.mjs` measures that and gates it at 10%.

## Stage 6 — a slower flight that follows edges, and a sky to fly through

Four asks: slower and softer autopilot, follow edges and go deeper, smaller morph steps, and make
it look more like fantasy space travel. All four are done and measured. Two bugs turned up on the
way, and one of them was in the instrument.

### The probe was measuring the background

`tests/legible.mjs` calls a frame washed out when it is *bright* (`lit > 0.45`) and has no spatial
detail (`grad < 3.0`). `lit` counted pixels over an absolute 8/255 — but the rasteriser's flat
clear colour is `[0.030, 0.035, 0.047]`, which comes out at 10–16 after gamma. **Every pixel of an
empty frame counted as lit.** So a crossing with nothing on screen scored `lit = 1.00`, and 99.6%
of cruise frames were reported as a fog bank when what they were was empty space.

The two faults want opposite fixes — dive less far in, versus aim at something on the way over —
so a measure that cannot tell them apart is worse than none. `lit` is now counted against the
background, and the probe reports *nearly empty* alongside *washed out*. The stage-4 figure of
28.2% is not comparable with anything below; re-measured with the corrected probe, the flight as
it stood was **26.1% washed out and 19.0% nearly empty** over 120 s.

### The dive was flying out of the attractor

`FlyCamera.dive()` closes a fraction of the distance to the child it has chosen, along the
heading. If the heading points away from that child — which happens after a sharp turn — the
camera moves away, the distance grows, and the next step is bigger: a geometric series the wrong
way round. `VoidCamera.cruise()` has had a guard against exactly this since stage 3 (`align`,
"2875 blank frames to none"); the dive never got one.

Left to run, a bare dive reaches **3e14 local units** from the attractor it was diving into.
`tests/reroot.mjs` had been printing `depth reached 0` for exactly this reason, and still passing,
because both of its assertions are on *errors* and errors are smallest when nothing happens. With
the guard, the same test reaches depth 22; it now asserts depth as well. In the full flight the
fault was masked — the autopilot's climb and retarget phases keep hauling the camera back — but it
is most of why the old dive spent so much of its time looking at a blur: it was never where it
meant to be.

Over an ensemble of 10 map sets x 6 seeds x 600 dive steps, the guard takes the fraction of frames
spent inside the attractor from **0.28 to 0.95** and the median depth reached from **3 to 24**.

### Edges, not mass

The dive scored each child by how big it would look and aimed at its centre. Both halves aim at
the middle of the mass, and the middle of a Gaussian field has no silhouette in it at any depth —
which is what a washed-out frame is a correct picture of.

Two changes, one knob each:

- **`edgeBias` (0.55)** leans the child score toward the children that sit furthest out of the
  node, ranked against their own siblings rather than against an absolute distance. It only bites
  where the children differ: on `lattice` it moves the mean outwardness of the child chosen from
  0.747 to 0.972, and on a Sierpinski tetrahedron, where every child is equally far out, it
  changes nothing at all and says so (`tests/pilot.mjs`).
- **`edgeAim` (0.52)** pushes the aim point out along that same direction, by half the child's own
  radius, so the approach grazes the boundary instead of plunging through it. It has to stay under
  `descendMargin` or the camera would circle the shell and never descend — that is asserted, not
  assumed.

**`edgeSlide` is off, and the number is why.** Curving the path as well as the aim sounded like
more of the same good thing; at 0.30 it made the dive-only washed-out rate *worse*, 33.7% against
30.0%. It is kept, switched off, with the measurement in the comment.

### Deeper is not free, and the climb was the worst part

Washed-out frames sit at a mean depth of 34 against 19 for legible ones, so depth does cost
legibility even with the dive following the boundary. `maxDepth` went 46 -> 56 rather than to the
64 first tried.

The surprise was the climb: **81% of climb frames were a wash**, against 21% of dive frames in the
same run. Backing out along the entry path keeps the camera in the middle of the structure the
whole way up. It now arcs — one lateral direction drawn per climb, held for the whole ascent, with
the backward component untouched so it still costs the same frames per level.

### Slower, softer

| | was | now |
|---|---|---|
| `speed` (the geometric time constant) | 0.0105 | 0.0062 |
| cruise / dive / climb seconds | 34 / 44 / 5 | 44 / 72 / 8 |
| dive heading follower | 0.25 | 0.14 |
| cruise heading follower | 4.0 | 2.4 |
| dive/climb reversal ease | 1.3 s | 2.4 s |
| challenger margin before changing child | 1.8x | 2.2x |

Two things that were not a matter of taste:

- **The browser never flew the flight that was measured.** `frameFlight` passes `flySpeed` into
  `pilot.step()`, which overrides `Autopilot.speed`; `flySpeed` defaulted to 0.02 while every
  headless test used the Autopilot's own 0.0105. The viewer now opens at the autopilot's speed.
- **The forward step is now `1 - exp(-speed)`, not `speed`.** It is documented as a time constant,
  and as a bare fraction it was not one: closing 5% twice leaves 0.9025, not 0.90. With the
  steering follower also taking the step it is given instead of a hardcoded 1/60, half the frame
  rate now flies the same flight — 0.26 degrees and 1% of a radius apart after half a second.
  Only the near term: the dive's child choice is a discrete decision inside a self-similar
  structure, and a perturbation of one part in a million in speed puts the heading 60 degrees away
  three seconds later. That is the system, not a fault, and `tests/pilot.mjs` reports it so that no
  single dive's outcome is ever mistaken for evidence.

### Smaller morph steps

`setSeconds` 9 -> 16 and the breath 0.18–1.90 -> 0.10–1.15, over a 40 s period. The step a rebuild
takes went from a mean of **1/171 of a segment to 1/463**, and from a p99 of **1/71 to 1/194** —
the p99 is the one that matters, because it is the largest jump a viewer actually sees. It costs
less rather than more: the rebuild quantum is distance travelled, so a slower morph asks for fewer
rebuilds (`tests/breath.mjs`: 0.58/frame, 0.44 ms against a 1.5 ms budget).

### The morph runs at the display's rate

Smaller steps made each jump smaller; they did not make the morph *smooth*, because they did not
change how often the drawn shape changes at all. A rebuild costs 0.44 ms and the budget affords a
couple a frame across eight slots, so each slot's shape was redrawn 18–26 times a second against a
60 Hz display. Measured: **61% of slot-frames drew exactly what the frame before drew.** That is a
20 fps animation inside a 60 fps flight, and no amount of rebuilding harder fixes it — eight slots
at 60 Hz is 3.5 ms a frame of reparam against a 1.5 ms budget.

The way out was already half-built and switched off. The premultiplied child table is *linear* in
the maps — `childMaps[k][i] = J[k] @ A[i]` — so `lerp(J@A0, J@A1) = J @ lerp(A0, A1)`: blending two
built tables is exactly premultiplying the blended maps, at a tenth of the cost of building one.
The old code materialised that blend into a third table every frame, which was correct and
unaffordable: 0.075 ms of arithmetic per slot that moved the frame's p99 from 103 ms to 564 ms,
because rewriting 288 KB evicts the cache the tree expansion is working in. So it shipped disabled,
with a comment saying it belonged on the GPU.

It belongs in the read. The traversal now reads 9 floats from each of two tables and lerps them,
instead of 9 from one table that someone rewrote first. Nothing is written, nothing is invalidated,
both tables stay resident. **Measured on one frame of 140k splats drawn both ways: +3.8% on the
expansion**, against a morph that went from 20 Hz to the display's own rate.

| | rebuild-rate morph | read-time interpolation |
|---|---|---|
| drawn shape changes, per slot | 25.6 Hz | **59.7 Hz** (60 Hz display) |
| slot-frames identical to the one before | 60.8% | **0.7%** |
| median step in the drawn shape | 3.0e-3 | **1.2e-3** |
| p90 step | 5.3e-3 | **2.4e-3** |
| p90 / median | 1.8 | 2.0 |

Two things had to be got right, and both were wrong first:

- **It extrapolates, it does not lag.** The fraction is `(u_now - uA) / (uB - uA)` over the
  parameters of the two most recent builds, so it normally sits between 1 and 2 and the drawn
  shape is at the parameter the morph is actually at. Clamping it at 1 — interpolating *between*
  two builds and waiting — leaves the shape standing at the newest build whenever the next rebuild
  is late, which the budget makes routine: 24 Hz, barely better than no interpolation.
- **The two tables must be two real builds.** The first version folded the drawn shape into the
  older table at each rebuild, to make the hand-over continuous. It does, and it destroys the
  measurement: the two samples then sit a hair apart in parameter and the slope between them is one
  tiny number over another. 10% of frames still stepped as far as they had with no interpolation.
  Swapping instead leaves a kink at each hand-over that is second order in the rebuild interval —
  the same order as the error of reading the curve as a straight line in the first place.

The small derived state (translations, per-child scales, the trunk map, the rod's moments, and the
LOD metrics `C`, `R`, `qMax`) is interpolated too, about sixty numbers a slot a frame. A table that
moves smoothly while the metric it is cut against jumps at 20 Hz is a different judder, not less.

Extrapolating that state is where it can go wrong, and it did. Interpolation between two valid
builds is safe for nothing: every quantity is a convex combination, so a variance positive in both
is positive all the way across. Running *past* the newest build is not safe, and over 200 s it put
`sig0` at **−0.0204**, a child scale at −0.047 and `R` at −0.42 — a negative variance, the same
fault `tests/breath.mjs` was written for when a clamp was hiding it in the spline. Two things fix
it, and both are needed:

- **The transient at the source.** The one pair of builds that is *not* a rebuild apart is the
  warm-up's pure set followed by the first spline build, and a fraction of 1.33 through that pair
  lands a third of the way past a shape nothing ever meant to reach. The reset now clears
  `_everBuilt`, so that rebuild is its own previous build and interpolation starts with the second.
- **A bound from the data, for the general case.** For every quantity that must stay positive, the
  line through the two builds reaches zero at `A / (A - B)`; `Attractor._btLimit()` allows 90% of
  the nearest of those, computed once per rebuild because both builds are fixed in between. Over
  200 s of flying it never binds (its smallest value is 14.6 against a fraction that reaches 2.5),
  which is the point: it is the guard, not the mechanism.

The fraction itself sits at a median of 1.33, p90 1.73, p99 2.01, and is held at its 2.5 cap in
0.29% of slot-frames — so the shape is a third of a rebuild ahead of the last build most of the
time, and essentially never pinned.

#### Both of these are switches in the panel now, with the rate beside them

The claim in the table above is the kind that is easy to make and hard to show, so the demo makes
it on screen: a *morph* section with **morphing** and **interpolate**, and two HUD rows that count
what they do. `morph fps` is how often the drawn shape actually changes; `rebuild fps` is how
often it is rebuilt. Unticking **interpolate** collapses the first onto the second, live.

The rate is counted, not inferred: a slot-frame counts as moved if that slot was rebuilt or its
blend fraction changed, which is the same event `tests/breath.mjs` counts offline, accumulated
over a half-second window. At the demo's own settings `tests/toggles.mjs` measures 57.7 drawn
changes a second from 2.7 rebuilds a second with interpolation on, and 7.0 from 7.0 with it off.

Writing the switches turned up two faults in the switching itself, neither visible from the flag:

- **Switching interpolation off snapped the picture backwards.** The blend extrapolates, so the
  drawn shape normally sits a third of a rebuild *past* the newest build; collapsing the pair to
  that build moved 8 slots of 8. `Attractor.setInterp` now folds the live blend into the newest
  build before collapsing, so the switch changes the motion without changing the frame it is
  clicked on. That is asserted directly: snapshot the drawn table either side of every flip, zero
  slots may differ.
- **The meters carried half a second of the old behaviour across the switch** -- which is exactly
  the half second someone who just clicked is reading. Both switches reset them.

Turning interpolation back on has a third hazard that was designed out rather than found: while it
is off, `reparam()` stops swapping the tables, so the older one holds whatever build was current
when it was switched off. Lerping from that would walk the shape back through a morph it already
did, and extrapolating from it is exactly the negative-variance case above. The pair is collapsed
on the way in, so the blend is a no-op until the next rebuild gives it two real builds again.

And a fourth that was not designed out, found by flying with the switch: the small derived state
-- the moments, the bounding radius, the child scales, the LOD metric, the map translations -- is
snapshotted under that same flag, by `reparam` itself. So while interpolation is off it stops
being taken, and the collapse above then scattered a snapshot that was a build old over a table
that was current. The drawn table was exactly right and everything that reads the shape *around*
it was a build behind: all 8 slots jumped at the click, `R` by as much as 1.37 and sigma0 by 8x,
which is the hard kind of wrong -- the geometry drawn at another build's level of detail, with its
children offset from where the camera thinks they are. Nothing put it right until that slot's next
rebuild re-gathered, and with the morph held, nothing ever did. `setInterp` re-gathers the live
state when it is the one turning interpolation on, and `tests/toggles.mjs` now snapshots the
moments, radius, scales and translations across every flip as well as the table.

#### The parameter was right and the picture still jumped

Everything above measures the morph as a parameter or as a table entry, and a viewer sees neither.
A splat's position is the premultiplied table composed down an address, times the interpolated
moment centre, plus the interpolated translations -- three blends in a product, and a product of
blends is not the blend of a product. `tests/drift.mjs` walks fixed addresses the way `visit()`
does, through the drawn tables at the drawn fraction and through an exact build at the same morph
parameter, and differences the two points. It was red:

- **17 slot-frames of 10,552 stepped more than 8x as far as the true splat moved**, up to 36x, and
  every one of them was a rebuild frame with the fraction at exactly 1.00.
- 7 moved *against* the morph; the worst position error was 9.6e-3 of the object's radius.

Three separate causes, none of them visible in the quantity it was written in terms of:

1. **The build was made at the parameter the morph had already reached.** That makes the newest
   sample the present, so the blend can only extrapolate -- and it puts the hand-over from one
   secant to the next at the far tip of the old one, where two lines disagree most. The fraction
   resets to `(u - uA)/(u - uA)` = 1 exactly, so whatever had been extrapolated is discarded in
   one frame. Building one schedule *quantum ahead* puts the pair either side of the present: the
   fraction runs 0 -> 1 between rebuilds and the hand-over lands on the sample both lines share,
   where they agree exactly. The drawn parameter does not move -- the secant is still read at `u`
   -- so nothing lags; only the spacing of the samples changes.
2. **A table resize threw the blend's pair away.** Two identical tables draw the same shape at
   every fraction, so the slot stopped moving for a whole rebuild interval, 150 times a minute, on
   whichever object had just become the most important one on screen: 9.4e-3 at p99 within 40
   frames of a resize against 6.0e-5 for a settled pair. It needs no stashed state to keep --
   at the top of `reparam` the newest build is still in `_cmB` and the parameters it was built
   from are still in `this.maps`, so a shrink needs nothing at all and a grow needs only its new
   entries filled.
3. **The resize also forced a rebuild immediately**, a quarter of a quantum after the last one,
   while the older sample still sat a whole quantum ahead of the present -- a fraction of -3.06.
   Clamping that is a freeze, and releasing it is a step. The resize now rides along with the
   slot's next ordinary rebuild, which is what `setTableBits` always claimed it did.

Two smaller things fell out of it. The lead must come from the schedule's **quantum**, which is
exact, and not from the smoothed measured gap: an average of an irregular series is larger than
the gap about half the time and each of those asks for a negative fraction, which is how 9 and
then 5 slot-frames came to sit exactly on the floor. And the floor itself had to become symmetric
with the ceiling (-1.5 against +2.5), because reading a secant a little before its older sample
is the same second-order read as reading it past the newer one.

With all three fixed: **0 jumps, 0 reversals**, worst position error 8.2e-5 of the radius against
1.0e-2 for not interpolating at all, and the drawn motion equals the true motion to 1.000x at p50,
p90 and p99. Reverting any one of the three turns `drift.mjs` red again.

#### The read is a secant, and that is not a shortcut

The morph is splined once, between map sets, and read linearly from there: the two most recent
builds are two samples of that spline, and the drawn shape is the line through them evaluated at
the parameter the morph has reached. `tests/secant.mjs` differences that read against an exact
build at the same parameter, entry by entry over the premultiplied table. Over 12,960 slot-frames
the error is 1.6e-4 at p50 and 8.9e-4 at p99, where drawing the newest build alone is 4.9e-3 and
the drawn table's own frame-to-frame step is 1.1e-3 -- so the straight line misses the curve by
about a seventh of the movement it is drawing, and by 0.15% of an entry at p99.

Which is the answer to the obvious next idea. A spline at the read level needs a third sample,
and a third sample is a third resident table (288 KB a slot) and a third set of reads in the hot
loop -- the budget that materialising the blend into one table blew, for a residual already an
order of magnitude under one frame of motion. And the fraction never falls below 1: this is an
extrapolation every frame, never an interpolation, so the error is largest exactly where a cubic
is worst -- 2.7e-5 at fractions under 1.25 against 5.1e-4 past 2.0. That tail belongs to the
2.7% of slot-frames whose rebuild the budget deferred, so it is bought back by the schedule, not
by the curve.

One line of this is load-bearing and looks like a typo: `_bt` is initialised to 0.5 and then to 1.
V8 takes a field's representation from the first value stored, so initialising it to the integer 1
means the first real fraction changes the representation, transitions the hidden class and throws
away the optimised `reparam` — 15 ms a rebuild instead of 2.4, for six rebuilds, right at startup.
`tests/field.mjs`'s "frames 0-3 are not a stall" check caught it.

`tests/breath.mjs` has a fifth section now, measuring the table as the traversal reads it rather
than the parameters that feed it: the pool can be morphing perfectly and still be showing a frozen
shape, which is the whole fault.

#### A sixth, on the dial that makes all of it affordable

Reading the drawn table rather than the parameters turned up a second thing wrong with it, nothing
to do with the blend. The pool sizes each slot's jitter table by how important the slot is on
screen -- the object filling the view gets 1024 rotations, the specks get 64, and that is what
makes a 60 Hz morph affordable at all. The resize is asked for in the ranking pass; the rebuild
that would fill the table is budgeted. They are not the same frame whenever the budget is spent.

A node picks its rotation with `hash & tableMask`, so moving the mask alone moves which entries
the traversal reads. Shrinking is harmless -- the active entries are a subset of what the last
build wrote. Growing is not: the entries above what the last build wrote hold whatever that slot's
table held the last time it was that big, which is another build of it, seconds of morph ago. And
it happens on the object that has just become the most important one on screen, which is the one
being looked at. Measured over 60 s of promotions and demotions: of 150 resizes, 11 to 47 were
drawn for one or two frames before their rebuild landed, the spread being how busy the machine was
-- the budget is spent in real milliseconds.

`setTableBits` now records the size wanted and the next rebuild applies it, so the mask and the
table it indexes always belong to the same build. The saving is untouched: it was never in the
mask, it is in that rebuild premultiplying K entries instead of 1024. `tests/breath.mjs` §6 flies
60 s of changing importance at a budget that defers nearly every resize, and asserts that no slot
is ever drawn at a mask its table was not built for -- 47 slot-frames before, 0 now.

#### And the dial itself was the jump

Flown with all of the above in place, the picture was still jumpy at times. It was not the blend.

`tests/drift.mjs` prints the drawn step beside the true one, and the step column had an outlier
nobody had read: **4.28e-2 of the object's radius at its worst against 2.61e-3 at p99** — 86x one
frame's own motion. Every one of the fourteen largest steps in a 1500-frame flight was a frame on
which a slot's jitter table changed *size*, and the median step on a resize frame was **9.75e-3
against 5.0e-4** for an ordinary one. The pool re-cuts 140 to 150 tables a minute.

A node picks its rotation with `hash & tableMask`, so a resize re-draws the branch jitter of every
node whose index moves — half of them for one bit, nearly all of them for four. It is a different
*realisation* of the same set, not a different shape, and there is no clever indexing that avoids
it: if the extra entries mean anything at all, the shape with 1024 of them is not the shape with
64. What can be fixed is that it happened in one frame.

**Why nothing caught it, which is the more useful half.** Both tests that measure the drawn table
compare it against an exact build *at the same mask* — so the reference re-draws its jitter in the
same frame and in the same way, and the error sat at 1.5e-5 while the picture snapped. The only
column that could see it was the step, and the step was asserted as a *ratio* against the true
motion, which was equally large: the ratio read 1.00 and the check passed. `drift.mjs` has an
absolute bound now, against the distribution of ordinary frames, and it is red at 86x if the ramp
below is switched off.

**The fix is to ramp the resize instead of switching it** (`Attractor.migrateRebuilds`, 4):

- A **grow** takes its new mask at once, with every new entry holding an exact copy of the entry
  its nodes were already reading — the mask moves and nothing on screen does — and then carries
  each moving entry to its own rotation over four rebuilds.
- A **shrink** keeps its size and merges the doomed entries onto their partners over the same four
  rebuilds; the mask is dropped only once *both* halves of the blend's pair are at the merged
  weight. Dropping it one rebuild earlier moves the picture by one ramp step, which is the thing
  being avoided.
- Each of those four steps is then smoothed across the frames between rebuilds by the read-time
  blend, which was already there for the morph. That is why four is enough: the ramp only has to
  break the change into steps no bigger than a rebuild's own.
- The blend is of the two rotation **matrices**, nine lerps an entry, not of their axes and angles:
  no trig in a per-rebuild loop, and no case where two nearly opposite axes interpolate through a
  vector of no length. The halfway matrix is not exactly a rotation — at a tenth of a radian its
  determinant dips about 1e-3, a 0.03% breath in the size of a splat, three orders under the
  1.2e-3 of its own radius that the splat moves in a frame anyway.

| worst drawn step in a 1500-frame flight | switched | ramped |
|---|---|---|
| in units of the object's radius | 4.28e-2 | **2.66e-3** |
| against the p99 of an ordinary frame | 16x | **1.03x** |
| against the median ordinary frame | 86x | **5x** |

Mid-resize frames now step *less* than ordinary ones (p99 2.04e-3 against 2.59e-3, median 5.2e-4
against 5.4e-4), because a slot's hand-over between realisations partly opposes the morph it is
riding on. Tracking is unchanged where it is defined: 1.5e-5 at p50, 7.9e-5 at worst.

**It costs nothing measurable.** A rebuild with a ramp in flight costs what a settled rebuild at
the larger of the two sizes costs — 0.507 ms either way on this machine, against 0.236 ms at 64
entries. What a ramp defers is the *saving* from a shrink, by four rebuilds.

**What it costs the tests** is worth stating, because it is a real loss. While a ramp runs the
drawn shape is deliberately part way between two realisations, so an exact build at the new size
is not what it should equal: `drift.mjs` leaves those slot-frames out of its tracking checks (40%
of them here, the ramp being four rebuilds of a seven-a-second schedule) and `secant.mjs` leaves
them out of its entry-wise error, and both print the count. The step checks cover them instead,
which is the honest division of the claim — a blend of two valid realisations of the same set at
the same morph parameter *is* a valid shape, and what has to be true of it is that it arrives
smoothly.

**Rejected on measurement: hysteresis on the ranking.** Sizes come from rank, so two objects of
nearly equal importance trading places re-cut both their tables although neither changed on
screen, which looked like most of the churn. A two-place dead band turned 137 resizes over 60 s
into 142 — nothing, because rank drifts steadily through all eight positions as the camera flies,
so a dead band delays a re-cut rather than preventing it. Not shipped.

#### And the next one needs a witness, so the moment is a link

Every number above was measured here, headless, over a flight this machine drove itself. None of
it can answer *"it went jumpy just then"*, and that is the form the next report will take, because
the only instrument pointed at the real thing is somebody's eye. A frame number cannot close the
gap either: the autopilot integrates real frame times and `tests/pilot.mjs` establishes that the
dive is chaotic -- one part in a million of speed puts the heading 60 degrees away in three
seconds -- so frame 4213 in a browser names a state that cannot be reached again from here.

So the page carries the state instead of pointing at it. **J**, or the panel's button, writes one
frame of the simulation into the URL fragment and copies the link.

What makes that cheap is the thing the whole engine is built on: every heavy array is a pure
function of a few parameters. A slot draws from two premultiplied child tables of 288 KB each,
and both of them are `premul(build(morphPath(u)), jitter table at bits, seed + k)` -- or, while a
resize migrates, the same build with the jitter rotations blended at weight `_migW`, which is
again a function of `(u, bits, migLo, migW, grow)`. Ten numbers a slot. The jitter axes come from
the entry index alone, so the seed fixes them; the palette is a function of `u`; the field's cells
are a hash of where the camera is. **20 KB of JSON, 7 KB of link** after gzip, for the pool, all
eight slots, the camera, the view settings, and the 90 frames before the keypress.

The window is there for one reason: a hand takes about three tenths of a second, which is eighteen
frames, so a snapshot of the present frame is a snapshot of the aftermath. Ninety frames at 60 Hz
is a second and a half and holds about ten rebuilds per slot, which is what lets a replay cross
the hand-overs rather than just sweep a fraction between two fixed builds. Per frame it carries
each slot's blend fraction and the splat movement its own probe measured, plus the frame's cost
and the splat count -- because three of the four things that look like a jump are not the morph at
all, and a 200 ms frame (twelve frames of motion at once) or a quarter of the splats changing is
answerable from the link alone.

`tests/mark.mjs` is the other half, and it does not trust itself. Every rebuild in the window
carries the state the browser reached -- mask, migration weight, which entries were still merging,
whether the mask drop was still pending -- and the replay drives the same `reparam()` with the
same requests and is checked against all of it, then against the browser's own per-frame splat
movement. Measured on a flight with the ramp switched off, so that there is a real jump in the
window to reproduce: **every one of 11 rebuild records reproduced exactly, and the replayed splat
movement matched the recorded movement to 0.005% at p50 and 0.017% at worst.** Only then does it
draw: a filmstrip of the frames either side of the worst one.

The live probe is `tests/drift.mjs`'s measurement at frame rate -- 8 fixed addresses at depth 3
per slot, 192 node compositions, against the 24 at depth 4 the offline test can afford -- with
each slot's own median step as the yardstick, which is what makes one threshold fit eight objects
of different sizes. The A/B that says it works is the fault this stage already fixed: over 900
frames with the resize **switched** it reads **212.6x the slot's median and flags 30 frames**;
with it **ramped**, **3.4x and nothing**. Cost **0.105 ms a frame** for eight slots, against the
1.5 ms the rebuild budget alone is allowed, so it stays on.

And the HUD prints `frame` and `worst jump` continuously, which makes the quiet answer available
too: if a minute of flying leaves `worst jump` at 2x, the morph is saying it is not what is being
complained about, and that is worth as much as finding a jump.

### The climb does not arc any more

Measured, 200 s each (`tests/legible.mjs`): arcing out of a dive rather than reversing back up the
line it came in on costs **37.6% washed-out frames against 30.5%**, 11.6% nearly-empty against
8.3%, and — the surprise — the *following dive* washes 41.0% against 31.6%. The arc leaves the
camera off to one side of everything, and the dive that starts from there has further to go before
it is inside anything. `Autopilot.arc` is 0 and the sideways direction is still computed, so it is
one number to put back.

**How much to trust that.** Not as much as it looks. `legible.mjs` built its autopilot with the
default `Math.random`, so those two runs were two different flights — and this repository's own
`tests/pilot.mjs` establishes that the dive is chaotic, that one part in a million of speed puts
the heading 60 degrees away in three seconds, and therefore that one flight against one flight is
not a comparison. Every A/B ever measured with this file has that weakness. It now takes a seed
(`node tests/legible.mjs 120 280 210 <seed>`), so a comparison can be *paired*: the same flight,
flown by two builds.

Pairing alone is not enough here, and the attempt showed why. One autopilot cycle is 44 s of
cruise, 72 s of dive and 8 s of climb -- **124 seconds for a single climb**, which is the phase
the arc exists in. A 120 s run contains one; several seeds of that are still a handful of climbs,
and a 600 s run that would hold five costs an hour of CPU per variant, because every frame is
rasterised in JavaScript. The measurement that answers this is not a longer flight. It is a probe
that dives, climbs, and records the wash through the climb and through the first seconds of the
dive that follows, over many trials: the same flight either side, many climbs per minute of CPU.
That is the next instrument to build, and until it exists the arc number above is one paired
flight and should be read as one.

### What the whole flight measures

200 s of autopilot, 200x150, the corrected probe, old code against new (`tests/legible.mjs`):

| | before | after |
|---|---|---|
| washed-out frames | 29.6% | 30.4% |
| nearly-empty frames | 22.9% | **11.8%** |
| dive frames washed out | 37.4% | **27.3%** |
| climb frames washed out | 40.9% | 77.9% |
| cruise frames nearly empty | 93.7% of 1389 | 95.2% of 644 |
| blank frames | 0.1% | 0.1% |

The dive is a third better at being worth looking at, while flying 1.7x slower and ten levels
deeper, and emptiness is halved. The headline number does not move, and the reason is in the two
rows that got worse: the flight now spends far more of itself diving and climbing (4633 + 723
frames against 3723 + 888) and far less crossing empty void, and the climb is the one phase that
is deep for its whole duration. Washed-out frames sit at a mean depth of 34 against 13 for
legible ones.

So this did not fix the wash; it moved the fault to where it can be seen clearly. **The remaining
win is the one `IDEAS.md` named at stage 4 and is now sharper: the climb.** It is 12% of the
flight and 78% of it is a fog bank, which is a third of every washed-out frame in the run.

### A sky to fly through

- **Stars**, and they do not crawl. The nebula is deliberately smooth because noise shimmers at
  this camera's scales; a star field is the ruled-out thing, so it is here on the splats' terms —
  each star's angular sigma is floored at 1.3 pixels, handed in from the real viewport, and the
  profile is a Gaussian, so there is no sub-pixel spike and no edge to alias. `tests/space.mjs`
  measures the difference: the floored field answers a 4x bigger turn with a 3.9x bigger change,
  while the same field at 0.08 px is already at its worst-pixel ceiling for a quarter-pixel turn
  and does not grow — which is what crawling is.
- **The nebula carries more colour**: broader lobes (`lobeK` 5/8/12 -> 3.6/6/9) and stronger
  colours, so fog reads as distance rather than as fading out.
- **A harder, cooler rim light** (0.55 -> 0.74, `rimPow` 3.0 -> 2.5). The rim is the only thing in
  the shader that draws a silhouette, and the dive now flies along silhouettes.
- **The camera banks into its turns**, up to 26 degrees, as a relative lean that unwinds when the
  turn ends — there is no world up to roll against, and a basis that re-roots underneath the
  camera cannot carry an absolute one.
- `cpu_raster.js` can draw the sky now, so the background can be looked at headless. Off by
  default: `tests/legible.mjs` measures spatial detail, and a star field behind the splats would
  raise its score without a single splat changing.

## Written up

`README.md` now describes the demo and carries a table of what was measured; `IDEAS.md` has a
section at the end recording which of its own predictions held and which did not, with the
predictions left standing rather than quietly edited.

## Not done

1. **The WebGPU host code is not written.** `gpu_kernel.js` has the shader and the binding layout
   it expects; there is no `device`/pipeline/buffer setup, and nothing wires it into the viewer.
2. **Adaptive expansion on the GPU.** The kernel above is uniform-depth only. Under an LOD cut you
   do not know a subtree's size until you have walked it, so the slot arithmetic has nothing to
   prefix-sum. It needs three passes — expand-and-count, a bottom-up size scan, a top-down slot
   assignment. Designed, not built. This is the one that would actually raise the flight's ceiling;
   the uniform-depth kernel only helps stage 1's static object.
3. ~~**Publishing.**~~ Done 2026-11-17, on the owner's go-ahead: `dist/` is live at
   <https://mikaelcromsjo.com/splat/>, served straight from `/srv/sites/mikaelcromsjo/splat/`
   by the existing nginx vhost (no config change needed; `.js` goes out as
   `application/javascript`, so the ES modules load). Published with `bin/deploy` since
   2026-11-18, which stamps the build id into the page and onto every module URL and then checks
   what the server returns; the id is on the panel as `build` and in the console.
   `README.txt` was deliberately left out of
   the published copy because it quotes internal `/srv/` paths. The directory also holds an
   unrelated `tree.ply` (2.6M, the 38229-splat parity export) which predates the publish and is
   publicly reachable; nothing in the demo references it.
4. **Audio-driven map parameters** — the plan's explicit stretch goal, untouched.
   **`tests/legible.mjs` is red, and was already.** At the documented invocation
   (`120 280 210 1`) it reports 13.4% washed-out frames against its own 10% limit — dive 13.1%,
   climb 20.9%, 2 blank frames of 3600. Checked against `HEAD` exported to a scratch tree and run
   the same way: **every reported outcome identical to the frame**, so the interpolation work of
   2026-11-18 neither caused it nor moved it. The limit is tighter than the flight has reached at
   this seed and length; it is an outcome probe to be compared in pairs, not a gate, and the
   legibility items above are the open work it belongs to.
5. ~~**The live site is a build behind.**~~ Republished 2026-11-17 on the owner's go-ahead:
   <https://mikaelcromsjo.com/splat/> now serves the field build (11 modules, `field.js` among
   them, all verified 200 with `application/javascript`). It opens on the field; **F** switches
   to stage 1's single object. Republished again the same day with stage 6 — every import
   re-verified 200, and the three test-only modules that had been copied up with the rest
   (`cpu_raster.js`, `gpu_ref.js`, `calibrate.js`) removed; `cpu_raster.js` imports `node:zlib`
   and would have failed in a browser had anything loaded it.
   **The star field has never been compiled.** There is no GL in this environment, so the
   background program's new loop and five uniforms went live unexercised. `makeBgProgram()` is
   therefore wrapped: a background that will not compile disables the nebula and leaves the flat
   clear colour, rather than taking the page down with it. The splat programs are unchanged from
   what was already live and are still fatal.
   Republished 2026-11-18 with the three interpolation fixes above (`setInterp` re-gathers, the
   table resize waits for its rebuild, `_btSafe` comes from its own pair) -- `adaptive.js` was the
   only file that differed, verified byte-identical to `dist/` over HTTP at
   `application/javascript`. Redeploying after a change no longer waits on a go-ahead.

## How to look at any of it without a browser

```
cd web
node tests/tour.mjs 150 out/tour.png      # fly the whole demo, filmstrip
node tests/fidelity.mjs                   # LOD vs ground truth, every set
node tests/shot.mjs lattice 150 out/x.png # one set, after diving 150 frames
node tests/void.mjs                       # the coordinate chain
node tests/gpu_transcribe.mjs 7           # the compute kernel vs the CPU
node tests/field.mjs                      # the cell field: determinism, separation, budget
node tests/breath.mjs                     # the morph: C1, breathing, inside the budget
node tests/toggles.mjs                    # the morph switches, and the panel's two rates
node tests/secant.mjs                     # the read-time blend vs an exact build
node tests/drift.mjs                      # drawn splat positions vs an exact build
node tests/jitterbits.mjs                 # the jitter table vs the reference, and its size
node tests/mapset.mjs                     # every set, every pair, and the whole spline
node tests/catalogue.mjs                  # all 11 sets, and a morph, as pictures
node tests/legible.mjs                    # is there anything to SEE, or is it a fog bank
node tests/order_field.mjs                # draw order in a crowd vs an exact sort
node tests/tour.mjs 40 out/f.png 260 200 field   # fly the field
node tests/tour.mjs 40 out/r.png 260 200 ring    # the old seven, as a regression
```
