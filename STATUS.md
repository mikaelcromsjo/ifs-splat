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

One line of this is load-bearing and looks like a typo: `_bt` is initialised to 0.5 and then to 1.
V8 takes a field's representation from the first value stored, so initialising it to the integer 1
means the first real fraction changes the representation, transitions the hidden class and throws
away the optimised `reparam` — 15 ms a rebuild instead of 2.4, for six rebuilds, right at startup.
`tests/field.mjs`'s "frames 0-3 are not a stall" check caught it.

`tests/breath.mjs` has a fifth section now, measuring the table as the traversal reads it rather
than the parameters that feed it: the pool can be morphing perfectly and still be showing a frozen
shape, which is the whole fault.

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
   `application/javascript`, so the ES modules load). `README.txt` was deliberately left out of
   the published copy because it quotes internal `/srv/` paths. The directory also holds an
   unrelated `tree.ply` (2.6M, the 38229-splat parity export) which predates the publish and is
   publicly reachable; nothing in the demo references it.
4. **Audio-driven map parameters** — the plan's explicit stretch goal, untouched.
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
node tests/jitterbits.mjs                 # the jitter table vs the reference, and its size
node tests/mapset.mjs                     # every set, every pair, and the whole spline
node tests/catalogue.mjs                  # all 11 sets, and a morph, as pictures
node tests/legible.mjs                    # is there anything to SEE, or is it a fog bank
node tests/order_field.mjs                # draw order in a crowd vs an exact sort
node tests/tour.mjs 40 out/f.png 260 200 field   # fly the field
node tests/tour.mjs 40 out/r.png 260 200 ring    # the old seven, as a regression
```
