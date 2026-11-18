# A geometric visual world

Notes from the design conversation of 2026-11-17. The generator in `ifs_tree_splat.py` works; this
is about what to build on it. Nothing here is implemented yet.

## Direction

**Build a geometric world, not a forest.**

The method's weakness as naturalism is its strength as style. IFS vegetation looks fractal, not
natural — true of every fractal-plant demo since the 1980s, which is long enough to call it a
property of the method rather than a problem to solve with effort. Real nature reads as natural
because it is non-stationary and historical: competition for light, broken limbs, disease, wind,
age. Self-similarity is the opposite of that. Each step taken toward naturalism (noise-modulated
map parameters, per-species variation, damage models) moves away from the method, until the IFS is
only a geometry compression scheme and the real work is conventional procedural vegetation.

So: a crystalline, lawful, obviously-constructed nature. Recursive and alien, flown through rather
than walked, carried by motion and sound. Weeks of work rather than years, and the ceiling is high
because nothing else looks like it.

The two things that cap the realistic version are worth writing down, because they will tempt
someone later:

- **Lighting.** What makes a real forest beautiful is dappled light and subsurface scatter in
  leaves. Splats with baked spherical harmonics cannot take a moving sun; forward-shaded splats
  lose the appearance that makes 3DGS impressive in the first place. A 3D ellipsoid also has no
  well-defined normal (a flat disc does, which is why foliage discs behave better than wood rods).
- **Scope.** "Open world" adds collision, streaming, terrain, water and gameplay — years of work in
  exactly the areas where the fractal contributes nothing. The interesting part is about 5% of it.

A geometric world sidesteps both: stylised shading is a choice rather than a compromise, and a
flythrough needs no collision or gameplay.

## What the method gives for free

Generation on the fly is unusually well suited here, because generation and level of detail are the
same mechanism rather than two systems fighting each other.

- **Addresses are random-access.** A node's affine is a matrix product along its path, so any
  address can be expanded without materialising its ancestors' other subtrees. A frustum walk
  touches only what is visible.
- **A node's Gaussian is a correct rendering of its whole subtree.** Stop recursing when projected
  size falls under a few pixels; the blob drawn instead of 4^n descendants has the right shape,
  position and anisotropy, because the Jacobian produced it.
- **Memory is O(visible), not O(world).** Store the map set and a seed field; nothing is persisted
  or streamed from disk.
- **Deterministic.** Same address, same splat, every machine. Caching and multiplayer consistency
  are free.
- **Moment-matched LOD in closed form.** The invariant measure's moments satisfy a linear
  fixed-point equation in the maps, so a parent Gaussian matching its children's mixture in mass,
  mean and covariance can be computed rather than fitted. Hierarchical 3DGS (Kerbl et al. 2024) and
  Octree-GS build that tree by fitting to captured data; an IFS hands it over directly.
  Note: this weakens once map parameters vary with world position, since subtree statistics then
  differ by location — fall back to sampling a few representatives per region.

## The cost model is backwards, and that is the real constraint

Splats are cheap to produce and expensive to composite. Correct alpha blending needs per-frame
back-to-front ordering, and canopy overdraw is dozens of translucent layers deep. Procedural
generation makes production nearly free — but production was never the bottleneck. A GPU radix sort
of 2–5M splats is roughly 2–5 ms, a large part of a 16 ms frame, spent before anything is shaded.

Which is what makes the next section the most valuable idea in this file.

## Ordered generation: emit already sorted

**Sorted order is lexicographic order on addresses, under a per-node permutation of the digits.**

Bound the attractor once by iterating the Hutchinson operator on a ball until it stops shrinking,
giving `B`. A node with accumulated affine `(M, b)` then has its entire subtree inside `M·B + b` —
an exact ellipsoid bound, no fitting. That is a bounding volume hierarchy with branching factor m,
zero build cost and zero storage.

Child `i`'s centre in the node's local frame is `c_i = A_i·MU0 + b_i`, fixed and known at build
time. Their order along view direction `d` is the order of `<c_i, M^T d>` — it depends only on the
direction `M^T d`, not on where the node sits in the world. So ordering means sorting m numbers,
not N.

### Ordered expansion

    level k is an array of nodes, already in back-to-front order
    for each node (in parallel):
        d' = M^T d                        # view direction in the node's frame
        rank = order of <c_i, d'>         # m+1 elements incl. the wood splat;
                                          # a 5-element sorting network is 9 compare-swaps
        child j writes to slot = parent_slot * m + rank[j]

Each level is sorted by construction, so the last level is the instance buffer in composite order.
One kernel per depth level, no sort pass, no prefix sum. Cost is about N/3 nodes x 9 compares,
shared by every descendant, against a global radix sort over N keys every frame.

Submission order suffices: within a draw call both GL and Vulkan guarantee fragments blend in
primitive order. Emit, draw with depth writes off, done.

### Closed form at uniform depth

A leaf's index in the sorted stream is

    index = SUM_k  rank_{p_k}(i_k) * m^(D-k)

a mixed-radix number whose digits are the address, each permuted by its node's ordering. A thread
holding an address already computes `M` incrementally along its path, so it gets its own output
slot in O(D) with no communication at all. Pure scatter.

With adaptive depth (wanted, for LOD) subtree sizes vary, so slots come from one scan over ordered
children per level. Still far cheaper than sorting.

### Front-to-back early-out — the bigger win

Once the stream is ordered, run it front-to-back and track per-tile transmittance. When a tile
saturates, stop expanding every node whose footprint lands only in saturated tiles, skipping whole
subtrees. That attacks overdraw, the actual bottleneck, not just the sort. A global sort cannot do
this: it must produce all N before anything is drawn. This is also where the traversal would go
per-tile rather than global, which is closer to per-ray-correct than a global sort ever is.

### Where the order is only approximate

Sibling order by bounding volume is exact only where siblings do not overlap along `d`. In
interpenetrating canopy they do. Three reasons that is acceptable:

- It is exactly as approximate as what web 3DGS viewers already ship, which sort globally by splat
  centre — also not per-ray correct.
- It is far more temporally stable. A node's permutation changes only when `M^T d` crosses one of
  the m(m-1)/2 great circles separating order cells: rare, local, and swapping adjacent elements.
  3DGS popping comes from order *changing*, not from order being wrong, so hierarchical ordering
  tends to look better in motion than an exact per-frame sort. StopThePop (2024) is essentially
  this observation applied to trained scenes.
- Where tighter order is needed, descend further: if two children's `[dmin, dmax]` intervals
  overlap, expand both one more level before emitting. A lazy k-way merge — pay only on ambiguous
  nodes.

The honest limit: none of this transfers to captured 3DGS scenes. It works because the geometry
*is* the hierarchy, with bounds for free. That is why the papers radix-sort instead.

## Other things that will be needed

- **Aliasing is not optional polish.** Small splats at distance shimmer violently under motion;
  Mip-Splatting-style covariance dilation by the pixel footprint is required. It changes how
  everything looks, so build it in from the start rather than retrofitting.
- **LOD popping.** Moment matching fixes the geometry; colour and opacity still jump at a split,
  needing a cross-fade band where parent and children both render — more splats and more ordering
  exactly while the camera moves.
- **Collision, if ever needed.** The wood rods already are capsules: position, axis, radius, from
  the same recursion. Ground and water are not.
- **Variety.** Pure self-similarity reads as synthetic. Map parameters as a function of world
  position via a low-frequency noise field keeps determinism and random access.
- **IFS covers branching only.** Trees, ferns, lichen, river networks, some cloud structure: good.
  Eroded terrain, grass, rock, water: not an IFS.

## What to measure first, before building anything

1. **Does the approximate order matter?** Emit with ordered expansion, compare against a
   ground-truth per-frame sort, count inversions weighted by opacity. A number instead of an
   argument. Cheap — it can be done in the Python generator, no GPU.
2. **What does the frame actually cost?** A few hundred thousand splats from a handful of trees on
   screen, per-frame ordering, camera in motion. Frame time and shimmer decide the project.

Neither needs a world. Both are a few days.

## Decided against

An open-world *game*. The fractal gives nothing to what makes open worlds good, and the work lands
almost entirely in areas where the method has no advantage. Memory footprint is the only axis where
it clearly beats conventional approaches.

---

# What happened when these were tested

Added 2026-11-17, after building the demo in [PLAN.md](PLAN.md). Everything above was written
before any of it was measured, and some of it was wrong. Rather than quietly edit the predictions,
they are left standing and the results recorded here. [STATUS.md](STATUS.md) has the full account.

### Held up

**Ordering by permuting each node's children, instead of sorting.** The central claim. Approximate
order costs about a third of the frame-to-frame change the camera itself produces, at every
percentile including the single worst pixel. Nothing N-sized is ever sorted, and at 152k splats
the whole per-frame ordering is 3.5 ms in JavaScript.

**Moment-matched parents from the maps' fixed-point equations.** They reproduce a fully-expanded
view to 26–44 dB at 2–12× fewer splats. The closed forms were checked against Monte-Carlo moments
of the actual generated subtree, and against the renderer.

**Re-rooting for precision.** This was called the main technical risk and it was: the first
implementation composed with `A_i` instead of `A_i⁻¹` and sent the world to infinity the moment
the camera descended a level. Once fixed, it holds to one float64 ulp in world units at depth 52.

**Address-hashed jitter.** Necessary exactly as predicted — and it is also what makes the GPU
kernel possible at all, because every thread can reconstruct any node's perturbation from its
address alone, without knowing what any other thread did.

### Did not

**"Inversions happen where the splats are nearly transparent, so they are hidden."** They are not.
Weighting inversions by how *visible* each splat actually is gives 16.0%, slightly *more* than
weighting by opacity alone (15.5%). The errors are not preferentially in the transparent regions.

**"Hierarchical ordering is more temporally stable than an exact sort, because a node's children
keep their relative order until the node itself flips."** The opposite: by pair-churn between
consecutive frames it is about 5× *less* stable. The intuition ignored that one flip high in the
tree reorders its whole subtree at once, which is a larger discrete event than the many small
independent swaps an exact sort makes.

**"Front-to-back traversal with an early-out on an opaque tile buffer will pay for itself."** Not
here. It skips 6% of nodes and roughly triples the per-node cost, for a clear net loss, because at
this level of detail the canopy is only about 2× overdrawn — there is not enough occlusion to
harvest. Implemented, measured, left behind a flag that defaults to off. It would likely pay on a
denser attractor, which is a reason to keep the code rather than a reason it works.

### Not predicted, and worth knowing

**A cyclic palette barely bleaches.** The worry was that averaging a parent's colour over its
children would grey everything out. Measured: a 12-entry ramp keeps 92% of its saturation, and
even a full hue wheel over 24 entries keeps 83%. The tree's olive canopy, which looked like
exactly this failure, is not — it is brown rods and green leaves mixing in *space*, below the
resolution limit, which is the correct answer rather than an artifact.

**The eigensolver is unnecessary when the maps are similarities.** The foliage disc needs the
covariance's principal axis, and the CPU finds it with a Jacobi solve. But every branch map is
`s·R` and so is every jitter rotation, so the accumulated affine is `s·R` exactly, its columns all
have length `s`, and the principal axis is just its third column normalised. Exact instead of
iterated, and it is what lets the whole thing run in a compute shader without a solver.

**Adaptive expansion does not port to the GPU the way uniform-depth expansion does.** At uniform
depth each thread can walk from the root to its own splat and compute its stream slot on the way,
because subtree sizes are known in advance — one pass, no atomics, no compaction, and the draw
order falls out of the same walk, so the per-frame CPU cost drops to zero. Under a level-of-detail
cut you do not know a subtree's size until you have walked it, so the slot arithmetic has nothing
to prefix-sum. It needs three passes: expand-and-count, a bottom-up size scan, then a top-down
slot assignment. Designed, not built, and it is the piece that would actually raise the flight's
ceiling.

---

## Stage 4: the field, and what it taught

**The dive goes too deep, and that is now the main thing to fix.** Measured over a 900-frame
flight: 28.2% of frames in the field (42.0% in the old ring) are a near-uniform wash of a single
colour, because the camera is *inside* a structure and a Gaussian field at that range has no
silhouette left to read. Nothing is wrong with the renderer — the frames are a correct picture of
being inside foliage. The autopilot is what is wrong: it treats "deeper" as "better", when the
shot that actually reads is the approach, structure resolving out of a blob as you close on it.
A dive that reversed at the range where projected detail peaks, rather than at `maxDepth`, would
spend the flight in the part of the zoom that looks like something. Cheap to try: track the
image's spatial variance (the measurement already exists, in the flatness probe) and turn the dive
around when it starts falling.

**The spike was never the arithmetic.** A settled frame measured 33 ms of "rebuild" cost against a
per-rebuild p99 of 2.06 ms over every morph pair. It was the garbage collector, triggered by
`frames()` allocating a matrix, four vectors and an object per visible object per frame — invisible
at seven objects, two thousand allocations a second at forty. Making `frames()` and `rootFrame()`
allocation-free took the mean from 1.99 to 1.18 ms/frame. The lesson is about measurement, not
allocation: the timer was around `build()` and `reparam()`, so every pause that happened to land
in that window was attributed to them, and the number pointed confidently at the wrong code.
A wall-clock maximum is a test of the collector; `tests/tour.mjs` now gates on p99 and reports the
maximum with its cause.

**Ordering in a crowd held up better than predicted, and the reason is the separation rule.**
The worry was that between-object ordering is a heuristic — a sort by nearest point of the
bounding sphere — and that forty objects would compound it. Measured: 8.18% mean opacity-weighted
inversion rate with 25-45 objects in view, against 8.76% for a single attractor. No worse. Because
the cells guarantee disjoint bounding spheres, the heuristic is nearly always right; it is only
interpenetrating objects that would break it, and the generator makes those impossible.

**Sharing an Attractor is not sharing an appearance.** The pool exists for cost — eight shapes
instead of one per object — and the expectation was that it would buy repetition. It does not,
because the per-object root hash is the root of the address-hashed jitter: the same map set walked
from a different hash is a different realisation all the way down. The variety that matters was
already in the hash, not in the map set.

## Stage 5: smoothness is a property you have to go and check

Stage 5 set out to make the world morph continuously, breathe, stop looking like a wood, and stop
looking like a lattice. The first three are done. What it taught was almost entirely about the
gap between "the maths is smooth" and "the thing on screen is smooth".

**Catmull-Rom is C1 by construction, and that guaranteed nothing.** Every single break in
continuity came from something wrapped *around* the spline, never from the spline:

- a **clamp**, which is flat while it is active and sloped the instant it is not. Any control
  value sitting exactly on the bound gets a corner — and `omega = 0` is exactly on the bound for
  every crystalline set, so each new set quietly put a velocity jump at its own waypoint. Worse,
  the scale clamp's bounds were drawn from *the segment being travelled*, so they changed at every
  waypoint by construction. A clamp added for stability removed the property the stage was about.
- an **angle tie at exactly 180 degrees**, where "the short way round" is genuinely ambiguous. The
  textbook formula resolves it the same way whichever direction it is asked about, so `delta(a,b)`
  and `delta(b,a)` were both -180 instead of being opposites, and the tangents either side of a
  waypoint disagreed. Then the fix was wrong too: it broke the tie on the raw values, but the
  unwrap chain hands the same angle back shifted by a multiple of 360, so the tie-break has to
  depend only on the residue. Two bugs, one line.
- a **floor**, `Math.max(1e-9, sig0)`, which was not a safety net but a blindfold: it was hiding
  `sig0` going to **-0.021**, a negative variance, which is a degenerate splat rather than a small
  one.

The general shape: **every guard that clips a value is a discontinuity you have chosen to add.**
Where a quantity must stay positive, reparameterise instead of clamping — log space for strictly
positive (scales, variances), the square of a splined square root for non-negative-and-sometimes-
exactly-zero (`omega`, `jitter`). Both are smooth everywhere and neither can produce an invalid
value, so the guard is not needed at all. The only honest clamp is one you have proved never
activates, and if you have proved that, you can delete it.

**Measure the thing, not its representation.** The first smoothness test reported 3791x lurches
that did not exist, because it compared angles in degrees and an unwrapped representation can
jump 360 with nothing moving. Comparing `(cos, sin)` instead dropped it to 9x. Separately, a
relative comparison between two velocities that are both numerically zero reports 100% forever;
several "failures" were that, and the real breaks had to be found by component.

**A constant can depend on an ordering you did not think was load-bearing.** `REACH_MAX` bounds
every object's extent and is what the non-overlap proof — and therefore the draw order — rests on.
It is measured over the morph path, and the morph path is a loop through the sets *in order*. An
ordering that happened to put the two highest-contraction sets next to each other sent the
spline's overshoot to 5.12 against a bound of 3.75, because R grows like 1/(1 - contraction) and
the last few percent of contraction are most of the size. Reordering fixed it. The ordering of a
list that looks purely cosmetic was silently part of a geometric guarantee, so the test now
re-measures the bound over the spline and the constant's comment says so.

**The knobs were already there.** "Not based on a tree" sounded architectural and was almost
entirely content: `omega` is the trunk's mass fraction, `omega = 0` is a bare attractor, and a
Sierpinski tetrahedron is four similarity maps. Seven sets had shipped with `omega` between 0.15
and 0.38 and an identical jitter of 0.16, which is the entire reason the field read as a wood.
Before concluding an engine needs changing, check whether a parameter has simply never been
varied.

**Cheap gates get run; expensive ones do not.** Extending `mapset.mjs` to validate the whole
spline made it an eleven-minute job, because each sample rendered a frame to measure cost. Whether
a spec is *valid* is a question about contraction, radius and moments — all known by the time the
attractor is built. Skipping the render took it to 12 seconds for 11 sets, 55 pairs and 660 spline
samples.

## Stage 5, part two: a hierarchy makes the proof shorter

The flat field's separation argument needed a page: every object inside the sphere inscribed in
its own cell, cells of one size on one grid, 26 neighbours enough because anything further is a
full cell-width away. Making the cells hierarchical sounded like it would make that argument
worse, and it made it **shorter**:

> A cell that holds an object never subdivides. Cells at any two levels are nested or disjoint.
> So no object's cell contains another object's cell, every pair of object-holding cells is
> disjoint, and each object is inside its own cell by the same arithmetic applied at that level's
> own side length. Disjoint boxes, each containing its object.

The lesson generalises: when a structural rule (*holds or subdivides, never both*) can carry the
invariant, it beats a geometric argument about distances, because it stops depending on the sizes
involved. The flat proof had to know that cells were all the same size; this one does not.

**Motion was free for the same kind of reason.** "Objects drift and breathe" sounds like it
invalidates a static non-overlap proof, and it does not, because the proof bounds the *envelope*
— the union of everywhere an object can ever be — rather than where it is now. Two rules keep the
envelope equal to the old static sphere: the drift amplitude is drawn from whatever is *left*
after the static offset (so `|base| + amplitude <= 1`), and the size breathes *downward* from the
cap rather than about the static size. Written the obvious way instead — pick a position, add a
wobble; pick a size, add a pulse — every object would exceed its reach at the top of its breath
and the guarantee would fail everywhere at once, intermittently, which is the worst way for it to
fail.

**A cap measured against the wrong thing looks like a design failure.** The per-root-cell object
cap was tested against the length of the caller's output array rather than against what that root
cell had contributed. With a fresh array per cell it was correct; with one accumulating array the
whole field truncated at 24 objects — 24 across 24,389 root cells. That does not read as an
off-by-one, it reads as "the hierarchy generates nothing", and the temptation is to go and tune
the probabilities. The check that found it counted objects and compared them against the
branching process's expected value, which is the kind of assertion that distinguishes "wrong
number" from "no number".

**Clustering improved the draw order, which was the opposite of the risk.** The worry was that
putting objects closer together would break the heuristic that sorts them by nearest point of
bounding sphere. Measured: 7.52% mean weighted inversions with clusters against 8.07% flat. The
heuristic is right whenever the spheres are disjoint, and a hierarchy guarantees disjointness just
as firmly as a grid does — the clusters are *denser*, not *overlapping*. What did tighten is the
worst-case margin, 0.481 world units down to 0.139, which is the number to watch if the depth or
the density ever goes up.

## Stage 6: the camera was the thing that was wrong

**"Aim at mass, and the screen stays full" was half right, and the other half was the wash.** The
dive scored children by apparent size and aimed at their centres, which does keep the screen full
— of the inside of a structure, which has no silhouette in it at any depth. The 28% of stage-4
frames that were a featureless wash were not a rendering fault and not a depth problem on their
own; they were the camera flying at mass. Aiming at the boundary instead — the children that sit
furthest out of their node, and a point half a child-radius out along that direction — is worth
about a third of them. The thing that reads is structure against void, and void is only ever found
at an edge.

**The dive had no alignment guard and therefore no brakes.** `cruise()` scales its step by how
much of the heading points at the target, with a comment recording the measurement that forced it
(2875 blank frames to none). `dive()` closed a fraction of a distance that its own motion could be
*increasing*, and a bare dive accelerates to 3e14 local units away from the attractor. The same
mistake, in the same file, with the fix sitting ten lines above it. The lesson is not about the
bug: it is that `tests/reroot.mjs` had been printing `depth reached 0` for as long as this was
true, and passing, because both of its assertions are on errors, and errors are smallest when
nothing happens. A test whose quantities all shrink toward zero has to assert that the thing it is
measuring actually happened.

**The instrument had the same shape of fault.** `tests/legible.mjs` called a frame "bright" if its
pixels were over 8/255, and the flat background it draws on is 10–16 after gamma. So empty space
was bright, 99.6% of crossing frames were classified as a fog bank, and the one measurement
pointing at the autopilot was pointing at the clear colour. Both faults are the same mistake:
a threshold against an absolute number rather than against what the system does when nothing is
happening.

**Half the frame rate was a different flight, and almost nothing would have shown it.** The
steering followers integrated 1/60 s per call whatever the frame length was, and the forward step
closed a bare fraction of the remaining distance, which does not compose — 0.95 twice is 0.9025,
not 0.90. Every headless measurement runs at 30fps and the browser at 60, so the two had been
drifting apart the whole time, invisibly, because nothing compared them. Both are one-line fixes
(`1 - exp(-speed)`, and pass the step in), and the first makes `speed` mean what its comment
already claimed: a time constant.

**No single dive's outcome is evidence of anything.** Perturb the speed by one part in a million
and the heading is 60 degrees away three seconds later; the depth reached goes from 23 to 27. The
dive's child choice is a discrete decision inside a self-similar structure, so it is chaotic by
construction. The first version of `tests/pilot.mjs` happily reported a 23-against-4 depth win for
the new aim point; over an ensemble of ten sets and six seeds the same comparison is a wash. What
can be asserted about a chaotic system is its *mechanism* — that the aim point is where the
geometry says, that it stays inside `descendMargin`, that the basis stays orthonormal — and the
aggregate behaviour belongs in a probe that averages, which is `legible.mjs`.

**Backing out the way you came in is the least legible shot in the flight.** 81% of climb frames
were a wash against 21% of dive frames. It is the one phase where the camera is guaranteed to be
in the middle of a structure, pointing at it, for its whole duration. Arcing sideways while
reversing costs nothing — the backward component is untouched, so the ascent still takes the same
frames per level — and it brings the void back into the frame.

**Stars were supposed to be impossible here.** The nebula is smooth exponential lobes precisely
because anything with sub-pixel detail crawls when the camera's scale changes by factors of ten.
But the splat shader has solved this since its first version: never ask for anything smaller than
the sampling grid. Flooring each star's angular sigma at 1.3 pixels of the real viewport makes a
point field band-limited by construction, and the measurement says the same thing the dilation
does — the floored field answers a 4x bigger turn with a 3.9x bigger change, while an unfloored
one is already at its worst-pixel ceiling for a quarter-pixel turn. "Deliberately NOT noise" was
the right call for the nebula and the wrong one to generalise from.

**A smooth path sampled twenty times a second is not a smooth animation.** The morph was
C1-continuous, breathing, and measured never to stall or lurch — as a *path*. What reached the
screen was that path sampled at whatever rate the rebuild budget allowed, 18–26 Hz, and held
between samples: 61% of slot-frames drew exactly the frame before. Every test pointed at the
parameters and none at the table the traversal actually reads, which is the only thing a viewer
sees. The fix was already in the file and switched off, with a comment concluding that it belonged
on the GPU — and it did not. It belonged in the read: the two tables are linear in the maps, so the
traversal can lerp them as it indexes them for +3.8% instead of having them lerped for it at a cost
that landed in the cache 20 ms later. **When something looks wrong in a system that measures
itself, suspect the thing nothing measures.**

**Interpolating between the last two samples lags; extrapolating through them does not.** The
obvious reading of "interpolate towards the frame rate" is to blend from the previous build to the
newest one over the interval between them — and that holds the drawn shape at the newest build
whenever the next rebuild is late, which a budget-driven schedule makes routine. Measured, that
clamp costs two thirds of the benefit: 24 Hz against 60. The fraction wanted is a parameter
position, not a progress bar, and it is allowed to exceed 1.

**Arcing out of the dive costs nothing — wrong, on one paired flight.** The prediction above reasoned
that the backward component is untouched, so the ascent takes the same frames per level and the arc
is free. It is free in the climb and expensive afterwards: 200 s each way, the arc costs 37.6%
washed-out frames against 30.5%, and the *following dive* washes 41.0% against 31.6%, because the
arc leaves the camera off to one side of everything and the next dive starts further from any
structure. The climb is still the worst phase; sliding sideways is not what fixes it.

**And the measurement that said so had the fault this file already named once.** `legible.mjs`
gave its autopilot `Math.random`, so the two runs were two different flights of a system that this
repository has already established is chaotic — the same mistake as reading a 23-against-4 depth
win off a single dive, made again by the probe that was written to replace it. The probe takes a
seed now. An A/B of a chaotic flight has to be paired, and a difference has to survive several
seeds before it is a difference.

**A field's representation is part of its type.** `this._bt = 1` instead of `this._bt = 0.5` — one
initialiser, on a number nobody would think twice about — cost 15 ms a rebuild instead of 2.4 for
the first six rebuilds of the run. V8 picks the representation from the first value stored, so an
integer-initialised field that later holds a fraction transitions the object's hidden class and
discards the optimised code for every method that reads its properties. The expensive method was
`reparam`, 50 property reads and a 36,864-iteration loop. Nothing about this is visible in the
source, and the only reason it was found is that `tests/field.mjs` asserts a bound on the first four
frames in milliseconds rather than as a ratio to the median.
