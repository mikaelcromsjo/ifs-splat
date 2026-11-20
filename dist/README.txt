Static bundle. No build step, no server code: it is index.html plus ES modules.

The endless field (stage 4), with the stage-6 flight and sky.  PUBLISHED 2026-11-17 to
https://mikaelcromsjo.com/splat/ on the owner's go-ahead, republished the same day with the
slower edge-following autopilot, the smaller morph steps and the nebula's star field, and again
on 2026-11-18 with the morph interpolated at display rate and the climb's outward arc switched
off, and again the same day with three fixes to that interpolation: the derived state (moments,
radius, scales, translations) is re-gathered when the `interpolate` switch is turned back on
rather than scattered from a snapshot that stopped being taken while it was off; a jitter-table
resize now lands with the rebuild that fills it instead of at the moment it is asked for; and the
extrapolation bound is taken from the pair it belongs to.

Republished 2026-11-19 with the jitter-table resize ramped over four rebuilds rather than
switched (it was the largest remaining jump in the morph, 86x an ordinary frame's own motion),
a link to the write-up on the panel, and the J key: it copies a URL whose fragment holds the
exact frame of the simulation you are looking at -- the pool, all eight slots' build parameters,
the camera, and the 90 frames before the keypress -- so a moment seen in a browser can be sent
back and replayed offline with `node tests/mark.mjs '<link>'`.  js/mark.js is the live probe
behind it and js/snapshot.js the packer; both are browser-only code with no node imports.

Republished 2026-11-20 with the throughput round: the detail controller now measures the whole
frame rather than only the expansion walk, so a cheaper draw turns into more structure instead of
an emptier frame (js/budget.js is new, and the panel gains `gpu` and `detail held by` rows); the
vertex shader's projection is written out instead of left to the matrix operators; the object
scene's two index buffers get a VAO each instead of re-specifying the format every frame; the
splats upload through a ring of three buffer sets rather than over contents the GPU may still be
reading; and the sky is drawn at half resolution and stretched, which is a quarter of its
fragments and is the new `half-res sky` switch on the panel -- the lobes are unchanged by it and
the stars come out twice as wide, so it is a switch and not a silent default.

Republished 2026-11-20, twice more, for two faults in the detail controller that only a real
screen could show -- both reported from one, within the hour of the first publish.

The first charged the splats for the whole frame. The sky, the pool rebuild and the present cost
the same whether there are 300k splats or one, so when that floor alone exceeded the budget no
number of splats satisfied the controller: it pushed the threshold to its clamp and pinned it
there. One splat, 900 px across, and a detail key that could not win. The frame is split now, by
fitting what a splat actually costs.

The second read the refresh period as the least frame interval ever seen. requestAnimationFrame
runs EARLY as well as late -- after a long frame the browser catches up and delivers the next
callback almost immediately -- so a 60 Hz display hands out the occasional 6 ms gap, and a minimum
believes it. The budget became 0.85 x 6 ms and the controller spent the session chasing 200 fps,
taking detail away every frame to get there: a scene that starts as a fractal and shrinks away.
The period is read from a window now, as the largest interval that explains the rest as integer
multiples of itself, which is what a vsync-locked interval always is.

Three query parameters exist because of all that, and are the first thing to try rather than the
last:

    ?frameterm=0    budget on the walk clock alone, as it did before any of the throughput work.
                    The one to reach for if the scene thins out or shrinks away.
    ?refresh=16.7   pin the refresh period instead of observing it.
    ?lod=2.5        start the detail threshold here instead of at 4.

js/shard.js is NOT published: it splits the expansion walk across threads, and in a browser that
needs SharedArrayBuffer and therefore cross-origin isolation, which is two nginx headers and
task 3 in root.md. Nothing in the page imports it yet, so it stays out on the same terms as the
modules below.

The published copy no longer carries cpu_raster.js, gpu_ref.js or calibrate.js: they had been
copied up with the rest on the first publish, and cpu_raster.js imports node:zlib, so a browser
that ever loaded it would have failed.  Nothing references them; they are removed.

To re-publish after a change, run `bin/deploy` (README.txt is left out on
purpose).  It copies index.html and js/ into the site's web root, stamps the
build id into the page and onto every module URL, and then verifies over HTTP
what the server actually returns.  Served by the existing nginx vhost with no
config change; .js goes out as application/javascript so the modules load.
Same origin as the page, so the CORS task in root.md does not apply.

The stamp is why the published copy is not byte-identical to dist/: `?v=<id>`
is appended to each import as it is copied, so a new build cannot be served
from a browser's cache under an old URL.  dist/ and web/ stay unstamped and
identical to each other.  The id reads `<UTC date>-<UTC time>-<hash of what
was published>`, and the page shows it on the panel as `build` and prints it
to the console, so which copy a browser has is answerable from the page.
nginx still sends no Cache-Control for index.html itself, which cannot stamp
the URL it was fetched from -- that is task 2 in root.md.

Left out of the bundle deliberately:
  cpu_raster.js   the headless renderer -- a test tool, not needed in a browser
  gpu_ref.js      the JS twin of the WGSL kernel, used only by the test
  calibrate.js    an offline measurement, imports cpu_raster
