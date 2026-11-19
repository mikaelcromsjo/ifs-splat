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
