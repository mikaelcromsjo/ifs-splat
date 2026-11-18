Static bundle. No build step, no server code: it is index.html plus ES modules.

The endless field (stage 4), with the stage-6 flight and sky.  PUBLISHED 2026-11-17 to
https://mikaelcromsjo.com/splat/ on the owner's go-ahead, republished the same day with the
slower edge-following autopilot, the smaller morph steps and the nebula's star field, and again
on 2026-11-18 with the morph interpolated at display rate and the climb's outward arc switched
off.

The published copy no longer carries cpu_raster.js, gpu_ref.js or calibrate.js: they had been
copied up with the rest on the first publish, and cpu_raster.js imports node:zlib, so a browser
that ever loaded it would have failed.  Nothing references them; they are removed.

To re-publish after a change, copy index.html and js/ into the site's web
root (README.txt is left out on purpose).  Served by the existing nginx vhost
with no config change; .js goes out as application/javascript so the modules
load.  Same origin as the page, so the CORS task in root.md does not apply.

Left out of the bundle deliberately:
  cpu_raster.js   the headless renderer -- a test tool, not needed in a browser
  gpu_ref.js      the JS twin of the WGSL kernel, used only by the test
  calibrate.js    an offline measurement, imports cpu_raster
