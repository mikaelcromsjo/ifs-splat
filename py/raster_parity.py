#!/usr/bin/env python3
"""Compare the shader's projection (transcribed and run on the CPU in JS)
against py/splat_raster.py, the reference stage 0 was measured with."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ifs_core as ifs, splat_raster as sr
from stage0 import write_png, side_by_side

js = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/js_raster.json'))
RES = js['RES']
r = ifs.expand(js['depth'], 0.16, 3)
cam = sr.Camera(js['eye'], np.array(js['eye']) + np.array(js['dir']), up=(0, 0, 1),
                fovy=js['fovy'], W=RES, H=RES)
order = np.array(js['order'])
py = sr.render(r['mu'], r['cov'], r['col'], r['alpha'], cam, order)
jsimg = np.array(js['img']).reshape(RES, RES, 3)

e = np.abs(py - jsimg)
print(f"shader-vs-reference on the same {len(order)} splats, same order, same camera")
print(f"  PSNR        {sr.psnr(py, jsimg):6.2f} dB")
print(f"  mean |err|  {e.mean()*255:6.3f} / 255")
print(f"  max  |err|  {e.max()*255:6.2f} / 255")
print(f"  pixels > 2/255: {np.mean(e.max(2) > 2/255)*100:.3f}%")
write_png('out/raster_parity.png', side_by_side(py, jsimg))
write_png('out/raster_parity_diff.png', np.clip(e * 16, 0, 1))
ok = sr.psnr(py, jsimg) > 45 and e.mean() * 255 < 0.5
print("\nRASTER PARITY", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
