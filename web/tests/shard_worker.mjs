// One shard of a frame's walk, on its own thread.
//
// It builds the same scene every other thread builds (shard_setup.mjs says
// why that is a module and not a message), then walks only its own contiguous
// run of the frames list, straight into its own region of the shared buffer.
// Nothing is copied back: the main thread reads the region the walk wrote.
import { parentPort, workerData } from 'node:worker_threads';
import { AdaptiveStream } from '../js/adaptive.js';
import { setup, frameList, walkRun, view } from './shard_setup.mjs';

const { sab, splatOffset, cap, steps, lodScale, lo, hi, rodFade, budget } = workerData;
const { world, cam } = setup(steps);
const frames = frameList(world, cam);

const st = new AdaptiveStream(frames[0].att, cap, { buffer: sab, splatOffset });
const per = new Array(frames.length).fill(0);
st.lodScale = lodScale; st.lo = lo; st.hi = hi; st.rodFade = rodFade;
// Frozen: the shard must not run the detail controller.  There is one
// threshold for the whole frame and the main thread owns it -- k threads each
// nudging their own copy would be k different levels of detail in one picture.
st.budgetBy = 'frozen';
if (budget) { st.targetMs = budget.targetMs; st.target = budget.target; }

parentPort.on('message', (m) => {
  if (m.cmd === 'walk') {
    const t0 = performance.now();
    st.begin(cam, view);
    walkRun(st, frames, m.i0, m.i1, per);
    const count = st.end();
    parentPort.postMessage({
      count, ms: performance.now() - t0, hit: st.budgetHit,
      // where the run landed inside this shard's own region
      head: st.offset, nodes: st.nodes,
      // what each object in this run actually cost, for next frame's split
      per: per.slice(m.i0, m.i1), i0: m.i0,
    });
  } else if (m.cmd === 'stop') parentPort.close();
});
parentPort.postMessage({ ready: true, objects: frames.length });
