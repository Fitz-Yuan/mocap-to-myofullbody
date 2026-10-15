"""Leave-one-marker-out check of the MMH Elevator IK (independent of the fit itself).

The marker residual of the IK only says how tightly the model was fitted to the
markers it saw. Here every body marker is left out in turn: the frame is solved
again without it (warm start = the trial's IK pose, same robust loss / toe prior,
no contact terms) and the model's prediction of the left-out marker is compared
with where it was measured. Large prediction errors on a segment mean the pose
of that segment is not pinned down by the other markers (or the subject
calibration is off), which the residual alone cannot show. The same solve with
all markers gives the residual to compare with.

Sampled frames: one per second per trial. Head markers are skipped (no neck
joint; they only follow the thorax).

  ELEV_NPZ_DIR=... ELEV_IK_DIR=... OMP_NUM_THREADS=1 uv run python \
    logs/rsl_rl/fullbody_liftbox/scripts/elev_ik_lomo.py [trial ...]
Writes <ELEV_IK_DIR>/lomo.json (per trial and per marker: median / p95
prediction error and residual, mm).
"""

import glob
import json
import multiprocessing as mp
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
NPZ = os.environ.get("ELEV_NPZ_DIR", os.path.join(ROOT, "npz"))
IKD = os.environ.get("ELEV_IK_DIR", os.path.join(ROOT, "ik"))
EVERY_S = float(os.environ.get("ELEV_LOMO_EVERY_S", 1.0))


def trial(name):
  import elev_ik as E
  import mmh_ik as mk
  K = np.load(os.path.join(IKD, f"{name}.npz"))
  P, rate, _ = mk.load_trial(os.path.join(NPZ, f"{name}.npz"), lowpass_hz=E.LOWPASS_HZ or None)
  Q = K["qpos"].astype(np.float64)
  mm = mk.MarkerModel()
  mm.set_offsets(np.load(os.path.join(IKD, f"{name.split('_')[0]}_offsets.npz"))["offsets"])
  body = [k for k, l in enumerate(mk.LABELS) if not l.endswith("HD")]
  frames = np.arange(0, min(len(P), len(Q)), int(EVERY_S * rate))
  pred = {k: [] for k in body}
  resid = {k: [] for k in body}
  for f in frames:
    q = Q[f]
    if np.isnan(q).any():
      continue
    x0 = np.r_[q[:3], np.zeros(3), q[mm.adr]]
    kw = dict(mtp_prior=E.MTP_PRIOR, robust=E.ROBUST or None)
    x_all, _ = mm.solve(P[f], x0, q[3:7], **kw)
    M_all = mm.markers(mm.pose(x_all, q[3:7]))
    for k in body:
      if np.isnan(P[f, k, 0]):
        continue
      resid[k].append(np.linalg.norm(M_all[k] - P[f, k]))
      Pk = P[f].copy()
      Pk[k] = np.nan
      x, _ = mm.solve(Pk, x_all, q[3:7], **kw)
      pred[k].append(np.linalg.norm(mm.markers(mm.pose(x, q[3:7]))[k] - P[f, k]))
  out = {}
  for k in body:
    if pred[k]:
      e, r = 1e3 * np.array(pred[k]), 1e3 * np.array(resid[k])
      out[mk.LABELS[k]] = dict(n=len(e), pred_med=round(float(np.median(e)), 1),
                               pred_p95=round(float(np.percentile(e, 95)), 1),
                               resid_med=round(float(np.median(r)), 1))
  return name, out


def main():
  names = sys.argv[1:] or sorted(os.path.basename(p)[:-4]
                                 for p in glob.glob(os.path.join(IKD, "S*_NE.npz")))
  workers = int(os.environ.get("IK_WORKERS", 12))
  res = {}
  with mp.get_context("spawn").Pool(min(len(names), workers)) as pool:
    for name, out in pool.imap_unordered(trial, names):
      res[name] = out
      allp = [v["pred_med"] for v in out.values()]
      print(f"{name}: prediction error median over markers {np.median(allp):.1f} mm "
            f"(worst {max(out, key=lambda l: out[l]['pred_med'])} "
            f"{max(allp):.1f}), residual {np.median([v['resid_med'] for v in out.values()]):.1f} mm",
            flush=True)
  if len(sys.argv) == 1:
    json.dump(res, open(os.path.join(IKD, "lomo.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
