"""Our inverse dynamics (pipeline/kinetics.py) against OpenSim Inverse Dynamics on the
OpenSim IK (pipeline/opensim_id.py), which uses the same external loads: per joint
moment and trial the correlation, the mean difference and the RMS difference over all
frames whose loads are determined, then the median over trials.
Writes <M2M_KIN_DIR>/compare_opensim.json and .md.
  python pipeline/validate/kinetics_compare.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402

NAMES = {"L5S1_ext": "L5/S1 extension", "hip_r_ext": "hip extension R", "hip_l_ext": "hip extension L",
         "hip_r_abd": "hip abduction R", "hip_l_abd": "hip abduction L",
         "knee_r_ext": "knee extension R", "knee_l_ext": "knee extension L",
         "ankle_r_plantar": "ankle plantarflexion R", "ankle_l_plantar": "ankle plantarflexion L",
         "shoulder_r_flex": "shoulder flexion R", "shoulder_l_flex": "shoulder flexion L",
         "elbow_r_flex": "elbow flexion R", "elbow_l_flex": "elbow flexion L"}


def main():
  config.usage(__doc__)
  trials = [t for t in config.trials(config.KIN_DIR)
            if os.path.exists(os.path.join(config.OSIM_ID_DIR, f"{t}.npz"))]
  per = {}
  for t in trials:
    K = np.load(os.path.join(config.KIN_DIR, f"{t}.npz"))
    O = np.load(os.path.join(config.OSIM_ID_DIR, f"{t}.npz"))
    per[t] = {}
    for k in NAMES:
      if f"moment_{k}" not in O.files:
        continue
      a, b = K[k], O[f"moment_{k}"]
      n = min(len(a), len(b))
      m = ~K["uncertain"][:n] & np.isfinite(a[:n]) & np.isfinite(b[:n])
      d = a[:n][m] - b[:n][m]
      per[t][k] = dict(r=float(np.corrcoef(a[:n][m], b[:n][m])[0, 1]), mean_diff=float(d.mean()),
                       rms=float(np.sqrt((d**2).mean())), p95_ours=float(np.percentile(a[:n][m], 95)),
                       p95_osim=float(np.percentile(b[:n][m], 95)),
                       peak_ours=float(a[:n][m].max()), peak_osim=float(b[:n][m].max()))
  med = {k: {s: float(np.median([per[t][k][s] for t in trials if k in per[t]]))
             for s in ("r", "mean_diff", "rms", "p95_ours", "p95_osim")}
         for k in NAMES if any(k in per[t] for t in trials)}
  json.dump(dict(per_trial=per, median=med, trials=len(trials)),
            open(os.path.join(config.KIN_DIR, "compare_opensim.json"), "w"), indent=1)
  lines = [f"Joint moments, ours (MyoFullBody) vs OpenSim ID (Rajagopal 2016), same external loads; "
           f"median over {len(trials)} trials", "",
           "| Moment | r | mean difference (N m) | RMS difference (N m) | 95th pct ours / OpenSim (N m) |",
           "|---|---|---|---|---|"]
  for k, v in med.items():
    lines.append(f"| {NAMES[k]} | {v['r']:.3f} | {v['mean_diff']:+.1f} | {v['rms']:.1f} | "
                 f"{v['p95_ours']:.0f} / {v['p95_osim']:.0f} |")
  open(os.path.join(config.KIN_DIR, "compare_opensim.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
