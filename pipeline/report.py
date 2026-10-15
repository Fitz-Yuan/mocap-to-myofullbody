"""Per-trial accuracy report of an IK run (one table, one JSON).

Collects what the other steps measured and adds the checks a small marker
residual cannot show:
  input      body-marker gaps (share of frames missing, over markers that were
             placed: a marker never seen is listed as absent), mislabel rejections
  fit        body-marker residual median / p95 (qa.py), IK joint flips
  predict    leave-one-marker-out prediction error (validate/lomo.py, if run)
  posture    quiet-standing posture: median knee and hip flexion over frames
             where the pelvis and both ankles move < 5 cm/s for 0.5 s and the
             pelvis is within 3 cm of its upright height (quiet standing: knee
             ~0-5 deg, hip ~0 deg). Subject calibration can trade pelvis tilt for
             hip / knee flexion, which shifts every frame of that subject
             (mocap2myo.ik.standing).
  clips      the task dataset's clips kept / dropped and why (box_lift: carry
             cycles, tasks/box_lift/build_dataset.py), if it was built
Each trial gets PASS / WARN / FAIL per check against the limits in LIMITS;
WARN and FAIL name the reason. Nothing is dropped here: the dataset's own
quality_ok decides what is trained on.

  M2M_NPZ_DIR=... M2M_IK_DIR=... M2M_DATA=... python pipeline/report.py
Writes <M2M_IK_DIR>/report.json and report.md.
"""

import collections
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mocap2myo import ik as mk  # noqa: E402
from mocap2myo.model import MYOFULLBODY_XML  # noqa: E402
from pipeline import config  # noqa: E402

NPZ = config.NPZ_DIR
IKD = config.IK_DIR
DATA = config.DATA_DIR
# (warn above, fail above); mm, share of frames, or degrees
LIMITS = {
  "gap_max": (0.20, 0.90),          # worst placed body marker missing
  "resid_med_mm": (12.0, 20.0),
  "resid_p95_mm": (40.0, 60.0),
  "flips": (0.003, 0.01),           # frames with a joint faster than 15 rad/s
  "lomo_med_mm": (20.0, 30.0),      # median over markers of the prediction error
  "stand_knee_deg": (10.0, 20.0),   # quiet standing, worse side
  "stand_hip_deg": (10.0, 20.0),    # quiet standing |hip flexion|, worse side
}


def grade(value, key):
  w, f = LIMITS[key]
  return "FAIL" if value > f else "WARN" if value > w else "PASS"


def main():
  config.usage(__doc__)
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  adr = {j: m.jnt_qposadr[m.joint(j).id] for j in
         ("knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_flexion_l")}
  qa = json.load(open(os.path.join(IKD, "qa.json")))
  lomo_p = os.path.join(IKD, "lomo.json")
  lomo = json.load(open(lomo_p)) if os.path.exists(lomo_p) else {}
  # the task dataset's clips, named <trial>_c<k> (tasks/box_lift/build_dataset.py)
  idx_p = os.path.join(DATA, "index.json")
  clips = None
  if os.path.exists(idx_p):
    clips = collections.defaultdict(list)
    for c in json.load(open(idx_p))["clips"]:
      clips[c["file"].rsplit("_c", 1)[0]].append(c)
  head = mk.HEAD
  rows = {}
  for n in config.trials(IKD):
    p = os.path.join(IKD, f"{n}.npz")
    D, K = np.load(os.path.join(NPZ, f"{n}.npz")), np.load(p)
    Pall = D["markers"].astype(np.float64)
    P = Pall[:, ~head]
    body = [l for l, h in zip(mk.LABELS, head) if not h]
    gaps = np.isnan(P[..., 0]).mean(0)
    absent = [l for l, g in zip(body, gaps) if g == 1.0]
    gaps[gaps == 1.0] = 0.0
    Q = K["qpos"].astype(np.float64)
    st = mk.standing(Pall, float(D["marker_rate"]))[:len(Q)]
    med = lambda j: float(np.degrees(np.nanmedian(Q[:len(st)][st, adr[j]])))
    knee = max(med(f"knee_angle_{s}") for s in "rl")
    hip = max(abs(med(f"hip_flexion_{s}")) for s in "rl")
    r = dict(
      gap_max=float(gaps.max()),
      gap_worst=body[int(gaps.argmax())], absent=absent, stand_frames=int(st.sum()),
      mislabel_rejected=float(D["mislabel_rejected"].mean()) if "mislabel_rejected" in D else 0.0,
      resid_med_mm=qa[n]["marker_err_body_mm"][0], resid_p95_mm=qa[n]["marker_err_body_mm"][1],
      flips=qa[n]["joint_speed_over_15rad_s_frac"],
      stand_knee_deg=round(knee, 1), stand_hip_deg=round(hip, 1))
    if n in lomo:
      v = lomo[n]
      r["lomo_med_mm"] = float(np.median([x["pred_med"] for x in v.values()]))
      r["lomo_worst"] = max(v, key=lambda l: v[l]["pred_med"])
    if clips is not None:
      cl = clips.get(n, [])
      r["clips_ok"] = sum(c["quality_ok"] for c in cl)
      r["clips_total"] = len(cl)
      r["clips_dropped"] = dict(collections.Counter(c["excluded"] or "failed QA"
                                                    for c in cl if not c["quality_ok"]))
    r["grades"] = {k: grade(r[k], k) for k in LIMITS if k in r}
    r["verdict"] = ("FAIL" if "FAIL" in r["grades"].values()
                    else "WARN" if "WARN" in r["grades"].values() else "PASS")
    rows[n] = r
  json.dump(rows, open(os.path.join(IKD, "report.json"), "w"), indent=1)
  cols = ["gap_max", "resid_med_mm", "resid_p95_mm", "flips", "lomo_med_mm",
          "stand_knee_deg", "stand_hip_deg"]
  lines = ["| trial | verdict | " + " | ".join(cols) + " | clips ok | reasons |",
           "|" + "---|" * (len(cols) + 4)]
  for n, r in rows.items():
    reasons = [f"{k} {r['grades'][k]}" for k in cols if r["grades"].get(k, "PASS") != "PASS"]
    vals = [f"{r[k]:.3f}" if k in ("gap_max", "flips") else f"{r[k]:.1f}" if k in r else "-"
            for k in cols]
    kept = f"{r['clips_ok']}/{r['clips_total']}" if "clips_ok" in r else "-"
    lines.append(f"| {n} | {r['verdict']} | " + " | ".join(vals) +
                 f" | {kept} | {', '.join(reasons) or '-'} |")
  open(os.path.join(IKD, "report.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))
  print(dict(collections.Counter(r["verdict"] for r in rows.values())))


if __name__ == "__main__":
  main()
