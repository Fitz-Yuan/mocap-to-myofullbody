"""Per-trial accuracy report of an IK run (one table, one JSON).

Collects what the other steps measured and adds the checks a small marker
residual cannot show:
  input      body-marker gaps (share of frames missing, over markers that were
             placed: a marker never seen is listed as absent), mislabel rejections
  fit        body-marker residual median / p95 (qa.py), the median frame's RMS and largest
             marker error against OpenSim's IK guideline (< 2 cm, < 2-4 cm), IK joint flips
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
quality_ok decides which clips it keeps.

  M2M_NPZ_DIR=... M2M_IK_DIR=... M2M_DATA=... python pipeline/report.py
Writes <M2M_IK_DIR>/report.json and report.md.
"""

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402  (first: it selects the model and marker set)
from mocap2myo import ik as mk  # noqa: E402

NPZ = config.NPZ_DIR
IKD = config.IK_DIR
DATA = config.DATA_DIR
# (warn above, fail above); mm, share of frames, or degrees
LIMITS = {
  "gap_max": (0.20, 0.90),          # worst placed body marker missing
  "resid_med_mm": (12.0, 20.0),
  "resid_p95_mm": (40.0, 60.0),
  # OpenSim's IK guideline (warn above it; fail at 1.5x): the median frame's RMS < 2 cm and
  # largest body-marker error < 4 cm
  "frame_rms_med_mm": (20.0, 30.0),
  "frame_max_med_mm": (40.0, 60.0),
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
  profile, ms = config.model_profile(), config.marker_set()
  m = profile.compile()
  # quiet-standing posture: the profile's knee / hip flexion angles (joint, sign)
  adr = {a: (m.jnt_qposadr[m.joint(profile.angle(a)[0]).id], profile.angle(a)[1])
         for a in ("knee_flex_r", "knee_flex_l", "hip_flex_r", "hip_flex_l")}
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
  head = ms.HEAD
  rows = {}
  for n in config.trials(IKD):
    p = os.path.join(IKD, f"{n}.npz")
    D, K = np.load(os.path.join(NPZ, f"{n}.npz")), np.load(p)
    Pall = D["markers"].astype(np.float64)
    P = Pall[:, ~head]
    body = [l for l, h in zip(ms.LABELS, head) if not h]
    gaps = np.isnan(P[..., 0]).mean(0)
    absent = [l for l, g in zip(body, gaps) if g == 1.0]
    gaps[gaps == 1.0] = 0.0
    Q = K["qpos"].astype(np.float64)
    st = mk.standing(Pall, float(D["marker_rate"]), ms=ms)[:len(Q)]
    med = lambda a: float(np.degrees(adr[a][1] * np.nanmedian(Q[:len(st)][st, adr[a][0]])) if adr[a][1] != 1
                          else np.degrees(np.nanmedian(Q[:len(st)][st, adr[a][0]])))
    knee = max(med(f"knee_flex_{s}") for s in "rl")
    hip = max(abs(med(f"hip_flex_{s}")) for s in "rl")
    r = dict(
      gap_max=float(gaps.max()),
      gap_worst=body[int(gaps.argmax())], absent=absent, stand_frames=int(st.sum()),
      mislabel_rejected=float(D["mislabel_rejected"].mean()) if "mislabel_rejected" in D else 0.0,
      resid_med_mm=qa[n]["marker_err_body_mm"][0], resid_p95_mm=qa[n]["marker_err_body_mm"][1],
      **({"frame_rms_med_mm": qa[n]["frame_rms_mm"][0], "frame_max_med_mm": qa[n]["frame_max_mm"][0]}
         if "frame_rms_mm" in qa[n] else {}),
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
  cols = ["gap_max", "resid_med_mm", "resid_p95_mm", "frame_rms_med_mm", "frame_max_med_mm", "flips",
          "lomo_med_mm", "stand_knee_deg", "stand_hip_deg"]
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
