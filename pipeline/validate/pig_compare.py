"""Our joint angles and moments against the Plug-in-Gait model outputs a Vicon C3D carries.

Vicon Nexus stores its Plug-in-Gait results as POINT channels (LHipAngles, LKneeMoment, ...);
pipeline/read_raw.py keeps them in the cache as other_labels / other_data. This compares, per
trial and side, hip flexion / adduction, knee flexion, ankle dorsiflexion and elbow flexion,
and the hip / knee / ankle sagittal moments: correlation, mean difference (offset) and RMS
difference after removing that offset. The sign of each Vicon moment against ours is taken
from the data (reported) and the unit from the size of the values (N mm, or N mm per kg).
Frames where Vicon has no value (NaN or exactly 0) are skipped, and moments are compared only
while that side's foot is on a force plate (Vicon's moments leave the ground out otherwise).

Writes <M2M_KIN_DIR>/pig_compare.json and .md.
  python pipeline/validate/pig_compare.py [trial ...]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402

ANGLES = [("hip flexion", "HipAngles", 0, "hip_flexion"), ("hip adduction", "HipAngles", 1, "hip_adduction"),
          ("knee flexion", "KneeAngles", 0, "knee_angle"), ("ankle dorsiflexion", "AnkleAngles", 0, "ankle_angle"),
          ("elbow flexion", "ElbowAngles", 0, "elbow_flex")]
MOMENTS = [("hip extension", "HipMoment", 0, "hip_{s}_ext"), ("knee extension", "KneeMoment", 0, "knee_{s}_ext"),
           ("ankle plantarflexion", "AnkleMoment", 0, "ankle_{s}_plantar")]


def stats(a, b):
  ok = np.isfinite(a) & np.isfinite(b)
  if ok.sum() < 10:
    return None
  a, b = a[ok], b[ok]
  d = a - b
  return dict(n=int(ok.sum()), r=float(np.corrcoef(a, b)[0, 1]), offset=float(d.mean()),
              rms_after=float(d.std()), rms=float(np.sqrt((d**2).mean())))


def vicon(cache, label, axis):
  labels = list(cache["other_labels"])
  if label not in labels:
    return None
  x = cache["other_data"][:, labels.index(label), axis].astype(np.float64)
  x[(x == 0) | ~np.isfinite(x)] = np.nan
  return x


def compare(trial):
  cache = np.load(os.path.join(config.CACHE_DIR, f"{trial}.npz"))
  ik = np.load(os.path.join(config.IK_DIR, f"{trial}.npz"))
  names = list(ik["joint_names"])
  import mujoco  # noqa: F401  (the model's qpos layout comes from the IK file's joint names)
  mm = config.marker_model()
  m = mm.m
  angle = {"hip_flexion": "hip_flex", "hip_adduction": "hip_add", "knee_angle": "knee_flex",
           "ankle_angle": "ankle_dorsi", "elbow_flex": "elbow_flex"}  # -> the profile's [angles]
  Q = ik["qpos"].astype(np.float64)
  out = {}
  for side, S in (("l", "L"), ("r", "R")):
    for name, lab, ax, j in ANGLES:
      v = vicon(cache, f"{S}{lab}", ax)
      a = mm.profile.angle(f"{angle[j]}_{side}")
      if v is None or a is None or a[0] not in names:
        continue
      ours = np.degrees(a[1] * Q[:, m.jnt_qposadr[m.joint(a[0]).id]])
      n = min(len(v), len(ours))
      out[f"{name} {S}"] = stats(ours[:n], v[:n])
  kp = os.path.join(config.KIN_DIR, f"{trial}.npz")
  if os.path.exists(kp):
    K = np.load(kp)
    mass = float(K["body_kg"]) + float(K["extra_kg"])
    for side, S in (("l", "L"), ("r", "R")):
      for name, lab, ax, key in MOMENTS:
        v = vicon(cache, f"{S}{lab}", ax)
        k = key.format(s=side)
        if v is None or k not in K.files:
          continue
        ours = K[k].astype(np.float64)
        n = min(len(v), len(ours))
        # Plug-in-Gait counts the ground reaction only while the foot is on a plate; with the
        # foot on the floor beside the plates its moments leave the ground out. Compare only
        # where Vicon has this side's ground reaction.
        grf = vicon(cache, f"{S}GroundReactionForce", 2)
        if grf is None:
          continue
        v = np.where(np.isfinite(grf[:n]), v[:n], np.nan) / 1000.0  # N mm -> N m
        if np.nanmax(np.abs(v)) < 10:  # Plug-in-Gait often normalises by body mass
          v = v * mass
        st = stats(ours[:n], v)
        if st is None:
          continue
        sign = 1.0
        if st["r"] < 0:  # Vicon's axis points the other way for this joint
          sign, st = -1.0, stats(ours[:n], -v)
        st["vicon_sign"] = sign
        out[f"{name} moment {S}"] = st
  return out


def main():
  config.usage(__doc__)
  trials = sys.argv[1:] or [t for t in config.trials(config.IK_DIR)
                            if os.path.exists(os.path.join(config.CACHE_DIR, f"{t}.npz"))]
  per = {}
  for t in trials:
    per[t] = {k: v for k, v in compare(t).items() if v}
    print(t, {k: round(v["r"], 3) for k, v in per[t].items()}, flush=True)
  keys = sorted({k for v in per.values() for k in v})
  med = {k: {s: float(np.median([per[t][k][s] for t in per if k in per[t]])) for s in ("r", "offset", "rms_after", "rms")}
         for k in keys}
  os.makedirs(config.KIN_DIR, exist_ok=True)
  json.dump(dict(per_trial=per, median=med, trials=len(per)),
            open(os.path.join(config.KIN_DIR, "pig_compare.json"), "w"), indent=1)
  lines = [f"Ours vs Vicon Plug-in-Gait, median over {len(per)} trials", "",
           "| Quantity | r | offset | RMS after the offset |", "|---|---|---|---|"]
  for k, v in med.items():
    u = "N m" if "moment" in k else "deg"
    lines.append(f"| {k} | {v['r']:.3f} | {v['offset']:+.1f} {u} | {v['rms_after']:.1f} {u} |")
  open(os.path.join(config.KIN_DIR, "pig_compare.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
