"""Diagnose the joint-angle offsets of an IK run on the static trials.

Each subject with a static trial (read by read_raw.py into <M2M_CACHE_DIR>/static/
<static name>.npz, M2M_STATIC_NAME; case study: 5 s standing still, arms down, all
markers visible; S07 has none):
  1. the IK pose of the static trial with the subject's current calibration
     (<M2M_IK_DIR>/<subject>_offsets.npz): knee / hip / ankle / lumbar flexion while
     the subject stands still and upright;
  2. segment lengths from marker pairs (subject markers scaled to the model's
     height, as the IK does) against the same pairs of the model's anatomical
     marker guesses (markerset.anat_world), as ratios: a ratio away from 1 is a
     segment the uniformly scaled model cannot match. With M2M_SCALE=segments, the
     subject's markers in their own size against the model scaled to that subject.
Ratios use bony landmarks only (ASIS, PSIS, knee, ankle, heel, toe, shoulder,
elbow, wrist, finger, C7; Plug-in-Gait labels). Subjects: the anthropometrics file.

  python pipeline/validate/static_pose.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402  (first: it selects the model and marker set)
from mocap2myo import ik as mk  # noqa: E402
from pipeline import prepare_markers as C  # noqa: E402

STATIC = os.path.join(config.CACHE_DIR, "static")  # <static name>.npz (read_raw.py)
IKD = config.IK_DIR
PAIRS = {
  "pelvis width": [("LASI", "RASI")],
  "pelvis depth": [("LASI", "LPSI"), ("RASI", "RPSI")],
  "thigh": [("LASI", "LKNE"), ("RASI", "RKNE")],
  "shank": [("LKNE", "LANK"), ("RKNE", "RANK")],
  "foot": [("LHEE", "LTOE"), ("RHEE", "RTOE")],
  "trunk": [("C7", "LPSI"), ("C7", "RPSI")],
  "shoulders": [("LSHO", "RSHO")],
  "upper arm": [("LSHO", "LELB"), ("RSHO", "RELB")],
  "forearm": [("LELB", "LWRA"), ("RELB", "RWRA")],
  "hand": [("LWRA", "LFIN"), ("RWRA", "RFIN")],
}
JOINTS = ["knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_flexion_l",
          "ankle_angle_r", "ankle_angle_l", "flex_extension"]  # MyoFullBody (Plug-in-Gait PAIRS)
MM = config.marker_model()


def static_markers(subj, anthro, mm=MM):
  """Mean static pose [n, 3] in the IK's scaled-subject frame (Z up; on a model scaled to the
  subject, M2M_SCALE=segments, the subject's own size)."""
  names, A, _ = C.read_take(f"static/{config.static_name(subj)}")
  P, H, _ = C.body_markers(names, A)
  C.place_head(P, H)
  scale = 1.0 if mm.scaling is not None else MM.model_height / (anthro[subj] / 100)
  return np.nanmedian(P, 0) * scale


def main():
  config.usage(__doc__)
  anthro = {s: a["height_cm"] for s, a in config.anthropometrics().items()}
  mm = MM
  mm.set_offsets(mk.anat_offsets(mm))  # the marker set's anatomical placement on the model
  anat = mm.markers(mm.qref)
  L = mm.ms.LABELS

  def seg(P):
    return {k: np.mean([np.linalg.norm(P[L.index(a)] - P[L.index(b)]) for a, b in v])
            for k, v in PAIRS.items()}

  model_len = seg(anat)
  adr = {j: mm.m.jnt_qposadr[mm.m.joint(j).id] for j in JOINTS}
  print("model segment lengths (m):", {k: round(v, 3) for k, v in model_len.items()})
  print(f"{'subj':5s} " + " ".join(f"{k[:9]:>9s}" for k in PAIRS) + " | static IK pose (deg): "
        + " ".join(j.replace("_angle", "").replace("flexion", "flx").replace("flex_extension", "lumbar")
                   for j in JOINTS) + " | err mm")
  for subj in sorted(anthro):
    if not os.path.exists(os.path.join(STATIC, f"{config.static_name(subj)}.npz")):
      print(f"{subj}  no static trial")
      continue
    off = os.path.join(IKD, f"{subj}_offsets.npz")
    if config.SCALE == "segments":  # against the subject's own model (ratios ~1 where measured)
      if not os.path.exists(off):
        print(f"{subj}  no calibration in {IKD}")
        continue
      mm = config.marker_model(subject=subj)
      mm.set_offsets(mk.anat_offsets(mm))
      model_len = seg(mm.markers(mm.qref))
    P = static_markers(subj, anthro, mm)
    ratio = {k: v / model_len[k] for k, v in seg(P).items()}
    pose = ""
    if os.path.exists(off):
      mm.set_offsets(np.load(off)["offsets"])
      pos, q = mk.root_from_pelvis(mm, P)
      x0 = np.r_[pos, np.zeros(3), mm.qref[mm.adr]]
      x, err = mm.solve(P, x0, q, mtp_prior=0.1, robust=0.025)
      qq = mm.pose(x, q)
      pose = " ".join(f"{np.degrees(qq[adr[j]]):6.1f}" for j in JOINTS) + f" | {1e3 * np.nanmedian(err):.1f}"
    print(f"{subj:5s} " + " ".join(f"{ratio[k]:9.3f}" for k in PAIRS) + " | " + pose, flush=True)


if __name__ == "__main__":
  main()
