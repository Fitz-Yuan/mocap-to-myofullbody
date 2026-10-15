"""Diagnose the joint-angle offsets of the MMH Elevator IK on the static trials.

Each subject (except S07) has a 5 s static trial (Raw Take Files/Subject<ss>_Anatomy.tak:
standing still, arms down, all 39 markers visible). For each subject:
  1. the IK pose of the static trial with the subject's current calibration
     (ELEV_IK_DIR/S<ss>_offsets.npz): knee / hip / ankle / lumbar flexion while
     the subject stands still and upright;
  2. segment lengths from marker pairs (subject markers scaled to the model's
     height, as the IK does) against the same pairs of the model's anatomical
     marker guesses (mmh_ik.anat_world), as ratios: a ratio away from 1 is a
     segment the uniformly scaled model cannot match.
Ratios use bony landmarks only (ASIS, PSIS, knee, ankle, heel, toe, shoulder,
elbow, wrist, finger, C7).

  uv run --with olefile python -I logs/rsl_rl/fullbody_liftbox/scripts/elev_static_diag.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import elev_convert as C  # noqa: E402
import mmh_ik as mk  # noqa: E402
import optitrack_tak as tak  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
STATIC = os.path.join(ROOT, "raw_tak", "static")  # S<ss>_Anatomy.tak (raw)
IKD = os.environ.get("ELEV_IK_DIR", os.path.join(ROOT, "ik_tak"))
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
          "ankle_angle_r", "ankle_angle_l", "flex_extension"]


def static_markers(subj, anthro):
  """Mean static pose [39, 3] in the IK's scaled-subject frame (Z up)."""
  tk = tak.read_tak(os.path.join(STATIC, f"{subj}_Anatomy.tak"))
  A = C.zup(tk["xyz"])
  P, H, _ = C.body_markers(tk["names"], A)
  hk = [mk.LABELS.index(l) for l in C.HEAD_PIG]
  P[:, hk], _ = C.assign_head(P, H)
  scale = mk.MODEL_HEIGHT / (anthro[int(subj[1:])] / 100)
  return np.nanmedian(P, 0) * scale


def main():
  rows = np.genfromtxt(os.path.join(ROOT, "meta", "anthropometrics.csv"), delimiter=",",
                       names=True, dtype=None, encoding=None)
  anthro = {int(r["Subject"]): float(r["Height"]) for r in rows}
  anat = np.array([mk.anat_world(l) for l in mk.LABELS])
  L = mk.LABELS

  def seg(P):
    return {k: np.mean([np.linalg.norm(P[L.index(a)] - P[L.index(b)]) for a, b in v])
            for k, v in PAIRS.items()}

  model_len = seg(anat)
  mm = mk.MarkerModel()
  adr = {j: mm.m.jnt_qposadr[mm.m.joint(j).id] for j in JOINTS}
  print("model segment lengths (m):", {k: round(v, 3) for k, v in model_len.items()})
  print(f"{'subj':5s} " + " ".join(f"{k[:9]:>9s}" for k in PAIRS) + " | static IK pose (deg): "
        + " ".join(j.replace("_angle", "").replace("flexion", "flx").replace("flex_extension", "lumbar")
                   for j in JOINTS) + " | err mm")
  for s in range(1, 12):
    subj = f"S{s:02d}"
    if not os.path.exists(os.path.join(STATIC, f"{subj}_Anatomy.tak")):
      print(f"{subj}  no static trial")
      continue
    P = static_markers(subj, anthro)
    ratio = {k: v / model_len[k] for k, v in seg(P).items()}
    off = os.path.join(IKD, f"{subj}_offsets.npz")
    pose = ""
    if os.path.exists(off):
      mm.set_offsets(np.load(off)["offsets"])
      pos, q = mk.root_from_pelvis(mm, P)
      x0 = np.r_[pos, np.zeros(3), mm.m.qpos0[mm.adr]]
      x, err = mm.solve(P, x0, q, mtp_prior=0.1, robust=0.025)
      qq = mm.pose(x, q)
      pose = " ".join(f"{np.degrees(qq[adr[j]]):6.1f}" for j in JOINTS) + f" | {1e3 * np.nanmedian(err):.1f}"
    print(f"{subj:5s} " + " ".join(f"{ratio[k]:9.3f}" for k in PAIRS) + " | " + pose, flush=True)


if __name__ == "__main__":
  main()
