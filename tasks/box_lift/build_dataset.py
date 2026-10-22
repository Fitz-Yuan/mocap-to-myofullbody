"""Box-lifting references (build_reference.py) -> lift motion dataset.

One clip per carry cycle (box on the elevator floor -> shelf -> back on the
floor; clip bounds at the middle of the floor rests between cycles), 100 Hz,
scaled-subject frame with z = 0 at the elevator platform (with M2M_SCALE=segments: the
subject's own size, qpos of the model scaled to that subject, which is not the model a
policy is trained on; the dataset for a common model is built with the default uniform
scaling). The motion format
of motion_io.write, plus the per-clip object / scene descriptor:
  obj_mass [n] kg (the box mass of the reference, 13.6 kg in the case study),
  grip_local [n, 2, 3] (l, r palm-sphere centres in the box frame while grasped =
  the grasp anchors), has_shelf [n], shelf_pos [n, 3] (centre of the board's top
  surface), shelf_half [n, 2] (board half sizes x, y), support [T] (0 floor,
  1 hands, 2 shelf).
obj_kind = "storage_box", obj_size = (x between the hands, y, z). quality_ok:
body-marker IK p95 <= 6 cm over the clip, every frame solved, no root step above
ROOT_STEP_MAX (a frozen-then-jumping reference passed the p95 test: nanpercentile
skips the frozen frames), the pelvis markers > 5 cm off in at most PELVIS_OFF_MAX
frames and the pelvis turning at most PELVIS_W_MAX (where the model cannot reach the
pose -- deep squats past its 120 deg knee / hip limits -- the IK fit loses the pelvis
and snaps back), and not excluded by the manual review (EXCLUDE).

  python tasks/box_lift/build_dataset.py
Input: <M2M_REF_DIR>, <M2M_IK_DIR>. Output: data/<M2M_DATA>/ (motion_<M2M_DATA_STEM>
[_ok]_100hz.npz, initial_poses_..., index.json).
"""

import json
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mocap2myo import markerset as ms  # noqa: E402
from mocap2myo.model import MYOFULLBODY_XML  # noqa: E402
from pipeline import config  # noqa: E402
from tasks.box_lift.motion_io import FPS, TRACKED, write  # noqa: E402

OUT = config.DATA_DIR
STEM = config.DATA_STEM
QUALITY_P95 = 0.06
ROOT_STEP_MAX = 0.05  # m per 10 ms frame (5 m/s; walking is ~0.015)
PELVIS_OFF_MAX = 10  # frames with the pelvis markers > 5 cm off (median of >= 3)
PELVIS_W_MAX = 700.0  # deg/s pelvis rotation
# left out after a manual review of the reference motions: a subject (all cycles) or
# one cycle "<trial>_c<k>"
EXCLUDE = {"S10": "manual review: poor reference overall",
           "S11": "manual review: poor reference overall",
           "S09_Spring_NE_c5": "manual review: right hand never reaches the box"}
IK_DIR = config.IK_DIR
REF_DIR = config.REF_DIR
SHELF_BOARD_MARGIN = 0.02  # m added around the 4 shelf markers


def main():
  config.usage(__doc__)
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  tb = [m.body(b).id for b in TRACKED]
  models = {}  # M2M_SCALE=segments: each subject's model (the tracked bodies' positions)
  out = {k: [] for k in ("gts", "root_pos", "root_quat", "qj", "box_pos", "box_quat",
                         "grasp", "phase")}
  extra = {k: [] for k in ("obj_mass", "grip_local", "has_shelf", "shelf_pos", "shelf_half")}
  support_all, index = [], []
  for name in config.trials(REF_DIR):
    R = np.load(os.path.join(REF_DIR, f"{name}.npz"))
    if float(R["rate"]) != FPS:
      raise SystemExit(f"{name}: the reference is at {float(R['rate']):g} Hz; the dataset "
                       f"is written at {FPS:g} Hz (resample the markers to {FPS:g} Hz first)")
    K = np.load(os.path.join(IK_DIR, f"{name}.npz"))
    config.check_scale(os.path.join(IK_DIR, f"{name}.npz"), K)
    mq, dq = m, d
    if config.SCALE == "segments":
      subj = config.subject_of(name)
      if subj not in models:
        mms = config.marker_model(subject=subj)
        models[subj] = (mms.m, mms.d)
      mq, dq = models[subj]
    body = [i for i, l in enumerate(K["labels"]) if l not in ms.HEAD_LABELS]
    E = K["marker_err"][:, body]
    pel = [i for i, l in enumerate(K["labels"]) if l in ms.PELVIS_LABELS]
    Q = R["qpos"].astype(np.float64)
    sh = R["shelf_xy"]
    lo, hi = sh.min(0) - SHELF_BOARD_MARGIN, sh.max(0) + SHELF_BOARD_MARGIN
    for c, (a, b) in enumerate(R["clips"]):
      q = Q[a:b]
      G = np.zeros((len(q), len(TRACKED), 3))
      for i in range(len(q)):
        dq.qpos[:] = q[i]
        mujoco.mj_kinematics(mq, dq)
        G[i] = dq.xpos[tb]
      grasp = R["grasp"][a:b]
      anyg = np.flatnonzero(grasp.any(1))
      e = E[a:b]
      p95 = float(np.nanpercentile(e, 95))
      unsolved = int(np.isnan(e).all(1).sum())
      root_step = float(np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1).max())
      pe = K["marker_err"][a:b][:, pel]
      k3 = (~np.isnan(pe)).sum(1) >= 3
      pel_off = int((np.nanmedian(pe[k3], 1) > 0.05).sum()) if k3.any() else 0
      rq = Rotation.from_quat(q[:, [4, 5, 6, 3]])
      pel_w = float(np.degrees((rq[1:] * rq[:-1].inv()).magnitude()).max() * FPS)
      out["gts"].append(G)
      out["root_pos"].append(q[:, :3])
      out["root_quat"].append(q[:, 3:7])
      out["qj"].append(q[:, 7:])
      out["box_pos"].append(R["box_pos"][a:b])
      out["box_quat"].append(R["box_quat"][a:b])
      out["grasp"].append(grasp)
      out["phase"].append(R["support"][a:b].astype(np.int8))
      support_all.append(R["support"][a:b].astype(np.int8))
      extra["obj_mass"].append(float(R["box_mass"]))
      extra["grip_local"].append(R["grip_local"])
      extra["has_shelf"].append(True)
      extra["shelf_pos"].append([*(lo + hi) / 2, float(R["shelf_top"])])
      extra["shelf_half"].append((hi - lo) / 2)
      subj, floor = config.subject_of(name), name.split("_")[1]  # trial = <subject>_<floor>_NE
      excl = EXCLUDE.get(subj) or EXCLUDE.get(f"{name}_c{c + 1}")
      index.append({"clip": len(index), "file": f"{name}_c{c + 1}", "subject": subj,
                    "condition": floor, "technique": f"elevator_{floor.lower()}",
                    "frames": int(b - a), "seconds": round((b - a) / FPS, 2),
                    "grab_step": int(anyg[0]) if len(anyg) else -1,
                    "place_step": int(anyg[-1]) if len(anyg) else -1,
                    "object": "storage_box",
                    "obj_size": [round(float(x), 4) for x in R["box_size"]],
                    "shelf_top": round(float(R["shelf_top"]), 3),
                    "marker_err_mm_median": round(1e3 * float(np.nanmedian(e)), 1),
                    "marker_err_mm_p95": round(1e3 * p95, 1),
                    "unsolved_frames": unsolved, "root_step_max_m": round(root_step, 3),
                    "pelvis_off_frames": pel_off, "pelvis_rot_max_dps": round(pel_w),
                    "excluded": excl or "",
                    "quality_ok": bool(not excl and p95 <= QUALITY_P95 and unsolved == 0
                                       and root_step <= ROOT_STEP_MAX
                                       and pel_off <= PELVIS_OFF_MAX
                                       and pel_w <= PELVIS_W_MAX)})
  os.makedirs(OUT, exist_ok=True)
  ok = [c["clip"] for c in index if c["quality_ok"]]
  for tag, keep in (("", list(range(len(index)))), ("_ok", ok)):
    obj_mass = np.array([extra["obj_mass"][i] for i in keep], np.float32)
    write(OUT, tag, out, index, m, keep, STEM, box_mass=np.median(obj_mass))
    f = os.path.join(OUT, f"motion_{STEM}{tag}_100hz.npz")
    z = dict(np.load(f))
    z["obj_mass"] = obj_mass
    z["grip_local"] = np.array([extra["grip_local"][i] for i in keep], np.float32)
    z["has_shelf"] = np.array([extra["has_shelf"][i] for i in keep])
    z["shelf_pos"] = np.array([extra["shelf_pos"][i] for i in keep], np.float32)
    z["shelf_half"] = np.array([extra["shelf_half"][i] for i in keep], np.float32)
    z["support"] = np.concatenate([support_all[i] for i in keep])
    np.savez(f, **z)
  json.dump({"clips": index, "ok_clips_in_order": ok,
             "source": "tasks/box_lift/build_reference.py references"},
            open(os.path.join(OUT, "index.json"), "w"), indent=1)
  n = sum(c["frames"] for c in index)
  print(f"{len(index)} clips, {n / FPS / 60:.1f} min; {len(ok)} quality-ok "
        f"({sum(index[i]['frames'] for i in ok) / FPS / 60:.1f} min); clip length "
        f"{min(c['seconds'] for c in index)}-{max(c['seconds'] for c in index)} s")


if __name__ == "__main__":
  main()
