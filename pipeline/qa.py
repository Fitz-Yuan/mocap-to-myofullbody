"""Quality report of the MMH Elevator references (elev_convert + elev_ik output).

Per trial: body-marker IK error (head excluded: no neck joint, K5 mask),
joint-speed outliers (IK flips), floor consistency (lowest sole point of the
model vs the measured floor; box bottom while resting vs floor / shelf top),
hand-box consistency while carrying (palm-sphere surface to box surface, palm
centres in the box frame), carry cycles (box lifted off the elevator floor ->
back on it). Everything in the scaled-subject frame of the IK.

  uv run python logs/rsl_rl/fullbody_liftbox/scripts/elev_qa.py [trial ...]
Writes ~/Projects/datasets/MMH_Elevator/ik/qa.json.
"""

import glob
import json
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mocap2myo import objects as O  # noqa: E402
from mocap2myo.kinematics import (  # noqa: E402
  foot_collision_geoms, sole_height)
from mocap2myo.model import MYOFULLBODY_XML  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
STEP = 5  # evaluate every 5th frame (20 Hz)
NPZ_DIR = os.environ.get("ELEV_NPZ_DIR", os.path.join(ROOT, "npz"))
IK_DIR = os.environ.get("ELEV_IK_DIR", os.path.join(ROOT, "ik"))


def box_local(p, c, q):
  R = sRot.from_quat(q[[1, 2, 3, 0]]).as_matrix()
  return R.T @ (p - c)


def sdf(loc, half):
  d = np.abs(loc) - half
  return np.linalg.norm(np.maximum(d, 0)) + min(d.max(), 0)


def cycles(bz, floor, rate):
  """Box lifted off the floor -> set back down (box centre height)."""
  low = bz < floor + 0.06
  ev = np.flatnonzero(np.diff(low.astype(int)))
  lifts = [i for i in ev if low[i] and not low[i + 1]]
  downs = [i for i in ev if not low[i] and low[i + 1]]
  out = []
  for a in lifts:
    b = next((x for x in downs if x > a), None)
    if b is not None and (b - a) / rate > 5:
      out.append((int(a), int(b)))
  return out


def qa(name, m, d, foot, hand, sph):
  D = np.load(os.path.join(NPZ_DIR, f"{name}.npz"))
  K = np.load(os.path.join(IK_DIR, f"{name}.npz"))
  s, rate = float(K["scale"]), float(K["rate"])
  Q = K["qpos"].astype(np.float64)
  E = K["marker_err"]
  body = [i for i, l in enumerate(K["labels"]) if not l.endswith("HD")]
  half = D["box_size"] * s / 2
  bpos, bquat = D["box_pos"] * s, D["box_quat"]
  floor = float(D["floor_z"]) * s
  shelf = float(D["shelf_top_z"]) * s
  # joint speed (hinge joints only, rad/s)
  hj = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  qd = np.abs(np.diff(Q[:, hj], axis=0)) * rate
  frames = np.arange(0, len(Q), STEP)
  sole = np.zeros(len(frames))
  palm = np.zeros((len(frames), 2, 3))
  for k, f in enumerate(frames):
    d.qpos[:] = Q[f]
    mujoco.mj_kinematics(m, d)
    sole[k] = sole_height(m, d, foot)
    for h in range(2):
      R = d.xmat[hand[h]].reshape(3, 3)
      palm[k, h] = d.xpos[hand[h]] + R @ sph["lr"[h]]
  bp, bq = bpos[frames], bquat[frames]
  bz = bp[:, 2]
  v = np.linalg.norm(np.gradient(bp, axis=0), axis=1) * rate / STEP
  bottom = bz - half[2]  # box upright (it is, while resting)
  rest_floor = (v < 0.02) & (bottom < floor + 0.05)
  rest_shelf = (v < 0.02) & (np.abs(bottom - shelf) < 0.05)
  carry = (bottom > floor + 0.15) & (bottom < shelf - 0.15) & (v > 0.2)
  loc = np.array([[box_local(palm[k, h], bp[k], bq[k]) for h in range(2)]
                  for k in range(len(frames))])
  gap = np.array([[sdf(loc[k, h], half) - O.HAND_R for h in range(2)]
                  for k in range(len(frames))])
  cyc = cycles(bpos[:, 2] - half[2], floor, rate)
  r = {
    "seconds": round(len(Q) / rate, 1), "scale": round(s, 4),
    "marker_err_body_mm": [round(float(np.nanmedian(E[:, body])) * 1e3, 1),
                           round(float(np.nanpercentile(E[:, body], 95)) * 1e3, 1)],
    "frames_p95_err_over_40mm": round(float(np.mean(np.nanpercentile(E[:, body], 95, axis=1) > 0.04)), 4),
    "joint_speed_over_15rad_s_frac": round(float(np.mean(qd.max(1) > 15)), 4),
    "sole_minus_floor_mm_median": round(float(np.median(sole) - floor) * 1e3, 1),
    "box_bottom_rest_floor_mm": round(float(np.median(bottom[rest_floor]) - floor) * 1e3, 1)
    if rest_floor.any() else None,
    "box_bottom_rest_shelf_mm": round(float(np.median(bottom[rest_shelf]) - shelf) * 1e3, 1)
    if rest_shelf.any() else None,
    "carry_gap_mm_median_lr": [round(float(np.median(gap[carry, h])) * 1e3, 1) for h in range(2)],
    "carry_palm_in_box_frame_lr": [np.round(np.median(loc[carry, h], 0), 3).tolist() for h in range(2)],
    "box_half_size": np.round(half, 3).tolist(),
    "cycles": len(cyc), "cycle_seconds": [round((b - a) / rate, 1) for a, b in cyc],
    "shelf_top_m": round(shelf, 3), "floor_m": round(floor, 3),
  }
  return r


def main():
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  foot = foot_collision_geoms(m)
  hand = [m.body("thirdmc_l").id, m.body("thirdmc_r").id]
  sph = O.hand_sphere_local()
  names = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in
                                 glob.glob(os.path.join(IK_DIR, "S*_NE.npz")))
  out = {}
  for n in names:
    out[n] = qa(n, m, d, foot, hand, sph)
    r = out[n]
    print(f"{n}: {r['seconds']}s err {r['marker_err_body_mm']} mm, flips "
          f"{r['joint_speed_over_15rad_s_frac']}, sole-floor {r['sole_minus_floor_mm_median']} mm, "
          f"box rest floor/shelf {r['box_bottom_rest_floor_mm']}/{r['box_bottom_rest_shelf_mm']} mm, "
          f"carry palm gap {r['carry_gap_mm_median_lr']} mm, palms in box {r['carry_palm_in_box_frame_lr']}, "
          f"cycles {r['cycles']} {r['cycle_seconds']}")
  if len(sys.argv) == 1:
    json.dump(out, open(os.path.join(IK_DIR, "qa.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
