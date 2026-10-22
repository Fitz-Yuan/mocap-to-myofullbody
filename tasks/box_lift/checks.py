"""Box lifting, QA: the checks pipeline/qa.py adds per trial (M2M_TASK=box_lift).

Floor and shelf consistency (box bottom while resting vs floor / shelf top),
hand-box consistency while carrying (palm-sphere surface to box surface, palm
centres in the box frame), carry cycles (box lifted off the platform -> back on
it). Everything in the scaled-subject frame of the IK (with M2M_SCALE=segments the subject's
own frame), on the frames qa.py samples.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from mocap2myo import objects as O


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


def qa_trial(D, Q, frames, step, s, rate, m, d, palm=None):
  """D: the trial's marker npz, Q: IK qpos [T, nq], frames: the sampled frames (every
  step-th), s: subject scale, rate: Hz, m / d: the model; palm (optional): (hand bodies,
  palm-sphere centres in them), l / r, of a model scaled to the subject (ik.MarkerModel
  hand_b, palm_c; default the hand spheres of MyoFullBody). Returns (checks, log line)."""
  hand = [m.body("thirdmc_l").id, m.body("thirdmc_r").id]
  sph = O.hand_sphere_local()
  if palm is not None:
    hand, sph = list(palm[0]), dict(zip("lr", palm[1]))
  half = D["box_size"] * s / 2
  bpos, bquat = D["box_pos"] * s, D["box_quat"]
  floor = float(D["floor_z"]) * s
  shelf = float(D["shelf_top_z"]) * s
  palm = np.zeros((len(frames), 2, 3))
  for k, f in enumerate(frames):
    d.qpos[:] = Q[f]
    mujoco.mj_kinematics(m, d)
    for h in range(2):
      R = d.xmat[hand[h]].reshape(3, 3)
      palm[k, h] = d.xpos[hand[h]] + R @ sph["lr"[h]]
  bp, bq = bpos[frames], bquat[frames]
  bz = bp[:, 2]
  v = np.linalg.norm(np.gradient(bp, axis=0), axis=1) * rate / step
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
    "box_bottom_rest_floor_mm": round(float(np.median(bottom[rest_floor]) - floor) * 1e3, 1)
    if rest_floor.any() else None,
    "box_bottom_rest_shelf_mm": round(float(np.median(bottom[rest_shelf]) - shelf) * 1e3, 1)
    if rest_shelf.any() else None,
    "carry_gap_mm_median_lr": [round(float(np.median(gap[carry, h])) * 1e3, 1) for h in range(2)],
    "carry_palm_in_box_frame_lr": [np.round(np.median(loc[carry, h], 0), 3).tolist() for h in range(2)],
    "box_half_size": np.round(half, 3).tolist(),
    "cycles": len(cyc), "cycle_seconds": [round((b - a) / rate, 1) for a, b in cyc],
    "shelf_top_m": round(shelf, 3),
  }
  line = (f"box rest floor/shelf {r['box_bottom_rest_floor_mm']}/{r['box_bottom_rest_shelf_mm']} mm, "
          f"carry palm gap {r['carry_gap_mm_median_lr']} mm, palms in box "
          f"{r['carry_palm_in_box_frame_lr']}, cycles {r['cycles']} {r['cycle_seconds']}")
  return r, line
