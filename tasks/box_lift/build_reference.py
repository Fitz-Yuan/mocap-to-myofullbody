"""Per-trial reference of the MMH Elevator trials (input to the replay videos and
to the training dataset): human IK + measured box + scene, one frame.

Frame: scaled subject (markers x scale, see mmh_ik.load_trial), floor at z = 0.
  Scene and box: shifted by -floor (the elevator markers' floor).
  Human: shifted so the median lowest sole point is on the floor while the
  feet are on the elevator platform (kinematics.sole_height); elsewhere the
  feet are up to ~1.5 cm above z = 0 (the force plates are raised). The subjects wore thick-soled
  slides: the barefoot model fitted to their markers floats 2.5-5.5 cm
  (elev_qa), so the human moves down by that much relative to the box; the
  hands hold the end faces, so this moves the grip down along those faces.
Box: Kabsch pose of the 6 markers (elev_convert), centre moved along x to the
  middle between the palms (median over trials). Size: width / height from the
  marker layout (elev_convert.BOX_SIZE); length from the palms -- the palm
  spheres sit beyond the 0.36 m estimate while carrying (end handles), so the
  end faces are put at the median |palm x| - HAND_R over all trials (unscaled).
Shelf: top = median box bottom while the box rests on it (the shelf markers
  sit ~2 cm below the top surface); footprint from the 4 shelf markers.
Grasp [T, 2] (l, r): the box is not resting (on floor / shelf) and that palm
  sphere is within GRASP_D of the box surface; gaps < 0.2 s filled, blips
  < 0.1 s dropped. grip_local [2, 3]: median palm-sphere centre in the box frame
  while grasped (the weld anchors).
Support [T]: 0 floor, 1 hands, 2 shelf. Cycles: box lifted off the floor ->
  set back on it; clip bounds = middle of the floor rests between cycles.

  uv run python logs/rsl_rl/fullbody_liftbox/scripts/elev_build_reference.py
Output: ~/Projects/datasets/MMH_Elevator/ref/<trial>.npz + summary.json
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
IK_DIR = os.environ.get("ELEV_IK_DIR", os.path.join(ROOT, "ik"))
NPZ_DIR = os.environ.get("ELEV_NPZ_DIR", os.path.join(ROOT, "npz"))
REF_DIR = os.environ.get("ELEV_REF_DIR", os.path.join(ROOT, "ref"))
GRASP_D = 0.03  # m, palm-sphere surface to box surface
REST_V = 0.02  # m/s


def clean_flags(g, fps, gap=0.2, blip=0.1):
  g = g.copy()
  for val, n in ((False, gap), (True, blip)):  # fill short gaps, then drop blips
    idx = np.flatnonzero(g == val)
    if not len(idx):
      continue
    for r in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
      if 0 < r[0] and r[-1] < len(g) - 1 and len(r) < n * fps:
        g[r] = not val
  return g


def kinematic_pass(m, d, Q, foot, hand, sph):
  T = len(Q)
  sole, palm, feet_xy = np.zeros(T), np.zeros((T, 2, 3)), np.zeros((T, 2))
  calcn = [m.body("calcn_l").id, m.body("calcn_r").id]
  for f in range(T):
    d.qpos[:] = Q[f]
    mujoco.mj_kinematics(m, d)
    sole[f] = sole_height(m, d, foot)
    feet_xy[f] = d.xpos[calcn][:, :2].mean(0)
    for h in range(2):
      palm[f, h] = d.xpos[hand[h]] + d.xmat[hand[h]].reshape(3, 3) @ sph["lr"[h]]
  return sole, palm, feet_xy


def to_box(p, c, q):
  R = sRot.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
  return np.einsum("nji,nj->ni", R, p - c)


def sdf(loc, half):
  dd = np.abs(loc) - half
  return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def main():
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  foot = foot_collision_geoms(m)
  hand = [m.body("thirdmc_l").id, m.body("thirdmc_r").id]
  sph = O.hand_sphere_local()
  names = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in
                                 glob.glob(os.path.join(IK_DIR, "S*_NE.npz")))
  os.makedirs(REF_DIR, exist_ok=True)
  # pass 1: kinematics, shifts, palm positions in the box frame
  T1 = {}
  for n in names:
    D = np.load(os.path.join(NPZ_DIR, f"{n}.npz"))
    K = np.load(os.path.join(IK_DIR, f"{n}.npz"))
    s, rate = float(K["scale"]), float(K["rate"])
    Q = K["qpos"].astype(np.float64)
    sole, palm, feet_xy = kinematic_pass(m, d, Q, foot, hand, sph)
    floor = float(D["floor_z"]) * s
    # human: soles on z = 0 while standing on the elevator platform (where the
    # box is lifted / set down; the plates near the shelf sit ~1.5 cm higher,
    # the walkway ~1 cm: one flat floor, aligned where the heavy work happens)
    el = D["elevator_markers"][:, :2] * s
    on_el = np.all((feet_xy > el.min(0) + 0.05) & (feet_xy < el.max(0) - 0.05), 1)
    dz_h = -float(np.median(sole[on_el] if on_el.sum() > 100 else sole))
    Q[:, 2] += dz_h
    palm[..., 2] += dz_h
    bpos = D["box_pos"] * s - [0, 0, floor]
    bquat = D["box_quat"]
    v = np.linalg.norm(np.gradient(bpos, axis=0), axis=1) * rate
    loc = np.stack([to_box(palm[:, h], bpos, bquat) for h in range(2)], 1)
    T1[n] = dict(D=D, s=s, rate=rate, Q=Q, palm=palm, loc=loc, bpos=bpos, bquat=bquat, v=v,
                 floor=floor, dz_h=dz_h, sole_float=float(np.median(sole) - floor))
  # box centre along x and length (unscaled, all trials) from the palms while the
  # box is carried: the marker-based centre is off by ~1 cm (hands asymmetric)
  half_x, cx = [], []
  for n, t in T1.items():
    w = (t["bpos"][:, 2] > 0.5) & (t["v"] > 0.2)
    cx.append(np.median(t["loc"][w, :, 0].mean(1)) / t["s"])
  c_x = float(np.median(cx))
  for n, t in T1.items():
    R = sRot.from_quat(t["bquat"][:, [1, 2, 3, 0]]).as_matrix()
    t["bpos"] = t["bpos"] + R[:, :, 0] * c_x * t["s"]
    t["loc"][..., 0] -= c_x * t["s"]
    w = (t["bpos"][:, 2] > 0.5) & (t["v"] > 0.2)
    half_x.append(np.median(np.abs(t["loc"][w, :, 0])) / t["s"])
  L = 2 * (float(np.median(half_x)) - O.HAND_R)
  size_u = np.array([L, *T1[names[0]]["D"]["box_size"][1:]])
  print(f"box length from the palms: {L:.3f} m (per trial half-x {np.round(half_x, 3).tolist()}), "
        f"centre shift along x {1e3 * c_x:+.1f} mm")
  summary = {"box_size_unscaled": size_u.round(4).tolist(), "box_centre_shift_x": round(c_x, 4),
             "trials": {}}
  for n, t in T1.items():
    D, s, rate = t["D"], t["s"], t["rate"]
    half = size_u * s / 2
    bpos, bquat, v, loc = t["bpos"], t["bquat"], t["v"], t["loc"]
    bottom = bpos[:, 2] - half[2]
    sh = D["shelf_markers"] * s
    on_shelf_xy = np.all((bpos[:, :2] > sh[:, :2].min(0) - 0.1) & (bpos[:, :2] < sh[:, :2].max(0) + 0.1), 1)
    rest = v < REST_V
    shelf_top = float(np.median(bottom[rest & on_shelf_xy & (bottom > 0.8)]))
    on_floor = rest & (bottom < 0.05)
    on_shelf = rest & on_shelf_xy & (np.abs(bottom - shelf_top) < 0.03)
    support = np.ones(len(bpos), np.int8)
    support[clean_flags(on_floor, rate)] = 0
    support[clean_flags(on_shelf, rate)] = 2
    gap = sdf(loc, half) - O.HAND_R  # T, 2
    grasp = np.stack([clean_flags((support == 1) & (gap[:, h] < GRASP_D), rate)
                      for h in range(2)], 1)
    held = support == 1
    grip_local = np.stack([np.median(loc[grasp[:, h], h], 0) for h in range(2)])
    # cycles + clip bounds
    fl = support == 0
    ev = np.flatnonzero(np.diff(fl.astype(int)))
    ups = [i + 1 for i in ev if fl[i]]  # floor -> lifted
    downs = [i + 1 for i in ev if not fl[i]]  # lifted -> floor
    cyc = []
    for a in ups:
      b = next((x for x in downs if x > a), None)
      if b is not None and (b - a) / rate > 5:
        cyc.append((a, b))
    bounds = [0] + [(cyc[k][1] + cyc[k + 1][0]) // 2 for k in range(len(cyc) - 1)] + [len(bpos)]
    clips = [(bounds[k], bounds[k + 1]) for k in range(len(cyc))]
    el = D["elevator_markers"] * s
    np.savez_compressed(
      os.path.join(REF_DIR, f"{n}.npz"), qpos=t["Q"].astype(np.float32), rate=rate, scale=s,
      box_pos=bpos.astype(np.float32), box_quat=bquat.astype(np.float32),
      box_size=(size_u * s).astype(np.float32), box_mass=float(D["box_mass"]),
      grasp=grasp.astype(np.float32), grip_local=grip_local.astype(np.float32),
      support=support, cycles=np.array(cyc), clips=np.array(clips),
      shelf_top=shelf_top, shelf_xy=sh[:, :2].astype(np.float32),
      elevator_xy=el[:, :2].astype(np.float32),
      floor_xy=(D["floor_markers"][:, :2] * s).astype(np.float32),
      human_dz=t["dz_h"], scene_dz=-t["floor"])
    r = {"seconds": round(len(bpos) / rate, 1), "scale": round(s, 4),
         "sole_float_mm": round(1e3 * t["sole_float"], 1),
         "shelf_top": round(shelf_top, 3),
         "shelf_marker_minus_top_mm": round(1e3 * (float(np.median(sh[:, 2])) - t["floor"] - shelf_top), 1),
         "box_rest_floor_bottom_mm": round(1e3 * float(np.median(bottom[on_floor])), 1),
         "held_frac": round(float(held.mean()), 3),
         "grasp_frac_of_held_lr": [round(float(grasp[held, h].mean()), 3) for h in range(2)],
         "palm_gap_held_mm_lr": [round(1e3 * float(np.median(gap[held, h])), 1) for h in range(2)],
         "grip_local_lr": np.round(grip_local, 3).tolist(),
         "cycles": len(cyc), "clip_seconds": [round((b - a) / rate, 1) for a, b in clips]}
    summary["trials"][n] = r
    print(n, r)
  json.dump(summary, open(os.path.join(REF_DIR, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
