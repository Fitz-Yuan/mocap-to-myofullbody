"""Quality checks of an IK run (prepare_markers + solve_ik output), per trial.

Body-marker IK error (head excluded: no neck joint), joint-speed outliers (IK
flips), the lowest sole point of the model vs the measured floor (when the task
measured one, floor_z in the marker npz), and the task's own checks (M2M_TASK;
box_lift: box rest heights, hands on the box while carrying, carry cycles,
tasks/box_lift/checks.py). Everything in the scaled-subject frame of the IK.

  python pipeline/qa.py [trial ...]
Writes <M2M_IK_DIR>/qa.json (all trials only).
"""

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mocap2myo import markerset as ms  # noqa: E402
from mocap2myo.kinematics import (  # noqa: E402
  foot_collision_geoms, sole_height)
from mocap2myo.model import MYOFULLBODY_XML  # noqa: E402
from pipeline import config  # noqa: E402

QA_HZ = 20.0  # frames evaluated per second (every 5th at 100 Hz)
NPZ_DIR = config.NPZ_DIR
IK_DIR = config.IK_DIR


def qa(name, m, d, foot, task):
  D = np.load(os.path.join(NPZ_DIR, f"{name}.npz"))
  K = np.load(os.path.join(IK_DIR, f"{name}.npz"))
  s, rate = float(K["scale"]), float(K["rate"])
  step = max(1, int(round(rate / QA_HZ)))
  Q = K["qpos"].astype(np.float64)
  E = K["marker_err"]
  body = [i for i, l in enumerate(K["labels"]) if l not in ms.HEAD_LABELS]
  floor = float(D["floor_z"]) * s if "floor_z" in D else None
  # joint speed (hinge joints only, rad/s)
  hj = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
  qd = np.abs(np.diff(Q[:, hj], axis=0)) * rate
  frames = np.arange(0, len(Q), step)
  sole = np.zeros(len(frames))
  for k, f in enumerate(frames):
    d.qpos[:] = Q[f]
    mujoco.mj_kinematics(m, d)
    sole[k] = sole_height(m, d, foot)
  r = {
    "seconds": round(len(Q) / rate, 1), "scale": round(s, 4),
    "marker_err_body_mm": [round(float(np.nanmedian(E[:, body])) * 1e3, 1),
                           round(float(np.nanpercentile(E[:, body], 95)) * 1e3, 1)],
    "frames_p95_err_over_40mm": round(float(np.mean(np.nanpercentile(E[:, body], 95, axis=1) > 0.04)), 4),
    "joint_speed_over_15rad_s_frac": round(float(np.mean(qd.max(1) > 15)), 4),
  }
  if floor is not None:
    r["sole_minus_floor_mm_median"] = round(float(np.median(sole) - floor) * 1e3, 1)
  line = (f"{name}: {r['seconds']}s err {r['marker_err_body_mm']} mm, flips "
          f"{r['joint_speed_over_15rad_s_frac']}")
  if floor is not None:
    line += f", sole-floor {r['sole_minus_floor_mm_median']} mm"
  if task is not None:
    extra, note = task.qa_trial(D, Q, frames, step, s, rate, m, d)
    r.update(extra)
    line += f", {note}"
  if floor is not None:
    r["floor_m"] = round(floor, 3)
  return r, line


def main():
  config.usage(__doc__)
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  foot = foot_collision_geoms(m)
  task = config.task()
  names = sys.argv[1:] or config.trials(IK_DIR)
  out = {}
  for n in names:
    out[n], line = qa(n, m, d, foot, task)
    print(line)
  if len(sys.argv) == 1:
    json.dump(out, open(os.path.join(IK_DIR, "qa.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
