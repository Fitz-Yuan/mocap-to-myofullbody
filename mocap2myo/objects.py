"""Hand contact spheres: one sphere per hand (radius HAND_R) in the palm.

The sphere stands for the palm's soft tissue over the metacarpals. The pipeline uses
its centre as the palm point: where a carried object's load is applied in the
inverse dynamics (mocap2myo.kinetics via ik.MarkerModel.palms), and how near a hand
is to the object (the task's grasp detection, tasks/box_lift/).
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

from mocap2myo.model import MYOFULLBODY_XML

HAND_R = 0.02  # m: palm soft tissue over the metacarpals
SPHERE_PALMAR = 0.0  # sphere centre on the metacarpal (surface = palm skin)
SPHERE_DISTAL = 0.015  # and toward the fingers


@lru_cache(maxsize=1)
def hand_sphere_local() -> dict[str, np.ndarray]:
  """Sphere centre in the thirdmc_{l,r} body frame.

  Palm normal from the hand bones' chirality (posture independent):
  across = index - little metacarpal, distal = proximal phalanx 3 - thirdmc;
  palmar = across x distal (right hand), distal x across (left hand).
  """
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  p = lambda n: d.xpos[m.body(n).id]
  out = {}
  for s in ("l", "r"):
    across = p(f"secondmc_{s}") - p(f"fifthmc_{s}")
    distal = p(f"3proxph_{s}") - p(f"thirdmc_{s}")
    distal /= np.linalg.norm(distal)
    n = np.cross(across, distal) if s == "r" else np.cross(distal, across)
    n /= np.linalg.norm(n)
    w = p(f"thirdmc_{s}") + SPHERE_PALMAR * n + SPHERE_DISTAL * distal
    R = d.xmat[m.body(f"thirdmc_{s}").id].reshape(3, 3)
    out[s] = R.T @ (w - p(f"thirdmc_{s}"))
  return out


@lru_cache(maxsize=1)
def hand_palmar_local() -> dict[str, np.ndarray]:
  """Unit palm normal (pointing out of the palm) in the thirdmc frame."""
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  p = lambda n: d.xpos[m.body(n).id]
  out = {}
  for s in ("l", "r"):
    across = p(f"secondmc_{s}") - p(f"fifthmc_{s}")
    distal = p(f"3proxph_{s}") - p(f"thirdmc_{s}")
    n = np.cross(across, distal) if s == "r" else np.cross(distal, across)
    R = d.xmat[m.body(f"thirdmc_{s}").id].reshape(3, 3)
    out[s] = R.T @ (n / np.linalg.norm(n))
  return out
