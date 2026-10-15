"""MyoFullBody kinematic helpers: joint couplings, sole height, rigid fits."""

from __future__ import annotations

import mujoco
import numpy as np


def apply_joint_equalities(m: mujoco.MjModel, q: np.ndarray) -> None:
  """Set every joint-equality follower from its leader, in place (full qpos).

  MuJoCo joint equality: q1 - q1_0 = c0 + c1 x + c2 x^2 + c3 x^3 + c4 x^4 with
  x = q2 - q2_0 (q_0 = qpos0); with no leader, q1 - q1_0 = c0.
  """
  for e in range(m.neq):
    if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
      continue
    j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
    a1 = m.jnt_qposadr[j1]
    c = m.eq_data[e][:5]
    if j2 < 0:
      q[a1] = m.qpos0[a1] + c[0]
    else:
      a2 = m.jnt_qposadr[j2]
      x = q[a2] - m.qpos0[a2]
      q[a1] = m.qpos0[a1] + c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


# --- foot soles ------------------------------------------------------------

FOOT_BODIES = ("calcn_l", "calcn_r", "toes_l", "toes_r")


def foot_collision_geoms(m: mujoco.MjModel, prefix: str = "") -> list[int]:
  """Collision geoms of the four foot bodies (heels, toes)."""
  names = {prefix + b for b in FOOT_BODIES}
  return [g for g in range(m.ngeom)
          if m.geom_contype[g] and m.body(m.geom_bodyid[g]).name in names]


def geom_lowest_z(gtype, size, pos_z, rot_row2):
  """Exact lowest world z of capsule / ellipsoid / sphere / box geoms.

  pos_z: centre z; rot_row2: third row of the geom rotation (world z of the
  geom's local axes); numpy arrays (last dim 3).
  Capsule: lower end of the axis minus the radius; ellipsoid: support point
  along -z, sqrt(sum (R[2,i] s_i)^2); box: sum |R[2,i]| s_i.
  (The simpler centre z - size[0] is wrong for the heel ellipsoid: size[0] is its
  8.5 cm half-length, not its ~4 cm vertical half-axis.)
  """
  cap = pos_z - np.abs(rot_row2[..., 2]) * size[..., 1] - size[..., 0]
  ell = pos_z - np.sqrt(((rot_row2 * size) ** 2).sum(-1))
  sph = pos_z - size[..., 0]
  box = pos_z - (np.abs(rot_row2) * size).sum(-1)
  return np.where(gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE), cap,
         np.where(gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), ell,
         np.where(gtype == int(mujoco.mjtGeom.mjGEOM_BOX), box, sph)))


def sole_height(m: mujoco.MjModel, d: mujoco.MjData, geoms: list[int]) -> float:
  """Lowest point of the given foot geoms for the pose in d (kinematics done)."""
  g = np.asarray(geoms)
  rot = d.geom_xmat[g].reshape(-1, 3, 3)[:, 2, :]
  return float(geom_lowest_z(m.geom_type[g], m.geom_size[g], d.geom_xpos[g, 2], rot).min())


# --- rigid fits --------------------------------------------------------------

def kabsch(A, B):
  """R, t minimising |R A + t - B| (rows = points)."""
  ca, cb = A.mean(0), B.mean(0)
  U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
  D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
  R = Vt.T @ D @ U.T
  return R, cb - R @ ca
