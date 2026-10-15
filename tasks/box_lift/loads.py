"""Loads of the carried box on the hands, for the inverse dynamics (pipeline/kinetics.py).

While the box is off its support (support == 1 in the reference) the hands carry it:
  F = m (g - a)  through the box centre, plus the couple -dH/dt,
from the box pose tracked by its markers and its mass. When it rests on the floor or the
shelf they carry nothing; frames where it is held but still touching a surface (lifting
off, setting down, sliding onto the shelf) are flagged "uncertain": the surface carries an
unknown share there.
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation as sR

from mocap2myo.kinetics import G, lowpass, omega
from pipeline import config


def box_hand_loads(box_centre, box_quat, box_size, box_kg, held, grasp, palm, hand_bodies,
                   rate, lp_hz=6.0, couple_w=400.0):
  """Loads of a carried rigid box on the two hands (subject units, world frame). While
  `held` the hands carry m (g - a) through the box centre and the couple -dH/dt; shared
  between the grasping hands by a minimum-norm split (forces at the palm centres, small
  couples, weight couple_w on the couples). Returns the two loads and the total."""
  bc = lowpass(box_centre, rate, lp_hz)
  ab = np.gradient(np.gradient(bc, axis=0), axis=0) * rate**2
  Rb = sR.from_quat(box_quat[:, [1, 2, 3, 0]]).as_matrix()
  wb = lowpass(omega(Rb, rate), rate, lp_hz)
  s = box_size
  Ibb = box_kg / 12 * np.array([s[1]**2 + s[2]**2, s[0]**2 + s[2]**2, s[0]**2 + s[1]**2])
  Hd = np.gradient(np.einsum("tij,j,tkj,tk->ti", Rb, Ibb, Rb, wb), axis=0) * rate
  F = np.where(held[:, None], box_kg * (G - ab), 0.0)
  M = np.where(held[:, None], -Hd, 0.0)
  T = len(F)
  Fh, Th = np.zeros((T, 2, 3)), np.zeros((T, 2, 3))
  for t in np.where(held)[0]:
    hands = [h for h in range(2) if grasp[t, h]] or [0, 1]
    n = len(hands)
    A = np.zeros((6, 6 * n))
    for i, h in enumerate(hands):
      r = palm[t, h] - box_centre[t]
      A[:3, 6 * i:6 * i + 3] = np.eye(3)
      A[3:, 6 * i:6 * i + 3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
      A[3:, 6 * i + 3:6 * i + 6] = np.eye(3)
    Wi = np.diag(np.tile(np.r_[np.ones(3), np.full(3, 1 / couple_w)], n))
    x = Wi @ A.T @ np.linalg.solve(A @ Wi @ A.T, np.r_[F[t], M[t]])
    for i, h in enumerate(hands):
      Fh[t, h], Th[t, h] = x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6]
  loads = [dict(body=hand_bodies[h], point=palm[:, h], force=Fh[:, h], torque=Th[:, h])
           for h in range(2)]
  return loads, F, M


def kinetics_loads(trial, kin, K, scale, rate):
  """Hook for pipeline/kinetics.py: the box loads on the hands of one trial.
  K: kin.kinematics(Q) of the IK qpos. Returns (loads, extra arrays to save)."""
  ref = np.load(os.path.join(config.REF_DIR, f"{trial}.npz"))
  mk = np.load(os.path.join(config.NPZ_DIR, f"{trial}.npz"))
  k = 1.0 / scale
  palm = K["palm"] * k
  # box centre in the IK frame: the reference's centre plus the floor shift it removed
  centre = (ref["box_pos"].astype(np.float64) + [0, 0, float(mk["floor_z"]) * scale]) * k
  held = ref["support"] == 1
  bottom = ref["box_pos"][:, 2] - ref["box_size"][2] / 2
  near = (np.abs(bottom) < 0.02) | (np.abs(bottom - float(ref["shelf_top"])) < 0.02)
  loads, F, M = box_hand_loads(centre, ref["box_quat"].astype(np.float64),
                               ref["box_size"].astype(np.float64) * k, float(ref["box_mass"]),
                               held, ref["grasp"] > 0.5, palm, kin.hand_b, rate,
                               lp_hz=config.KIN_LOWPASS_HZ)
  extra = dict(held=held, uncertain=held & near, box_force_on_hands=F, box_couple=M,
               box_centre=centre, box_kg=float(ref["box_mass"]), palm=palm,
               hand_force=np.stack([l["force"] for l in loads], 1),
               hand_couple=np.stack([l["torque"] for l in loads], 1))
  return loads, extra
