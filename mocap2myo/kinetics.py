"""Joint loads (net forces and moments) from IK on MyoFullBody: inverse dynamics.

Newton-Euler on subtrees, in the subject's own size and mass. For a joint with point P,
the force and moment the parent segment exerts on everything distal to it:
  F   = sum_b m_b (a_b - g) - sum_e F_e
  M_P = sum_b [(c_b - P) x m_b (a_b - g) + dH_b/dt] - sum_e [(p_e - P) x F_e + T_e]
over the distal bodies b and the external loads e acting on them.

Top-down (no ground reaction needed): L5/S1, shoulders, elbows; the carried object
enters as external loads on the hands. The same sum over the whole body gives the total
ground reaction (force, centre of pressure, free moment). Bottom-up for the legs: the
total ground reaction is assigned to the loaded foot; in double support it is shared by
the lever rule along the line between the two feet's contact centres (an estimate: the
split is not determined by kinematics), each foot's share applied under that foot.

Conventions: moments are internal (net muscle) moments in N m, reported with the sign
named in the key (e.g. knee_r_ext > 0 = net extensor moment). Anatomical frames follow
the model / OpenSim: x anterior, y superior, z to the right.
"""

import mujoco
import numpy as np
from scipy.signal import butter, filtfilt

G = np.array([0.0, 0.0, -9.81])
D_EXT, D_FLEX = 0.05, 0.08  # erector spinae / abdominal moment arms about L5/S1 (m)


def lowpass(x, rate, hz):
  """Zero-phase 4th-order Butterworth low-pass along axis 0."""
  if not 0 < hz < rate / 2:
    raise ValueError(f"low-pass {hz} Hz must lie between 0 and half the rate ({rate:g} Hz)")
  b, a = butter(4, hz / (rate / 2))
  return filtfilt(b, a, x, axis=0)


def omega(R, rate):
  """World angular velocity [T, ..., 3] from rotation matrices [T, ..., 3, 3]."""
  W = (np.gradient(R, axis=0) * rate) @ np.swapaxes(R, -1, -2)
  return 0.5 * np.stack([W[..., 2, 1] - W[..., 1, 2], W[..., 0, 2] - W[..., 2, 0],
                         W[..., 1, 0] - W[..., 0, 1]], -1)


def subtree(m, root):
  out = []
  for b in range(root, m.nbody):
    p = b
    while p > root:
      p = m.body_parentid[p]
    if p == root:
      out.append(b)
  return np.array(out)


# joint: distal body, point (joint anchor or body origin), frame body, reported components
# (name, axis source, sign): axis "x"/"y"/"z" of the frame, or "jnt:<joint>" (its axis)
JOINTS = {
  "L5S1": ("lumbar5", "jnt:flex_extension", "lumbar5",
           [("ext", "z", 1), ("latbend", "x", 1), ("twist", "y", 1)]),
  "shoulder_r": ("humerus_r", "body:humerus_r", "thorax", [("flex", "z", 1), ("abd", "x", -1)]),
  "shoulder_l": ("humerus_l", "body:humerus_l", "thorax", [("flex", "z", 1), ("abd", "x", 1)]),
  "elbow_r": ("ulna_r", "jnt:elbow_flex_r", "ulna_r", [("flex", "jnt:elbow_flex_r", 1)]),
  "elbow_l": ("ulna_l", "jnt:elbow_flex_l", "ulna_l", [("flex", "jnt:elbow_flex_l", 1)]),
  "hip_r": ("femur_r", "jnt:hip_flexion_r", "pelvis",
            [("ext", "jnt:hip_flexion_r", -1), ("abd", "jnt:hip_adduction_r", -1)]),
  "hip_l": ("femur_l", "jnt:hip_flexion_l", "pelvis",
            [("ext", "jnt:hip_flexion_l", -1), ("abd", "jnt:hip_adduction_l", -1)]),
  "knee_r": ("tibia_r", "jnt:knee_angle_r", "tibia_r", [("ext", "jnt:knee_angle_r", -1)]),
  "knee_l": ("tibia_l", "jnt:knee_angle_l", "tibia_l", [("ext", "jnt:knee_angle_l", -1)]),
  "ankle_r": ("talus_r", "jnt:ankle_angle_r", "tibia_r", [("plantar", "jnt:ankle_angle_r", -1)]),
  "ankle_l": ("talus_l", "jnt:ankle_angle_l", "tibia_l", [("plantar", "jnt:ankle_angle_l", -1)]),
}
TOP_DOWN = ("L5S1", "shoulder_r", "shoulder_l", "elbow_r", "elbow_l")


class Kinetics:
  """mm: an ik.MarkerModel (MyoFullBody with marker sites, palms and sole points)."""

  def __init__(self, mm):
    self.mm, self.m, self.d = mm, mm.m, mm.d
    m = self.m
    self.sets = {j: subtree(m, m.body(v[0]).id) for j, v in JOINTS.items()}
    self.all = subtree(m, m.body("Full Body").id)
    self.hand_b = [m.body("thirdmc_l").id, m.body("thirdmc_r").id]
    self.foot_b = {s: m.body(f"calcn_{s}").id for s in "lr"}
    self.sole = {s: np.array([i for i, k in enumerate(mm.fp_kind) if k[0] == s]) for s in "lr"}

  def kinematics(self, Q):
    m, d = self.m, self.d
    T = len(Q)
    C, R = np.zeros((T, m.nbody, 3)), np.zeros((T, m.nbody, 3, 3))
    P, Fr, Ax = {}, {}, {}
    for j in JOINTS:
      P[j], Fr[j] = np.zeros((T, 3)), np.zeros((T, 3, 3))
    axes = sorted({a[1] for v in JOINTS.values() for a in v[3]} | {v[1] for v in JOINTS.values()})
    axes = [a for a in axes if a.startswith("jnt:")]
    for a in axes:
      Ax[a] = np.zeros((T, 3))
    palm = np.zeros((T, 2, 3))
    sole = np.zeros((T, len(self.mm.fp_kind), 3))
    for t in range(T):
      d.qpos[:] = Q[t]
      mujoco.mj_kinematics(m, d)
      C[t], R[t] = d.xipos, d.ximat.reshape(-1, 3, 3)
      for j, (_, pt, fr, _) in JOINTS.items():
        kind, name = pt.split(":")
        P[j][t] = d.xanchor[m.joint(name).id] if kind == "jnt" else d.xpos[m.body(name).id]
        Fr[j][t] = d.xmat[m.body(fr).id].reshape(3, 3)
      for a in axes:
        Ax[a][t] = d.xaxis[m.joint(a[4:]).id]
      palm[t] = self.mm.palms()[0]
      sole[t] = self.mm.foot_points()
    return dict(C=C, R=R, P=P, Fr=Fr, Ax=Ax, palm=palm, sole=sole)

  def solve(self, K, rate, scale, body_kg, extra_kg=None, loads=(), floor_z=None, lp_hz=6.0, plates=None,
            legs=True):
    """K: self.kinematics(Q) for IK qpos Q in model units (the model is uniformly `scale` x
    the subject).
    extra_kg: {body name: kg} carried on the body (lumped at that body's centre of mass).
    loads: external loads on bodies, subject units / world frame, each a dict(body=id,
    point=[T,3], force=[T,3], torque=[T,3]). floor_z: floor height [T] (subject units) as
    the model's soles see it; default: the lowest sole point of either foot within +-1.5 s
    (floors differ by 1-2 cm between areas, and the soles sit a shoe sole above the real
    floor)."""
    m = self.m
    k = 1.0 / scale
    mu = body_kg / float(m.body_subtreemass[0])
    mass = m.body_mass * mu
    for b, kg in (extra_kg or {}).items():
      mass[m.body(b).id] += kg
    # Bodies under 10 g are kinematic helpers (attachments, MyoArm's shoulder phantoms, the
    # arm model's massless thorax); their inertias are placeholders (the phantoms' 1 kg m^2
    # each, more than the whole trunk). Keep the mass, drop that inertia.
    Ib = np.where((m.body_mass < 0.01)[:, None], 0.0, m.body_inertia) * mu * k**2
    T = len(K["C"])
    if floor_z is None:
      floor_z = floor_from_soles(K["sole"] * k, [self.sole[s] for s in "lr"], rate)
    floor_z = np.broadcast_to(np.asarray(floor_z, float), (T,)).copy()
    C = lowpass(K["C"] * k, rate, lp_hz)
    a = np.gradient(np.gradient(C, axis=0), axis=0) * rate**2
    w = lowpass(omega(K["R"], rate), rate, lp_hz)
    R = K["R"]
    Hd = np.gradient(np.einsum("tbij,bj,tbkj,tbk->tbi", R, Ib, R, w), axis=0) * rate
    P = {j: v * k for j, v in K["P"].items()}
    gm = mass[None, :, None] * (a - G)

    def wrench(S, Pj, ext):
      F = gm[:, S].sum(1)
      M = (np.cross(C[:, S] - Pj[:, None], gm[:, S]) + Hd[:, S]).sum(1)
      for e in ext:
        if e["body"] in S:
          F = F - e["force"]
          M = M - np.cross(e["point"] - Pj, e["force"]) - e["torque"]
      return F, M

    out = dict(rate=rate, body_kg=body_kg, extra_kg=float(sum((extra_kg or {}).values())))
    # total ground reaction (force on the body from the floor), about a point on the floor
    O = np.zeros((T, 3))
    O[:, 2] = floor_z
    F, M = wrench(self.all, O, list(loads))
    Fz = np.where(np.abs(F[:, 2]) > 1.0, F[:, 2], np.nan)
    cop = np.stack([-M[:, 1] / Fz, M[:, 0] / Fz, floor_z], 1)
    free = M[:, 2] - (cop[:, 0] * F[:, 1] - cop[:, 1] * F[:, 0])
    out.update(grf=F, cop=cop, grf_free_moment=free, floor_z=floor_z)
    ext = list(loads)
    if legs:
      sole = K["sole"] * k
      share, foot_cop, contact = self.split_feet(sole, cop, rate, floor_z)
      foot_F = share[..., None] * F[:, None, :]
      foot_free = share * free[:, None]
      source = np.zeros((T, 2), np.int8)  # 0 estimated (lever rule), 1 measured, 2 remainder
      if plates is not None:
        foot_F, foot_cop, foot_free, source, contact = feet_from_plates(
          plates, foot_F, foot_cop, foot_free, contact, sole, self.sole, F, M, floor_z)
        out.update(plates_check(plates, contact, source, F))
      out.update(foot_share=share, foot_cop=foot_cop, foot_contact=contact, foot_force=foot_F,
                 foot_free=foot_free, foot_source=source)
      for i, s in enumerate("lr"):
        ext.append(dict(body=self.foot_b[s], point=np.nan_to_num(foot_cop[:, i]),
                        force=foot_F[:, i],
                        torque=foot_free[:, i, None] * np.array([0.0, 0.0, 1.0])))
    for j, (_, _, _, comps) in JOINTS.items():
      if not legs and j not in TOP_DOWN:
        continue
      Fj, Mj = wrench(self.sets[j], P[j], ext)
      Rj = K["Fr"][j]
      out[f"{j}_force"] = np.einsum("tji,tj->ti", Rj, Fj)  # in the frame body's axes
      for name, axis, sign in comps:
        if axis.startswith("jnt:"):
          u = K["Ax"][axis]
        else:
          u = Rj[:, :, "xyz".index(axis)]
        out[f"{j}_{name}"] = sign * np.einsum("ti,ti->t", Mj, u)
    e = out["L5S1_ext"]
    out["L5S1_compression"] = (out["L5S1_force"][:, 1] + np.maximum(e, 0) / D_EXT
                               + np.maximum(-e, 0) / D_FLEX)
    out["L5S1_shear_ap"] = out["L5S1_force"][:, 0]
    return out

  def split_feet(self, sole, cop, rate, floor_z, dz=0.025, v_max=0.6):
    """Share of the ground reaction per foot [T, 2] (l, r), each foot's centre of pressure
    [T, 2, 3], and the contact flags [T, 2]. A foot is loaded when one of its sole points
    is within dz of the floor and moving slower than v_max; in double support the share
    follows the lever rule along the line between the feet's contact centres, and both
    feet's pressure centres shift together so their weighted mean is the total one."""
    T = len(sole)
    z = sole[..., 2] - floor_z[:, None]
    v = np.linalg.norm(np.gradient(sole, axis=0), axis=-1) * rate
    contact = np.zeros((T, 2), bool)
    centre = np.zeros((T, 2, 3))
    for i, s in enumerate("lr"):
      idx = self.sole[s]
      near = (z[:, idx] < dz) & (v[:, idx] < v_max)
      contact[:, i] = near.any(1)
      wgt = np.where(near, 1.0, 0.0) + 1e-9
      centre[:, i] = (sole[:, idx] * wgt[..., None]).sum(1) / wgt.sum(1)[:, None]
    share = np.zeros((T, 2))
    only = contact.sum(1) == 1
    share[only] = contact[only]
    both = contact.all(1) | (contact.sum(1) == 0)
    u = centre[:, 1, :2] - centre[:, 0, :2]
    L2 = np.maximum((u**2).sum(1), 1e-6)
    wr = np.clip(((np.nan_to_num(cop[:, :2]) - centre[:, 0, :2]) * u).sum(1) / L2, 0, 1)
    share[both, 1], share[both, 0] = wr[both], 1 - wr[both]
    foot_cop = centre.copy()
    mean = (share[..., None] * centre).sum(1)
    shift = np.nan_to_num(cop) - mean
    shift[:, 2] = 0
    foot_cop += shift[:, None]
    foot_cop[:, :, 2] = floor_z[:, None]
    return share, foot_cop, contact


def feet_from_plates(plates, foot_F, foot_cop, foot_free, contact, sole, sole_idx, F, M, floor_z,
                     on_plate=0.25, min_fz=20.0):
  """Measured ground reactions where the feet are on force plates.

  plates: dict(force [T, P, 3] N on the body, cop [T, P, 3] m, free [T, P] N m) in the same world
  frame and units as the kinetics (subject size). A plate carrying more than min_fz belongs to
  the foot whose centre is within `on_plate` of its centre of pressure (none if both are: the
  two feet share the plate and stay estimated); that foot counts as on the ground. A foot gets the sum of its
  plates (source 1). If only one foot is measured and the other is also on the ground, the
  other carries the rest of the total ground reaction from the whole-body dynamics (source 2),
  at the centre of pressure that balances the moments. Elsewhere the estimate stays (0).
  """
  T = len(F)
  foot_F, foot_cop, foot_free, contact = foot_F.copy(), foot_cop.copy(), foot_free.copy(), contact.copy()
  source = np.zeros((T, 2), np.int8)
  centre = np.zeros((T, 2, 3))
  for i, s in enumerate("lr"):
    centre[:, i] = sole[:, sole_idx[s]].mean(1)
  pf, pc, pm = plates["force"], plates["cop"], plates["free"]
  for t in range(T):
    acc = [[np.zeros(3), np.zeros(3), 0.0, 0.0] for _ in range(2)]  # force, sum(Fz * cop), Fz, free
    for p in range(pf.shape[1]):
      fz = pf[t, p, 2]
      if not np.isfinite(fz) or fz < min_fz or not np.isfinite(pc[t, p]).all():
        continue
      d = np.linalg.norm(centre[t, :, :2] - pc[t, p, :2], axis=1)
      near = [i for i in range(2) if d[i] < on_plate]  # the plate, not the kinematics, says which foot is down
      if len(near) != 1:
        continue
      a = acc[near[0]]
      a[0] = a[0] + pf[t, p]
      a[1] = a[1] + fz * pc[t, p]
      a[2] += fz
      a[3] += pm[t, p]
    measured = [acc[i][2] > 0 for i in range(2)]
    contact[t] |= np.array(measured)
    for i in range(2):
      if measured[i]:
        foot_F[t, i] = acc[i][0]
        foot_cop[t, i] = acc[i][1] / acc[i][2]
        foot_free[t, i] = acc[i][3]
        source[t, i] = 1
    if sum(measured) == 1:
      m, o = (0, 1) if measured[0] else (1, 0)
      if contact[t, o]:
        Fo = F[t] - foot_F[t, m]
        rm = foot_cop[t, m] - np.array([0.0, 0.0, floor_z[t]])
        Mo = M[t] - np.cross(rm, foot_F[t, m]) - np.array([0.0, 0.0, foot_free[t, m]])
        if Fo[2] > min_fz:
          cop = np.array([-Mo[1] / Fo[2], Mo[0] / Fo[2], floor_z[t]])
          if np.linalg.norm(cop[:2] - centre[t, o, :2]) > on_plate:
            cop = np.array([centre[t, o, 0], centre[t, o, 1], floor_z[t]])
          foot_F[t, o] = Fo
          foot_cop[t, o] = cop
          foot_free[t, o] = Mo[2] - (cop[0] * Fo[1] - cop[1] * Fo[0])
        else:
          foot_F[t, o] = 0.0
          foot_free[t, o] = 0.0
        source[t, o] = 2
      else:
        foot_F[t, o] = 0.0
        foot_free[t, o] = 0.0
  return foot_F, foot_cop, foot_free, source, contact


def plates_check(plates, contact, source, F):
  """Where every loaded foot is measured: the plates' total vertical force against the
  whole-body prediction (validation of the dynamics, independent of any split)."""
  all_measured = (source == 1).sum(1) == contact.sum(1)
  all_measured &= contact.any(1)
  fz = np.nansum(np.where(plates["force"][..., 2] > 20.0, plates["force"][..., 2], 0.0), 1)
  return dict(plates_all_measured=all_measured, plates_fz=fz, plates_pred_fz=F[:, 2])


def floor_from_soles(sole, feet, rate, window_s=3.0, v_still=0.15, q=20):
  """Floor height [T] as the model's soles see it: the height of the lowest sole point of a
  foot that stands still (slower than v_still), as a running low percentile over window_s.
  The lowest point overall is not used: the model's feet dip a few cm below the standing
  level at heel strike and toe-off, which put the floor too low and the standing foot
  "above" it. Frames with no still foot in the window take the nearest estimate."""
  T = len(sole)
  low = np.stack([sole[:, idx, 2].min(1) for idx in feet], 1)  # [T, 2]
  arg = np.stack([sole[:, idx, 2].argmin(1) for idx in feet], 1)
  v = np.linalg.norm(np.gradient(sole, axis=0), axis=-1) * rate
  vlow = np.stack([v[np.arange(T), np.asarray(idx)[arg[:, i]]] for i, idx in enumerate(feet)], 1)
  still = vlow < v_still
  half = int(window_s * rate / 2)
  out = np.full(T, np.nan)
  for t in range(T):
    a, b = max(0, t - half), min(T, t + half + 1)
    z = low[a:b][still[a:b]]
    if len(z):
      out[t] = np.percentile(z, q)
  if np.isnan(out).all():
    return np.full(T, np.percentile(low, 5))
  idx = np.where(~np.isnan(out))[0]
  return np.interp(np.arange(T), idx, out[idx])
