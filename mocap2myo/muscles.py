"""Muscle forces from the net joint moments: static optimisation.

For every frame, the activations a (0..1) of the model's muscles that minimise
  sum_i w_i a_i^2 + sum_c (r_c / R)^2
subject to, for every constrained moment component c of the inverse dynamics
(mocap2myo/kinetics.py: a joint's net moment about one axis),
  sum_i A_ci a_i + r_c = tau_c - P_c.
A_ci is the moment of muscle i at full activation about that axis, P_c the moment of the
muscles' passive forces, tau_c the net moment, and r_c a reserve: an ideal torque about the
same axis, penalised with weight 1 / R^2 (R = `reserve`, N m). With R small (1 N m) a
reserve stays near zero while the muscles can supply the moment (below 0.1 N m) and takes
the remainder where they cannot (saturated, or no muscle crosses the joint): infeasible
frames are visible as reserves, not hidden. w_i = 1 (sum of squared activations), or the
muscle's fibre volume (F0 x optimal fibre length, normalised to mean 1) with weight "volume":
a cost that does not change when a muscle is split into several slips.

Muscle moments are taken the way the inverse dynamics takes the net moment: about the
joint's point, on the subtree distal to it. A muscle's tension pulls on every body its
path touches; a straight segment of the path between a body of the subtree and one outside
it pulls the inner end toward the outer one (a wrapping surface takes the pull of the two
segments at its tangent points). Summed over the segments that cross, and projected on the
component's axis, that is the muscle's moment arm for this component: it covers muscles
crossing several joints, any joint layout, and joint axes that are not the model's own
(MyoFullBody's shoulder moments are about the thorax axes). Muscles that cross no
constrained joint are not in the problem (their activation is NaN).

Muscle model: MuJoCo's (mju_muscleGain / mju_muscleBias, reproduced here to machine
precision): force = F0 (FL(L) FV(V) a + FP(L)), with a rigid tendon. The normalised fibre
length L maps the muscle-tendon length linearly onto the muscle's operating range
(gainprm range over the actuator's lengthrange), the normalised velocity V is the
muscle-tendon lengthening rate over L0 * vmax (L0 = lengthrange span / range span, the
optimal fibre length). No pennation, no tendon compliance. Lengths are the model's at the
frame's pose (the model is uniformly `scale` x the subject: normalised lengths and
velocities do not change with size; moment arms are divided by scale). F0 is the model's.

Solver: the dual of the problem is an unconstrained, strictly convex, piecewise quadratic
function of the m reserves (y = r / R^2, one per component), and a_i = clip(A_i . y / w_i,
0, 1) at its minimum. A semismooth Newton method with a backtracking line search finds it
exactly in a few iterations (m x m systems; warm start from the previous frame).
"""

import mujoco
import numpy as np

from mocap2myo.kinetics import lowpass

TOL = 1e-9  # N m: moment balance of the solver (|R^2 y - (b - A a)|)
MAX_ITER = 200


def gain_length(L, lmin, lmax):
  """Active force-length curve (MuJoCo's mju_muscleGainLength), vectorised."""
  a = 0.5 * (lmin + 1)
  b = 0.5 * (1 + lmax)
  x1 = (L - lmin) / np.maximum(mujoco.mjMINVAL, a - lmin)
  x2 = (1 - L) / np.maximum(mujoco.mjMINVAL, 1 - a)
  x3 = (L - 1) / np.maximum(mujoco.mjMINVAL, b - 1)
  x4 = (lmax - L) / np.maximum(mujoco.mjMINVAL, lmax - b)
  f = np.where(L <= a, 0.5 * x1**2, np.where(L <= 1, 1 - 0.5 * x2**2,
                                            np.where(L <= b, 1 - 0.5 * x3**2, 0.5 * x4**2)))
  return np.where((L < lmin) | (L > lmax), 0.0, f)


def gain_velocity(V, fvmax):
  """Force-velocity curve (MuJoCo's, in mju_muscleGain): V < 0 shortening."""
  y = fvmax - 1
  return np.where(V <= -1, 0.0, np.where(V <= 0, (V + 1)**2, np.where(
    V <= y, fvmax - (y - V)**2 / np.maximum(mujoco.mjMINVAL, y), fvmax)))


def passive_length(L, lmax, fpmax):
  """Passive force-length curve (MuJoCo's mju_muscleBias, as a fraction of F0)."""
  b = 0.5 * (1 + np.maximum(1, lmax))
  x1 = (L - 1) / np.maximum(mujoco.mjMINVAL, b - 1)
  x2 = (L - b) / np.maximum(mujoco.mjMINVAL, b - 1)
  return np.where(L <= 1, 0.0, np.where(L <= b, fpmax * 0.5 * x1**2, fpmax * (0.5 + x2)))


def is_muscle(m, i):
  return (m.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_MUSCLE
          and m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_MUSCLE)


class MuscleParams:
  """The muscle parameters of actuators `ids` (MuJoCo muscles), as arrays."""

  def __init__(self, m, ids, strength=1.0):
    ids = np.asarray(ids, int)
    lr = m.actuator_lengthrange[ids]
    acc0 = m.actuator_acc0[ids]

    def unpack(prm):
      F0 = strength * np.where(prm[:, 2] > 0, prm[:, 2], prm[:, 3] / np.maximum(mujoco.mjMINVAL, acc0))
      L0 = (lr[:, 1] - lr[:, 0]) / np.maximum(mujoco.mjMINVAL, prm[:, 1] - prm[:, 0])
      return F0, L0
    g, b = m.actuator_gainprm[ids, :9], m.actuator_biasprm[ids, :9]
    self.lr, self.g, self.b = lr, g, b
    self.F0, self.L0 = unpack(g)
    self.F0p, self.L0p = unpack(b)
    self.gear = m.actuator_gear[ids, 0]

  def norm_length(self, length, prm, L0):
    return prm[:, 0] + (length - self.lr[:, 0]) / np.maximum(mujoco.mjMINVAL, L0)

  def capacity(self, length, velocity):
    """Active force at full activation and passive force (N, model F0) [..., n] at
    actuator lengths / velocities [..., n]; and the normalised fibre length."""
    g, b = self.g, self.b
    L = self.norm_length(length, g, self.L0)
    V = velocity / np.maximum(mujoco.mjMINVAL, self.L0 * g[:, 6])
    active = self.F0 * gain_length(L, g[:, 4], g[:, 5]) * gain_velocity(V, g[:, 8])
    Lp = self.norm_length(length, b, self.L0p)
    passive = self.F0p * passive_length(Lp, b[:, 5], b[:, 7])
    return active, passive, L


def solve_frame(A, b, w, R2, y0=None):
  """min sum w a^2 + sum (r^2 / R2), A a + r = b, 0 <= a <= 1 (A [m, n], b [m], w [n],
  R2 [m] = R^2). Returns (a, r, y, iterations): r the reserves, y = r / R2 (the dual)."""
  m = len(b)
  y = np.zeros(m) if y0 is None else y0.copy()

  def phi(y):
    """The dual function, its rounding error, and A' y."""
    s = A.T @ y
    h = np.where(s <= 0, 0.0, np.where(s < w, 0.5 * s * s / w, s - 0.5 * w))
    q, l, hs = 0.5 * np.dot(R2 * y, y), np.dot(b, y), h.sum()
    return q - l + hs, 64 * np.finfo(float).eps * (q + abs(l) + hs), s

  grad = lambda y, s: R2 * y - b + A @ np.clip(s / w, 0.0, 1.0)  # gradient of the dual
  tol = TOL * max(1.0, np.max(np.abs(b)))
  f, _, s = phi(y)
  F = grad(y, s)
  for it in range(MAX_ITER):
    if np.max(np.abs(F)) < tol:
      return np.clip(s / w, 0.0, 1.0), R2 * y, y, it
    free = (s > 0) & (s < w)
    Af = A[:, free]
    J = np.diag(R2) + (Af / w[free]) @ Af.T
    dy = -np.linalg.solve(J, F)
    # Armijo backtracking on phi. The full step solves the problem on the current set of
    # free / bound muscles exactly; near the solution phi's decrease falls below its
    # rounding error, so there a full step that keeps phi within that error and reduces
    # the residual is taken.
    slope, t = np.dot(F, dy), 1.0
    while True:
      fn, err, sn = phi(y + t * dy)
      Fn = grad(y + t * dy, sn)
      if fn <= f + 1e-4 * t * slope or (t == 1.0 and fn <= f + err
                                       and np.max(np.abs(Fn)) < np.max(np.abs(F))):
        break
      t *= 0.5
      if t < 1e-12:
        raise RuntimeError(f"static optimisation: no descent (moment residual "
                           f"{np.max(np.abs(F)):.3g} N m)")
    y, f, s, F = y + t * dy, fn, sn, Fn
  raise RuntimeError(f"static optimisation did not converge in {MAX_ITER} iterations "
                     f"(moment residual {np.max(np.abs(F)):.3g} N m)")


class Muscles:
  """Static optimisation on a kinetics.Kinetics model (its joints, subtrees and axes).

  comps: the moment components to match, [(joint, component name)] of kin.joints, e.g.
  [("L5S1", "ext"), ("elbow_r", "flex")]; reserve: R (N m); weight: "none" or "volume";
  strength: factor on every maximum isometric force (active and passive)."""

  def __init__(self, kin, comps, reserve=1.0, weight="none", strength=1.0):
    m = kin.m
    self.kin, self.m, self.d = kin, m, mujoco.MjData(m)
    self.comps = list(comps)
    for j, c in self.comps:
      if j not in kin.joints or c not in [x[0] for x in kin.joints[j][3]]:
        raise ValueError(f"no kinetics component {j}_{c} in {sorted(kin.joints)}")
    self.joints = sorted({j for j, _ in self.comps}, key=list(kin.joints).index)
    W = mujoco.mjtWrap
    musc = [i for i in range(m.nu) if is_muscle(m, i)]
    trn = [i for i in musc if m.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_TENDON]
    if trn:
      raise ValueError(f"muscles without a tendon transmission are not supported: "
                       f"{[m.actuator(i).name for i in trn]}")
    self.all_ids = np.array(musc, int)
    self.names = [m.actuator(i).name for i in musc]
    self.other = [m.actuator(i).name for i in range(m.nu) if i not in musc]
    # each muscle's path: its sites in order (the data's site points follow them), and the
    # bodies the path can touch (sites and wrapping geoms)
    tid = m.actuator_trnid[self.all_ids, 0]
    sites, bodies = [], []
    for t in tid:
      w = range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t])
      if any(m.wrap_type[k] == W.mjWRAP_PULLEY for k in w):
        raise ValueError(f"tendon {m.tendon(t).name}: pulleys are not supported")
      if any(m.wrap_type[k] == W.mjWRAP_JOINT for k in w):
        raise ValueError(f"tendon {m.tendon(t).name}: fixed tendons are not supported")
      s = [m.wrap_objid[k] for k in w if m.wrap_type[k] == W.mjWRAP_SITE]
      sites.append(s)
      bodies.append({int(m.site_bodyid[x]) for x in s}
                    | {int(m.geom_bodyid[m.wrap_objid[k]]) for k in w
                       if m.wrap_type[k] in (W.mjWRAP_SPHERE, W.mjWRAP_CYLINDER)})
    self.inS = {}
    for j in kin.joints:
      mask = np.zeros(m.nbody, bool)
      mask[kin.sets[j]] = True
      self.inS[j] = mask
    # crosses[j][i]: muscle i's path touches bodies on both sides of joint j's cut
    self.crosses = {j: np.array([any(self.inS[j][b] for b in bs) and not all(self.inS[j][b] for b in bs)
                                 for bs in bodies]) for j in kin.joints}
    use = np.zeros(len(musc), bool)
    for j in self.joints:
      use |= self.crosses[j]
    self.use = use
    self.ids = self.all_ids[use]
    self.tid = tid[use]
    self.n = int(use.sum())
    sel = [sites[k] for k in np.where(use)[0]]
    self.site_off = np.r_[0, np.cumsum([len(s) for s in sel])[:-1]].astype(int)
    self.site_flat = np.array([x for s in sel for x in s], int)
    self.prm = MuscleParams(m, self.ids, strength)
    self.reserve = float(reserve)
    if weight == "none":
      self.w = np.ones(self.n)
    elif weight == "volume":
      v = self.prm.F0 * self.prm.L0
      self.w = v / v.mean()
    else:
      raise ValueError(f"weight {weight!r}: expected none or volume")
    self.weight = weight

  # --- geometry ---------------------------------------------------------------------
  def _paths(self):
    """This frame's straight path segments of the problem's muscles: start / end points,
    their bodies, the muscle (0..n-1) of each."""
    m, d = self.m, self.d
    adr, num = d.ten_wrapadr[self.tid], d.ten_wrapnum[self.tid]
    owner = np.repeat(np.arange(self.n), num)
    start = np.repeat(np.cumsum(num) - num, num)
    idx = np.repeat(adr, num) + np.arange(num.sum()) - start
    obj = d.wrap_obj.ravel()[idx]
    X = d.wrap_xpos.reshape(-1, 3)[idx]
    site = obj == -1
    cs = np.cumsum(site)
    first = np.repeat(cs[np.cumsum(num) - num] - site[np.cumsum(num) - num], num)
    rank = cs - first - 1
    body = np.empty(len(obj), int)
    body[site] = m.site_bodyid[self.site_flat[self.site_off[owner[site]] + rank[site]]]
    body[~site] = m.geom_bodyid[obj[~site]]
    # consecutive points of one path, except the arc over a wrapping geom (its two tangent
    # points carry the same geom)
    seg = (owner[:-1] == owner[1:]) & ~((obj[:-1] >= 0) & (obj[1:] == obj[:-1]))
    return X[:-1][seg], X[1:][seg], body[:-1][seg], body[1:][seg], owner[:-1][seg]

  def _axes(self, K, t):
    """Unit axes (world) of the constrained components at frame t, with their signs."""
    out = []
    for j, c in self.comps:
      _, _, fr, cs = self.kin.joints[j]
      axis, sign = next((a, s) for n, a, s in cs if n == c)
      out.append((self.axis(K, j, axis, t), sign))
    return out

  def axis(self, K, j, axis, t):
    """As kinetics.Kinetics.solve: "jnt:<joint>" (its axis), "anat:<direction>" (fixed in
    the frame body at neutral), or x / y / z of the frame body."""
    Rj = K["Fr"][j][t]
    if axis.startswith("jnt:"):
      return K["Ax"][axis][t]
    if axis.startswith("anat:"):
      return Rj @ self.kin.anat[(self.kin.joints[j][2], axis[5:])]
    return Rj[:, "xyz".index(axis)]

  def geometry(self, Q, K, scale, force_joints=()):
    """Per frame: muscle-tendon lengths [T, n] (model units) and moment arms [T, m, n]
    (subject metres, the components' signs applied: tension x arm = the muscle's share of
    the reported moment). force_joints: joints whose per-unit-tension force on the distal
    subtree is also returned, {joint: [T, n, 3] in the frame body's axes}.
    K: kin.kinematics(Q)."""
    m, d = self.m, self.d
    T = len(Q)
    k = 1.0 / scale
    length = np.zeros((T, self.n))
    arms = np.zeros((T, len(self.comps), self.n), np.float32)
    fdir = {j: np.zeros((T, self.n, 3), np.float32) for j in force_joints}
    comp_j = [j for j, _ in self.comps]
    for t in range(T):
      d.qpos[:] = Q[t]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      mujoco.mj_tendon(m, d)
      length[t] = d.ten_length[self.tid] * self.prm.gear
      xa, xb, ba, bb, own = self._paths()
      u = xb - xa
      u /= np.linalg.norm(u, axis=1, keepdims=True)
      axes = self._axes(K, t)
      for j in self.joints + [j for j in force_joints if j not in self.joints]:
        ia, ib = self.inS[j][ba], self.inS[j][bb]
        cross = ia != ib
        xin = np.where(ia[:, None], xa, xb)[cross]
        dirn = np.where(ia[:, None], u, -u)[cross]  # from the inner end toward the outer
        P = K["P"][j][t]
        M = np.zeros((self.n, 3))
        np.add.at(M, own[cross], np.cross(xin - P, dirn))
        for c, (jc, (ax, sign)) in enumerate(zip(comp_j, axes)):
          if jc == j:
            arms[t, c] = sign * (M @ ax) * k
        if j in fdir:
          Fd = np.zeros((self.n, 3))
          np.add.at(Fd, own[cross], dirn)
          fdir[j][t] = Fd @ K["Fr"][j][t]  # in the frame body's axes
    return length, arms, fdir

  # --- optimisation -------------------------------------------------------------------
  def solve(self, length, arms, target, rate, lp_hz=6.0, fv=True, passive=True):
    """Activations [T, n], tensions [T, n] (N), reserves [T, m] (N m), the passive
    muscles' moments [T, m], solver iterations [T]. length [T, n] model units, arms
    [T, m, n] (geometry), target [T, m] the net moments (N m), rate (Hz); fv / passive: the
    force-velocity and passive curves on (off: FV = 1 / no passive force)."""
    T = len(length)
    vel = np.gradient(lowpass(length, rate, lp_hz), axis=0) * rate
    if not fv:
      vel = np.zeros_like(vel)
    act_cap, pas, Lf = self.prm.capacity(length, vel)
    if not passive:
      pas = np.zeros_like(pas)
    R2 = np.full(len(self.comps), self.reserve**2)
    a = np.zeros((T, self.n))
    r = np.zeros((T, len(self.comps)))
    pm = np.einsum("tcn,tn->tc", arms, pas)
    iters = np.zeros(T, np.int16)
    y = None
    for t in range(T):
      A = arms[t] * act_cap[t]
      b = target[t] - pm[t]
      a[t], r[t], y, iters[t] = solve_frame(A.astype(np.float64), b, self.w, R2, y)
    tension = act_cap * a + pas
    return dict(act=a, tension=tension, reserve=r, passive_moment=pm, iters=iters,
                fibre_length=Lf, capacity=act_cap)
