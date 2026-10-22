"""Marker-based IK onto a musculoskeletal model: marker model, per-frame solve, subject calibration.

The model is a profile (mocap2myo.profile; default M2M_MODEL, MyoFullBody) and the markers a
marker set (mocap2myo.markerset; default M2M_MARKER_SET, Plug-in-Gait): MarkerModel(profile,
markerset).

Input: marker npz files (pipeline/prepare_markers.py): markers [T, n, 3] in the
order of the marker set's labels, marker_rate, height_m; metres, Z-up.

Pipeline per subject:
  1. scale the marker cloud by the model's height / the subject's height (the model is
     not geometrically scaled; per-marker body offsets absorb what remains);
  2. calibrate marker offsets on their bodies: initial offsets from the marker set's
     anatomical placement on the model's neutral pose, then alternate (IK on K diverse
     frames) <-> (closed-form offset update with a prior toward the previous
     offsets), as in OpenSim/AddBiomechanics marker registration;
  3. per-frame IK, warm-started from the previous frame.
Unknowns: root (3 + 3 rotvec) and the profile's IK joints (MyoFullBody: 14 leg, 3 lumbar,
2 x 7 arm DoF, 37 in all). Every other joint follows through the model's joint equalities
or stays at the reference pose.
"""

from __future__ import annotations

import fnmatch

import numpy as np
import mujoco
import scipy.optimize
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as sRot

from mocap2myo import markerset as _ms
from mocap2myo import objects as O
from mocap2myo import profile as _prof
from mocap2myo.kinematics import apply_joint_equalities

MODEL_HEIGHT = 1.72  # MyoFullBody: bone height 1.689 m + soft tissue (its profile's model_height)
FREE = [f"{j}_{s}" for s in ("r", "l") for j in
        ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle",
         "subtalar_angle", "mtp_angle")] + \
       ["flex_extension", "lat_bending", "axial_rotation"] + \
       [f"{j}_{s}" for s in ("r", "l") for j in
        ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flex", "pro_sup",
         "deviation", "flexion")]  # MyoFullBody's IK joints (its profile lists them)
# temporal regularisation (ik_trial smooth, ankle_smooth, the gated joints) is given for
# this rate and scaled to the data's (the same smoothing time constant at any rate)
SMOOTH_RATE = 100.0


# Optional neck (MarkerModel(neck=True)): MyoFullBody has no neck joint, so the head is
# rigid on the trunk and its markers miss by ~3 cm (9 cm in the worst frames) whenever
# the subject nods or turns. Three hinges at the neck body, pivot at about C7/T1,
# appended to the IK's free joints; the IK output keeps the model's own qpos layout
# and stores these three angles separately (split_qpos / full_qpos). Where they go is the
# profile's [ik] neck (MyoFullBody, MyoLeg: these values).
NECK = ("neck_extension", "neck_bending", "neck_rotation")
_NECK_PIVOT = (0.0, 0.39, 0.0)  # in the neck body's frame (y up): ~1.49 m at qpos0
_NECK_AXES = ((0, 0, 1), (1, 0, 0), (0, 1, 0))  # extension (+), bending to the right (+), rotation
_NECK_RANGE = ((-1.2, 1.0), (-0.8, 0.8), (-1.4, 1.4))


def load_trial(path, max_gap_s=0.2, lowpass_hz=None, mm=None, gap_fill="spline"):
  """Markers [T, n, 3] (the marker set's order) scaled to the model; short gaps interpolated.
  mm: the MarkerModel (its marker set and model height; default the selected marker set
  and MyoFullBody's MODEL_HEIGHT). A model scaled to the subject (mm.scaling,
  mocap2myo.scaling) takes the markers as recorded: scale 1.
  lowpass_hz (optional, default off; the stages pass M2M_LOWPASS_HZ, 6 Hz): zero-phase
  2nd-order Butterworth (filtfilt) per marker coordinate over each gap-free run, the
  usual marker filtering before IK (Winter: ~6 Hz for whole-body motion).
  Gaps of at most max_gap_s seen on both sides are filled per coordinate: gap_fill "linear"
  (between the two neighbouring samples) or "spline" (a cubic spline through the samples
  within max_gap_s on either side, at least two on each; linear otherwise)."""
  if gap_fill not in ("linear", "spline"):
    raise ValueError(f"gap_fill={gap_fill!r}: expected linear or spline")
  LABELS = mm.ms.LABELS if mm is not None else _ms.LABELS
  height = mm.model_height if mm is not None else MODEL_HEIGHT
  d = np.load(path)
  labs = list(d["marker_labels"])
  P = np.full((d["markers"].shape[0], len(LABELS), 3), np.nan, np.float64)
  for k, l in enumerate(LABELS):
    if l in labs:
      P[:, k] = d["markers"][:, labs.index(l)]
  rate = float(d["marker_rate"])
  t = np.arange(len(P))
  for k in range(len(LABELS)):
    bad = np.isnan(P[:, k, 0])
    if bad.all() or not bad.any():
      continue
    good = ~bad
    filled = np.stack([np.interp(t, t[good], P[good, k, c]) for c in range(3)], 1)
    # only fill gaps shorter than max_gap_s
    runs = np.split(np.where(bad)[0], np.where(np.diff(np.where(bad)[0]) != 1)[0] + 1)
    w = int(round(max_gap_s * rate))
    for r in runs:
      if len(r) and len(r) <= max_gap_s * rate and r[0] > 0 and r[-1] < len(P) - 1:
        P[r, k] = filled[r]
        if gap_fill == "spline":
          lo = t[max(0, r[0] - w):r[0]][good[max(0, r[0] - w):r[0]]]
          hi = t[r[-1] + 1:r[-1] + 1 + w][good[r[-1] + 1:r[-1] + 1 + w]]
          if len(lo) >= 2 and len(hi) >= 2:
            kn = np.r_[lo, hi]
            P[r, k] = CubicSpline(kn, P[kn, k], axis=0)(r)
  if lowpass_hz:
    if not 0 < lowpass_hz < rate / 2:
      raise ValueError(f"{path}: low-pass {lowpass_hz} Hz must lie between 0 and half the "
                       f"marker rate ({rate:g} Hz)")
    bb, aa = butter(2, lowpass_hz / (rate / 2))
    pad = 3 * max(len(aa), len(bb))
    for k in range(len(LABELS)):
      idx = np.flatnonzero(~np.isnan(P[:, k, 0]))
      if len(idx):
        for r in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
          if len(r) > pad:
            P[r, k] = filtfilt(bb, aa, P[r, k], axis=0)
  if getattr(mm, "scaling", None) is not None:
    return P, rate, 1.0
  s = height / float(d["height_m"])
  return P * s, rate, s


class MarkerModel:
  """A model profile with a marker set's markers as sites, the IK's free joints, sole points
  and palms. profile / markerset: names or objects (default M2M_MODEL / M2M_MARKER_SET).
  Markers on a segment the model lacks (e.g. arm markers on MyoLeg) get no weight
  (self.active False): their columns stay in the arrays, their error is NaN.
  scaling (optional): a subject's mocap2myo.scaling.Scaling; the model is then scaled to the
  subject segment by segment (self.scaling.A: each body's matrix) and takes the markers in
  the subject's own size (load_trial)."""

  def __init__(self, profile=None, markerset=None, neck=False, scaling=None):
    self.profile = _prof.load(profile) if profile is not None else _prof.selected()
    self.ms = _ms.load(markerset) if markerset is not None else _ms.selected()
    p, ms = self.profile, self.ms
    spec = p.spec()
    self.scaling = scaling
    if scaling is not None:
      scaling.apply(spec, p)
    root_body = p.root_body or spec.joint(p.root_joint).parent.name
    bodies = {l: p.segment_body(ms.segment(l)) for l in ms.LABELS}
    self.active = np.array([bodies[l] is not None for l in ms.LABELS])
    for l in ms.LABELS:
      spec.body(bodies[l] or root_body).add_site(name=f"mk_{l}", pos=[0, 0, 0], size=[0.01, 0, 0])
    self.neck = neck
    nk = p.ik_cfg.get("neck") if neck else None
    if neck and nk is None:
      raise SystemExit(f"{p.name}: no optional neck for this model ([ik] neck in its profile)")
    if neck:
      pivot = list(nk["pivot"])
      if scaling is not None:  # the pivot scales with its body
        pivot = (scaling.A[spec.body(nk["body"]).id] @ np.asarray(pivot, float)).tolist()
      for n, ax, rg in zip(NECK, nk["axes"], nk["ranges"]):
        spec.body(nk["body"]).add_joint(name=n, type=mujoco.mjtJoint.mjJNT_HINGE, pos=pivot,
                                        axis=list(ax), range=list(rg), limited=1)
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.free = p.ik_joints(m) + (list(NECK) if neck else [])
    self.qref = p.reference_qpos(m)  # MyoFullBody: qpos0
    self.root_bid = int(m.jnt_bodyid[m.joint(p.root_joint).id])
    LABELS = ms.LABELS
    self.sid = np.array([m.site(f"mk_{l}").id for l in LABELS])
    # Sites created at the body origin get site_sameframe=1, and kinematics
    # then copies the body pose and ignores site_pos -- every offset written
    # later would silently do nothing. Clear the flag.
    m.site_sameframe[self.sid] = 0
    self.bid = np.array([m.body(bodies[l] or root_body).id for l in LABELS])
    # a marker on a body that no IK joint moves relative to the pelvis (MyoLeg's trunk and
    # head, MS-Human-700 Locomotion's arms) would only pull the pelvis: left out
    jbody = {int(m.jnt_bodyid[m.joint(n).id]) for n in p.ik_joints(m)}
    pel = p.segment_body("pelvis")
    for k, l in enumerate(LABELS):
      if not self.active[k] or bodies[l] == pel:
        continue
      b, moved = self.bid[k], False
      while b > 0 and not moved:
        moved = b in jbody
        b = m.body_parentid[b]
      self.active[k] = moved
    # a joint with no active marker beyond it (the neck of MASS and MS-Human-700 with a set
    # that has no head markers) has nothing to hold it: left at the reference pose, not
    # solved. The toes (profile [ik] mtp) keep their prior instead.
    kids = [[] for _ in range(m.nbody)]
    for b in range(1, m.nbody):
      kids[m.body_parentid[b]].append(b)
    marked = {int(b) for b, a in zip(self.bid, self.active) if a}

    def reaches(b):
      return b in marked or any(reaches(c) for c in kids[b])
    keep = set(p.ik_cfg.get("mtp", [])) | {p.root_joint} | (set(NECK) if neck else set())
    self.unsolved_joints = [n for n in self.free if n not in keep
                            and not reaches(int(m.jnt_bodyid[m.joint(n).id]))]
    self.free = [n for n in self.free if n not in self.unsolved_joints]
    self.w = np.array([ms.MARKERS[l][1] if a else 0.0 for l, a in zip(LABELS, self.active)])
    self.adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.free])
    self.mtp = np.array([self.free.index(n) for n in p.ik_cfg.get("mtp", [])], int)
    self.mtp_q0 = self.qref[self.adr[self.mtp]]  # the toes' neutral (MyoFullBody: 0)
    self.static_neutral = list(p.ik_cfg.get("static_neutral", []))
    self.neck_adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NECK]) if neck else np.zeros(0, int)
    jid = [m.joint(n).id for n in self.free]
    self.lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jid])
    self.hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jid])
    heavy = p.ik_cfg.get("smooth_heavy", [])
    self.smooth_heavy = np.array([any(fnmatch.fnmatchcase(n, g) for g in heavy) for n in self.free])
    self.smooth_gated = [(self.free.index(j), self.free.index(g))
                         for j, g in p.ik_cfg.get("smooth_gated", {}).items()]
    self._model_height = p.data.get("height", {}).get("model_height")
    self._foot_points()
    self._palms()

  @property
  def model_height(self) -> float:
    """Standing height the subjects are scaled to (MyoFullBody: MODEL_HEIGHT)."""
    if self._model_height is None:
      self._model_height = self.profile.model_height
    return float(self._model_height)

  @property
  def labels(self):
    return self.ms.LABELS

  def _palms(self):
    """Palm points: MyoFullBody's hand sphere (mocap2myo.objects); another model's from its
    landmarks: centre 65 % of the way wrist -> 3rd metacarpal head, palm normal from
    the radial / ulnar styloids and that axis (as objects.py does with the metacarpals)."""
    m, p = self.m, self.profile
    pb = p.contact.get("palm_body")
    self.hand_b, self.palm_c, self.palm_n, self.palm_d = [], [], [], []
    if p.name.startswith("myofullbody"):
      self.hand_b = [m.body("thirdmc_l").id, m.body("thirdmc_r").id]
      pl, sp = O.hand_palmar_local(), O.hand_sphere_local()
      self.palm_n = [pl["l"], pl["r"]]
      self.palm_c = [sp["l"], sp["r"]]
      if self.scaling is not None:  # the sphere moves with the hand; the normal stays normal
        A = [self.scaling.A[b] for b in self.hand_b]
        self.palm_c = [a @ c for a, c in zip(A, self.palm_c)]
        self.palm_n = [np.linalg.solve(a.T, n) / np.linalg.norm(np.linalg.solve(a.T, n))
                       for a, n in zip(A, self.palm_n)]
      self.d.qpos[:] = self.qref
      mujoco.mj_kinematics(m, self.d)  # qpos0
      for sd, b in zip("lr", self.hand_b):  # distal (finger) axis in the thirdmc frame
        v = self.d.xpos[m.body(f"3proxph_{sd}").id] - self.d.xpos[b]
        self.palm_d.append(self.d.xmat[b].reshape(3, 3).T @ (v / np.linalg.norm(v)))
      return
    if not pb:
      return
    from mocap2myo import landmarks as L
    nt = L.Neutral(p, "floor", m)
    lm = L.compute(p, neutral=nt).landmarks
    for sd in "lr":
      need = [f"{k}_{sd}" for k in ("wrist", "mc3_head", "styloid_rad", "styloid_uln")]
      if any(k not in lm for k in need):
        self.hand_b, self.palm_c, self.palm_n, self.palm_d = [], [], [], []
        return
      wr, mc, ra, ul = (lm[k].pos for k in need)
      b = m.body(pb[sd]).id
      R, x = nt.d.xmat[b].reshape(3, 3), nt.d.xpos[b]
      distal = (mc - wr) / np.linalg.norm(mc - wr)
      across = ra - ul
      n = np.cross(across, distal) if sd == "r" else np.cross(distal, across)
      self.hand_b.append(b)
      self.palm_c.append(R.T @ (wr + 0.65 * (mc - wr) - x))
      self.palm_n.append(R.T @ (n / np.linalg.norm(n)))
      self.palm_d.append(R.T @ distal)

  def split_qpos(self, Q):
    """IK qpos of this model -> (qpos in the plain model's layout, neck angles [.., 3])."""
    Q = np.asarray(Q)
    return np.delete(Q, self.neck_adr, axis=-1), Q[..., self.neck_adr]

  def full_qpos(self, q, neck):
    """Inverse of split_qpos."""
    q = np.asarray(q)
    out = q
    for k, a in enumerate(self.neck_adr):  # increasing addresses: insert in order
      out = np.insert(out, a, np.asarray(neck)[..., k], axis=-1)
    return out

  def palms(self):
    """Hand sphere centres, palm normals, finger axes [2, 3] each (l, r); NaN for a
    model without hands."""
    c, n, f = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))
    if not self.hand_b:
      return c * np.nan, n * np.nan, f * np.nan
    for k, b in enumerate(self.hand_b):
      R = self.d.xmat[b].reshape(3, 3)
      c[k] = self.d.xpos[b] + R @ self.palm_c[k]
      n[k] = R @ self.palm_n[k]
      f[k] = R @ self.palm_d[k]
    return c, n, f

  def _foot_points(self):
    """Sole points, stored as (body, local offset) with (side, heel / toe / -). With the
    profile's sole_geoms = "collision" and capsules (MyoFullBody): both ends of every
    capsule of the sole bodies, radius below the axis, at the reference pose (feet flat);
    heel = rear ends on the first sole body (calcn), toe = front ends on the last (toes;
    the model faces -y). Otherwise mocap2myo.landmarks.sole_points (spheres, ellipsoids,
    boxes; or points on the foot bones)."""
    m, d = self.m, self.d
    p = self.profile
    d.qpos[:] = self.qref
    mujoco.mj_kinematics(m, d)
    self.fp_body, self.fp_local, self.fp_kind = [], [], []  # kind: (side, heel/toe/-)
    c = p.contact
    sb = c.get("sole_bodies", {})
    if c.get("sole_geoms", "collision") == "collision" and c.get("sole_geom_types") == ["capsule"]:
      where = {}
      for sd, bl in sb.items():
        for k, bn in enumerate(bl):
          where[bn] = (sd, k == 0, k == len(bl) - 1)
      for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        bn = m.body(b).name
        if not (m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                and bn in where):
          continue
        side, first, last = where[bn]
        R = d.geom_xmat[g].reshape(3, 3)
        ends = [d.geom_xpos[g] + s * R[:, 2] * m.geom_size[g][1] for s in (1, -1)]
        ys = [e[1] for e in ends]
        for k, e in enumerate(ends):
          w = e - np.array([0, 0, m.geom_size[g][0]])
          Rb = d.xmat[b].reshape(3, 3)
          kind = "-"
          if first and ys[k] == max(ys):
            kind = "heel"
          if last and ys[k] == min(ys):
            kind = "toe"
          self.fp_body.append(b)
          self.fp_local.append(Rb.T @ (w - d.xpos[b]))
          self.fp_kind.append((side, kind))
    elif sb:
      from mocap2myo import landmarks as L
      for sp in L.sole_points(L.Neutral(p, "floor", m)):
        self.fp_body.append(m.body(sp["body"]).id)
        self.fp_local.append(sp["local"])
        self.fp_kind.append((sp["side"], sp["kind"]))
      d.qpos[:] = self.qref
      mujoco.mj_kinematics(m, d)
    self.fp_body = np.array(self.fp_body, int)
    self.fp_local = np.array(self.fp_local).reshape(-1, 3)
    self.heel_pts = {sd: [i for i, k in enumerate(self.fp_kind) if k == (sd, "heel")] for sd in "lr"}
    self.toe_pts = {sd: [i for i, k in enumerate(self.fp_kind) if k == (sd, "toe")] for sd in "lr"}

  def foot_points(self):
    """World sole points for the pose currently in self.d."""
    R = self.d.xmat[self.fp_body].reshape(-1, 3, 3)
    return self.d.xpos[self.fp_body] + np.einsum("nij,nj->ni", R, self.fp_local)

  def set_offsets(self, off):
    self.m.site_pos[self.sid] = off

  def pose(self, x, q_root_ref):
    """x = [root pos 3, root rotvec 3 (relative to q_root_ref), self.free joints]."""
    q = self.qref.copy()
    q[:3] = x[:3]
    r = sRot.from_rotvec(x[3:6]) * sRot.from_quat(q_root_ref[[1, 2, 3, 0]])
    q[3:7] = r.as_quat()[[3, 0, 1, 2]]
    q[self.adr] = x[6:]
    apply_joint_equalities(self.m, q)
    return q

  def markers(self, q):
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    return self.d.site_xpos[self.sid].copy()

  def solve(self, P, x0, q_root_ref, x_prev=None, smooth=0.0, contact=None, outlier=None,
            mtp_prior=0.0, flat=0.0, robust=None, posture=None):
    """IK for one frame. P [n, 3] (LABELS order) with NaN = missing.

    robust (optional, m; replaces the outlier re-solve): Cauchy loss on each
    marker's distance, rho = f^2 log(1 + (d / f)^2) with f = robust, so a
    marker's pull fades smoothly as it moves away (influence peaks at d = f and
    decays like 1 / d). The outlier re-solve switches markers fully in or out:
    a marker hovering at the threshold flipped between frames, the fit jumped,
    and since only the joints are smoothed / speed-limited the root took the
    jump (case study: root acceleration p99 80 m/s^2, >100 m/s^2 in 0.7 % of
    frames, 10x enriched at outlier-set changes). With robust,
    `outlier` only marks 1-3 markers farther than it as NaN in the returned
    error (mislabels, as the re-solve; more = a bad solve, kept for the QA); the
    solve is untouched. The 4 pelvis markers keep the squared loss (bony
    landmarks on the root body, made rigid in the conversion): with their pull
    capped too, a deep stoop could leave the pelvis behind -- the hips ran to
    their 120 deg limit with the pelvis markers 16 cm off, then the fit snapped
    back, the pelvis turning 50 deg in 0.1 s against the lumbar (case study, with
    the root warm-started from the previous frame: pelvis markers > 5 cm off in
    0.36 % of frames; 650 frames of pelvis rotation > 300 deg/s).

    outlier (optional, m): if 1-3 markers end up farther than this from the
    model, they are treated as mislabelled for this frame and the frame is
    solved again without them; their error is returned as NaN. More than 3 is
    taken as a bad solve, not bad labels, and left as it is. Head markers are
    never rejected (no neck joint in the model: they are off by design, at low
    weight).

    mtp_prior (optional, per rad): pulls both toe joints toward 0. No marker
    sits on the toes (TOE is on calcn), so without it mtp drifts to its limit
    (case study: -30 deg in nearly every frame, toe tips 5-8 cm up while
    standing); the floor term still bends the toes when the heel rises.

    flat (optional, per m): pulls each foot's heel and toe-tip sole points to
    the same height -- used on quiet-standing frames during calibration, so
    that the shoe (TOE marker on the toe box, thick heel) ends up in the
    marker offsets instead of a pitched foot.

    contact (optional) = dict(floor=h, anchors={point index: xyz}, max_step=rad,
    w_floor, w_anchor): sole points may not go below the floor z = h, anchored
    sole points (stance) are pulled to their anchor, and every joint moves at
    most max_step from x_prev (velocity limit: no IK flips between frames).

    posture (optional) = (idx, target, weight): pulls FREE joints idx toward
    target (rad) with weight (per rad). For the static-trial calibration
    (calibrate_static): the subject stands in a known, near-neutral pose.
    """
    PELVIS_MASK, HEAD = self.ms.PELVIS_MASK, self.ms.HEAD
    ok = ~np.isnan(P[:, 0]) & self.active
    w = np.sqrt(self.w[ok])[:, None]
    lo = np.r_[[-np.inf] * 3, [-np.pi] * 3, self.lo]
    hi = np.r_[[np.inf] * 3, [np.pi] * 3, self.hi]
    if contact is not None and x_prev is not None:
      st = contact["max_step"]
      lo[6:] = np.maximum(lo[6:], x_prev[6:] - st)
      hi[6:] = np.minimum(hi[6:], x_prev[6:] + st)
    if contact is not None:
      aid = np.array(sorted(contact["anchors"]), int)
      atg = np.array([contact["anchors"][i] for i in aid]).reshape(-1, 3)

    def res(x):
      e = (self.markers(self.pose(x, q_root_ref))[ok] - P[ok]) * w
      if robust:
        # per marker: residual vector of norm f sqrt(log(1 + (|e| / f)^2)), along e
        n = np.linalg.norm(e, axis=1, keepdims=True)
        rn = robust * np.sqrt(np.log1p((n / robust) ** 2))
        rn = np.where(PELVIS_MASK[ok][:, None], n, rn)  # pelvis: squared loss
        e = e * (rn / np.maximum(n, 1e-12))
      r = [e.ravel()]
      if x_prev is not None and np.any(smooth > 0):
        r.append(smooth * (x[6:] - x_prev[6:]))
      if mtp_prior and len(self.mtp):
        r.append(mtp_prior * (x[6 + self.mtp] - self.mtp_q0))
      if posture is not None:
        pi, pt, pw = posture
        r.append(pw * (x[6 + pi] - pt))
      if flat:
        fp = self.foot_points()  # pose already set by markers()
        r.append(flat * np.array([fp[self.heel_pts[sd], 2].mean() - fp[self.toe_pts[sd], 2].mean()
                                  for sd in "lr"]))
      if contact is not None:
        fp = self.foot_points()  # pose already set by markers()
        r.append(contact["w_floor"] * np.minimum(fp[:, 2] - contact["floor"], 0.0))
        if len(aid) and contact.get("centroid"):
          # one point per foot: the centroid of its anchored sole points (the
          # foot keeps its place but may turn about it, see ik_trial)
          for sd in ("l", "r"):
            sel = [k for k, i in enumerate(aid) if self.fp_kind[i][0] == sd]
            if sel:
              r.append(contact["w_anchor"] * (fp[aid[sel]].mean(0) - atg[sel].mean(0)))
        elif len(aid):
          r.append(contact["w_anchor"] * (fp[aid] - atg).ravel())
      return np.concatenate(r)

    sol = scipy.optimize.least_squares(res, np.clip(x0, lo + 1e-9, hi - 1e-9),
                                       bounds=(lo, hi), xtol=1e-6, ftol=1e-6,
                                       max_nfev=100)
    err = np.linalg.norm(self.markers(self.pose(sol.x, q_root_ref)) - P, axis=1)
    err[~self.active] = np.nan
    if robust:
      if outlier is not None:  # as the re-solve: 1-3 far markers = mislabels (NaN in
        bad = (np.nan_to_num(err) > outlier) & ~HEAD  # the error), more = a bad solve
        if 0 < bad.sum() <= 3:
          err[bad] = np.nan
      return sol.x, err
    if outlier is not None:
      bad = (np.nan_to_num(err) > outlier) & ~HEAD
      if 0 < bad.sum() <= 3:
        P2 = P.copy()
        P2[bad] = np.nan
        return self.solve(P2, sol.x, q_root_ref, x_prev, smooth, contact,
                          mtp_prior=mtp_prior, flat=flat, posture=posture)
    return sol.x, err


def anat_offsets(mm):
  """Marker offsets in body frames from the marker set's anatomical placement: the
  explicit guesses of its [override] for this model (Plug-in-Gait on MyoFullBody: world,
  qpos0 with the root at (0, 0, 0.99), identity), else each marker's landmark + offset on
  the model at its neutral pose (mocap2myo.landmarks; markers whose landmark the model
  lacks stay at their body's origin and are switched off). On a model scaled to the subject
  (mm.scaling) the guesses are those of the unscaled model, moved with their bodies (x A),
  as a model's markers are scaled with it in OpenSim."""
  if getattr(mm, "scaling", None) is not None:
    mm0 = MarkerModel(mm.profile, mm.ms, neck=mm.neck)
    off = anat_offsets(mm0)
    mm.active &= mm0.active
    mm.w = np.where(mm.active, mm.w, 0.0)
    return np.einsum("kij,kj->ki", mm.scaling.A[mm.bid], off)
  m, d, ms = mm.m, mm.d, mm.ms
  LABELS = ms.LABELS
  off = np.zeros((len(LABELS), 3))
  if ms.has_override(mm.profile.name):
    d.qpos[:] = m.qpos0
    d.qpos[:3] = mm.profile.neutral_cfg.get("root_pos", [0, 0, 0.99])
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_kinematics(m, d)
    for k, l in enumerate(LABELS):
      R = d.xmat[mm.bid[k]].reshape(3, 3)
      off[k] = R.T @ (ms.anat_world(l) - d.xpos[mm.bid[k]])
    return off
  from mocap2myo import landmarks as L
  nt = L.Neutral(mm.profile, "floor", m)
  lm = L.compute(mm.profile, neutral=nt).landmarks
  for k, l in enumerate(LABELS):
    if not mm.active[k]:
      continue
    g = ms.guess_world(l, lm)
    if g is None:
      mm.active[k] = False
      mm.w[k] = 0.0
      print(f"  {mm.profile.name}: no landmark {ms.markers[l].landmark} for {l}: marker not used")
      continue
    R = nt.d.xmat[mm.bid[k]].reshape(3, 3)
    off[k] = R.T @ (g - nt.d.xpos[mm.bid[k]])
  return off


def root_from_pelvis(mm, P):
  """Root pose from a rigid (Kabsch) fit of the model pelvis markers."""
  m, d = mm.m, mm.d
  PELVIS = mm.ms.PELVIS
  d.qpos[:] = mm.qref
  d.qpos[:3] = 0
  d.qpos[3:7] = [1, 0, 0, 0]
  mujoco.mj_kinematics(m, d)
  A = d.site_xpos[mm.sid[PELVIS]]  # model, root at origin/identity
  B = P[PELVIS]
  ca, cb = A.mean(0), B.mean(0)
  H = (A - ca).T @ (B - cb)
  U, _, Vt = np.linalg.svd(H)
  D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
  R = Vt.T @ D @ U.T
  pos = cb - R @ ca
  return pos, sRot.from_matrix(R).as_quat()[[3, 0, 1, 2]]


# Joints held near neutral in the static (anatomical) standing trial, see
# calibrate(static=...): MyoFullBody's; a model's own are its profile's [ik] static_neutral
STATIC_NEUTRAL = [f"{j}_{s}" for s in ("r", "l") for j in
                  ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle",
                   "subtalar_angle")]


# Canonical segments whose marker placement the static trial decides (the posture prior
# holds for them); see calibrate(static_segments=...)
STATIC_SEGMENTS_LEGS = ("pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l", "foot_r", "foot_l")
# legs only: the pelvis markers stay with the motion trials (skin over the iliac
# crests slides in deep stoops; placed from the upright static pose they made the
# pelvis jump in two case-study subjects when both ASIS were hidden by the box)
STATIC_SEGMENTS_LEGS_ONLY = STATIC_SEGMENTS_LEGS[1:]


def calibrate(mm, trials, n_frames=24, iters=5, prior=2.0, max_shift=0.06, seed=0,
              outlier=None, stand=None, flat=10.0, robust=None, static=None,
              static_share=0.5, static_w=1.0, static_segments=None):
  """Fit per-marker body offsets for one subject from several trials.
  robust: MarkerModel.solve (Cauchy marker loss instead of the outlier re-solve).
  max_shift: float or per-marker array [n] (m) bounding the move from the
  anatomical guess. outlier: see MarkerModel.solve; rejected samples are left
  out of the offset update. stand (optional): extra quiet-standing frames
  [k, n, 3], solved with the flat-foot term (weight `flat`).

  static (optional, default off): frames [k, n, 3] of the subject's static
  standing trial. They are solved with flat feet and the STATIC_NEUTRAL joints
  pulled toward 0 (weight static_w per rad), and carry static_share of the
  weight of every offset update. Without them the offsets and the pose are not
  separable: a knee marker a few cm further back on the femur fits as well as a
  knee bent 20 deg, and the dynamic frames alone settled on bent knees / tilted
  pelvis in some subjects (case study: quiet-standing knee flexion 10-24 deg in 16
  of 22 trials, confirmed against Xsens on one trial). With the static trial: knee
  ~0 deg, residual unchanged or lower in most subjects, at most +2 mm. "Near neutral":
  the profile's [ik] static_neutral joints toward the model's neutral pose (MyoFullBody:
  hips, knees, ankles, subtalar at 0).
  static_segments (optional): only markers on these canonical segments take their placement
  from the static frames (e.g. STATIC_SEGMENTS_LEGS); with all markers, the hand
  markers were fitted to the loose hanging wrists of the static pose and their
  leave-one-out prediction got worse (finger 33 -> 48 mm)."""
  rng = np.random.default_rng(seed)
  m, d = mm.m, mm.d
  LABELS, PELVIS = mm.ms.LABELS, mm.ms.PELVIS
  off0 = anat_offsets(mm)
  off = off0.copy()
  mm.set_offsets(off)
  # diverse frames: spread over all trials, require full pelvis markers
  frames = []
  for P in trials:
    idx = rng.choice(len(P), size=max(1, n_frames // len(trials)), replace=False)
    frames += [P[i] for i in idx if not np.isnan(P[i, PELVIS]).any()]
  n_rand = len(frames)
  if stand is not None:
    frames += [P for P in stand if not np.isnan(P[PELVIS]).any()]
  n_dyn = len(frames)
  wt = np.ones(n_dyn)
  posture = None
  if static is not None:
    st = [P for P in static if not np.isnan(P[PELVIS]).any()]
    frames += st
    wt = np.r_[wt, np.full(len(st), static_share / (1 - static_share) * n_dyn / max(len(st), 1))]
    # with the optional neck, the static trial also holds the neck near neutral: the head
    # markers' placement and the neck angles are not separable otherwise (as for the knees)
    pidx = np.array([mm.free.index(j) for j in mm.static_neutral + (list(NECK) if mm.neck else [])])
    posture = (pidx, mm.qref[mm.adr[pidx]], static_w)  # the neutral pose (MyoFullBody: 0)
  use_static = np.ones(len(LABELS), bool)
  if static_segments is not None:
    use_static = np.array([mm.ms.segment(l) in static_segments for l in LABELS])
  for it in range(iters):
    mm.set_offsets(off)
    poses, used = [], []
    for fi, P in enumerate(frames):
      fl = flat if fi >= n_rand else 0.0
      is_static = fi >= n_dyn
      pos, q = root_from_pelvis(mm, P)
      x0 = np.r_[pos, np.zeros(3), mm.qref[mm.adr]]
      best = None
      for k in range(3 if it == 0 and not is_static else 1):
        xs = x0.copy()
        if k:
          xs[6:] = mm.lo + rng.random(len(mm.lo)) * (mm.hi - mm.lo)
        x, err = mm.solve(P, xs, q, outlier=outlier, flat=fl, mtp_prior=0.1 if fl else 0.0,
                          robust=robust, posture=posture if is_static else None)
        if best is None or np.nanmean(err) < np.nanmean(best[1]):
          best = (x, err)
      poses.append(mm.pose(best[0], q))
      used.append(~np.isnan(best[1]))
    # closed-form offset update with a prior toward the current offsets
    acc = np.zeros_like(off)
    cnt = np.zeros(len(LABELS))
    for fi, (P, qq, u, w) in enumerate(zip(frames, poses, used, wt)):
      d.qpos[:] = qq
      mujoco.mj_kinematics(m, d)
      for k in range(len(LABELS)):
        if np.isnan(P[k, 0]) or not u[k] or (fi >= n_dyn and not use_static[k]):
          continue
        R = d.xmat[mm.bid[k]].reshape(3, 3)
        acc[k] += w * (R.T @ (P[k] - d.xpos[mm.bid[k]]))
        cnt[k] += w
    fit = acc / np.maximum(cnt, 1e-9)[:, None]  # weighted counts (static frames)
    off = np.where(cnt[:, None] > 0, (cnt[:, None] * fit + prior * off0)
                   / (cnt + prior)[:, None], off0)
    # keep each marker within max_shift of its anatomical guess
    dlt = off - off0
    n = np.linalg.norm(dlt, axis=1, keepdims=True)
    cap = np.broadcast_to(np.asarray(max_shift, float), (len(LABELS),))[:, None]
    off = off0 + dlt * np.minimum(1.0, cap / np.maximum(n, 1e-9))
    mm.set_offsets(off)
    errs = []
    for P, qq in zip(frames, poses):
      e = np.linalg.norm(mm.markers(qq) - P, axis=1)
      errs.append(np.nanmean(e[mm.active]))
    print(f"  calib iter {it}: mean marker err {1e3 * np.mean(errs):.1f} mm "
          f"(max shift {1e3 * np.linalg.norm(off - off0, axis=1).max():.0f} mm)")
  return off


def sole_stance(P, rate, v_max=0.1, dz=0.02, min_s=0.1, ms=None):
  """Stance mask per sole marker {(side, heel|toe): [T] bool} from the markers:
  slow (< v_max m/s) and low (< its 5th-percentile height + dz), runs shorter
  than min_s dropped. ms: the marker set (default the selected one)."""
  ms = ms or _ms.MARKER_SET
  LABELS, SOLE = ms.LABELS, ms.SOLE
  out = {}
  for side in ("l", "r"):
    for kind in ("heel", "toe"):
      p = P[:, LABELS.index(SOLE[(side, kind)])]
      v = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) * rate]
      k = max(1, int(0.05 * rate))
      v = np.convolve(np.nan_to_num(v, nan=1e3), np.ones(k) / k, "same")
      z = p[:, 2]
      s = (v < v_max) & (z < np.nanpercentile(z, 5) + dz) & ~np.isnan(z)
      idx = np.where(s)[0]
      for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
        if 0 < len(r) < min_s * rate:
          s[r] = False
      out[(side, kind)] = s
  return out


def standing(P, rate, ms=None):
  """Quiet upright standing frames from the markers (scaled-subject frame); ms: the
  marker set (default the selected one)."""
  ms = ms or _ms.MARKER_SET
  L, PELVIS, ANKLES = ms.LABELS, ms.PELVIS, ms.ANKLES
  pel = P[:, PELVIS].mean(1)
  ank = P[:, [L.index(l) for l in ANKLES]]

  def speed(x):  # [T] m/s; for several points the fastest one; NaN when unseen
    s = np.linalg.norm(np.diff(x, axis=0), axis=-1) * rate
    return np.r_[np.nan, s.max(1) if s.ndim == 2 else s]

  slow = (speed(pel) < 0.05) & (speed(ank) < 0.05)
  up = pel[:, 2] > np.nanpercentile(pel[:, 2], 95) - 0.03
  ok = slow & up
  w = int(0.5 * rate)  # sustained for 0.5 s
  run = np.convolve(ok.astype(float), np.ones(w), "same") >= w - 1
  return ok & run


def first_frame(mm, P, restarts=8, seed=0, outlier=None, mtp_prior=0.0, robust=None):
  """ik_trial's first solved frame, P [n, 3] with all pelvis markers: the root from the
  pelvis markers (root_from_pelvis), then `restarts` solves, the first from the reference
  pose, the others from joint angles drawn uniformly within their limits (seed: an int or a
  numpy Generator); the best by mean marker error. Returns x, err (MarkerModel.solve) and
  q_ref, the root orientation x's rotation vector is relative to."""
  rng = np.random.default_rng(seed)
  pos, q_ref = root_from_pelvis(mm, P)
  cands = []
  for k in range(restarts):
    xs = np.r_[pos, np.zeros(3), mm.qref[mm.adr]]
    if k:
      xs[6:] = mm.lo + rng.random(len(mm.lo)) * (mm.hi - mm.lo)
    if mtp_prior:
      xs[6 + mm.mtp] = mm.mtp_q0
    cands.append(mm.solve(P, xs, q_ref, outlier=outlier, mtp_prior=mtp_prior, robust=robust))
  x, err = min(cands, key=lambda c: np.nanmean(c[1]))
  return x, err, q_ref


# tracking loss (ik_trial recover): a frame whose fit leaves LOST_N or more seen body markers
# farther than LOST_DIST from the model is solved again from scratch (RECOVER_STARTS starts,
# no smoothing, no speed limit); the fits compare by the sum of squared distances, each capped
# at LOST_CAP (a mislabelled marker cannot dominate), and the new one is kept if it is better
# by RECOVER_GAIN
LOST_DIST, LOST_N, LOST_CAP, RECOVER_STARTS, RECOVER_GAIN = 0.06, 2, 0.15, 4, 0.8


LIMBS = {"arm_l": ("humerus_l", "radius_l", "ulna_l", "hand_l"),
         "arm_r": ("humerus_r", "radius_r", "ulna_r", "hand_r"),
         "leg_l": ("femur_l", "tibia_l", "foot_l"), "leg_r": ("femur_r", "tibia_r", "foot_r")}


def limb_lost(E, rejected, labels, ms):
  """Frames where the IK lost a limb: LOST_N or more of the limb's markers (LIMBS, by the
  marker set's segments) farther than LOST_DIST or rejected as mislabelled (a lost limb's
  markers can look like 1-3 mislabels). E: marker error [T, n] (NaN: missing or rejected),
  rejected [T, n] bool or None, labels [n]. Returns {limb: [T] bool}."""
  out = {}
  for limb, segs in LIMBS.items():
    idx = [i for i, l in enumerate(labels) if l in ms.markers and ms.segment(l) in segs]
    if len(idx) < LOST_N:
      continue
    far = np.nan_to_num(E[:, idx], nan=0.0) > LOST_DIST
    if rejected is not None:
      far |= np.asarray(rejected[:, idx], bool)
    out[limb] = far.sum(1) >= LOST_N
  return out


def longest_run(mask):
  """Length of the longest run of True in a 1-D bool array."""
  if not mask.any():
    return 0
  d = np.diff(np.r_[0, mask.astype(int), 0])
  return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def fit_score(mm, x, q_ref, P):
  """Sum over the seen body markers (head left out) of the squared model-marker distance,
  each capped at LOST_CAP; the distances [n]; how many exceed LOST_DIST."""
  d = np.linalg.norm(mm.markers(mm.pose(x, q_ref)) - P, axis=1)
  use = np.isfinite(d) & mm.active & ~mm.ms.HEAD
  return float(np.sum(np.minimum(d[use], LOST_CAP) ** 2)), d, int(np.sum(d[use] > LOST_DIST))


def recover_frame(mm, P, xs, q_ref, x, err, seed, t, contact=None, outlier=None, mtp_prior=0.0,
                  robust=None):
  """ik_trial's tracking-loss step for frame t: (x, err, recovered) -- x unchanged unless
  LOST_N or more markers are lost and a fresh solve fits better (see LOST_DIST)."""
  s0, _, lost = fit_score(mm, x, q_ref, P)
  if lost < LOST_N:
    return x, err, False
  rng = np.random.default_rng([seed, t])
  best = (s0, x, err)
  for k in range(RECOVER_STARTS):
    xs2 = xs.copy()
    if k:  # the first from the warm start, the others from random joint angles
      xs2[6:] = mm.lo + rng.random(len(mm.lo)) * (mm.hi - mm.lo)
      if mtp_prior:
        xs2[6 + mm.mtp] = mm.mtp_q0
    xr, er = mm.solve(P, xs2, q_ref, contact=contact, outlier=outlier, mtp_prior=mtp_prior,
                      robust=robust)
    s = fit_score(mm, xr, q_ref, P)[0]
    if s < best[0]:
      best = (s, xr, er)
  if best[0] < RECOVER_GAIN * s0:
    return best[1], best[2], True
  return x, err, False


def ik_trial(mm, P, smooth=0.05, restarts=8, seed=0, rate=None, contact=False,
             v_joint=15.0, w_floor=10.0, w_anchor=3.0,
             ankle_smooth=0.08, outlier=None, mtp_prior=0.0, anchor_centroid=False,
             pelvis_gap="carry", robust=None, root_init="prev", recover=False, progress=None):
  """Per-frame IK over a whole trial. progress: called as progress(t, T) every tenth of the
  trial. Returns qpos [T, nq] and marker err [T, n]
  (NaN where a marker is missing or was rejected, see MarkerModel.solve).

  Unsolved frames have an all-NaN row in marker err (that is how to find them):
  the frames before the first one with all 4 pelvis markers, and with
  pelvis_gap="carry" every later frame with a pelvis marker missing. Their qpos is
  not left empty: the frames before the first solved one hold its pose, and a
  carried frame repeats the previous pose.

  anchor_centroid: lock each stance foot at the centroid of its anchored sole
  points instead of every point, so it may turn about it. MyoFullBody's knee
  has no axial rotation: in a pivot turn on a planted foot (case study) the
  whole twist went into hip rotation, which sat on its -40 deg limit and
  snapped back when the foot lifted (500-860 deg/s).

  pelvis_gap: what a frame with a missing pelvis marker gets once the first
  frame is solved. "carry" (the function's default) repeats the previous pose;
  "solve" (the pipeline's default, M2M_PELVIS_GAP) solves it from the markers
  that are there (the root then rests on the thigh / trunk markers and the warm
  start). Case study: the carried box hides the front pelvis markers for 0.3-1.4 s
  per bend, and carrying froze the reference there, then jumped it 0.1-0.6 m in
  one frame (15 of 107 clips).

  root_init: warm start of the root in each frame. "prev" (default): the previous
  frame's solution, like the joints. "pelvis": the rigid fit of this frame's 4
  pelvis markers (root_from_pelvis) when all 4 are there, else "prev". Warm-
  starting the root from the previous frame let the fit slide into a wrong
  basin during a deep stoop and stay there (case study: the root sank 9 cm, the
  hips ran to their 120 deg limit, pelvis markers 16 cm off for up to 0.4 s, then
  a 50 deg snap back; the same frames solved from scratch fit).

  contact=True (needs rate): sole points kept above the floor (the lowest sole
  point of the first, standing frame), sole points in marker-detected stance
  locked where they touched down (no skating), joint speed <= v_joint rad/s.

  smooth / ankle_smooth (and the gated joints' 0.05-1.05) are per-frame weights at
  SMOOTH_RATE (100 Hz), scaled by sqrt(rate / SMOOTH_RATE): each frame is pulled toward
  the previous frame's solution, so the smoothing lags by a number of frames that grows
  with the square of the weight; the square root keeps that lag the same in seconds at
  any frame rate (scaling the weight with the rate itself doubled the lag at 200 Hz).
  """
  rng = np.random.default_rng(seed)
  m = mm.m
  LABELS, PELVIS = mm.ms.LABELS, mm.ms.PELVIS
  T = len(P)
  Q = np.zeros((T, m.nq))
  E = np.full((T, len(LABELS)), np.nan)
  x_prev, q_ref = None, None
  stance = sole_stance(P, rate, ms=mm.ms) if contact else None
  # smoothing weights are per frame at SMOOTH_RATE; the lag of the frame-to-frame pull grows
  # with the weight squared, so sqrt(rate) keeps it the same in seconds (x 1.0 at 100 Hz)
  fs = np.sqrt(rate / SMOOTH_RATE) if rate else 1.0
  if not contact:
    smooth = smooth * fs
  if contact:
    # stronger frame-to-frame smoothing where the IK chattered: ankle/subtalar
    # wobbled at the speed limit in deep squats; elv_angle (shoulder elevation
    # plane) is undefined with the arm hanging and flipped by pi between the
    # two equivalent fits -- so it is held only while shoulder_elv is small
    # (a constant 1.0 lagged the raised arm: elbow marker error 8.7 -> 14 mm;
    # 0.2 on every joint lagged fast clips)
    base = np.where(mm.smooth_heavy, ankle_smooth, smooth) * fs
    gated = mm.smooth_gated  # (joint, gate joint) index pairs into mm.free

    def smooth_at(x):
      w = base.copy()
      for e, se in gated:
        w[e] = (smooth + 1.0 * np.exp(-(x[6 + se] / 0.35) ** 2)) * fs
      return w
  floor, anchors, last_t = None, {}, None

  def contact_args(t):
    if floor is None:
      return None
    for i, (side, kind) in enumerate(mm.fp_kind):
      if kind == "-" or not stance[(side, kind)][t]:
        anchors.pop(i, None)
    return dict(floor=floor, anchors=dict(anchors), w_floor=w_floor,
                centroid=anchor_centroid,
                w_anchor=w_anchor, max_step=v_joint * (t - last_t) / rate)

  def touch_down(t):
    fp = mm.foot_points()  # pose of the frame just solved
    for i, (side, kind) in enumerate(mm.fp_kind):
      if kind != "-" and stance[(side, kind)][t] and i not in anchors:
        anchors[i] = np.r_[fp[i, :2], max(fp[i, 2], floor)]

  for t in range(T):
    if progress is not None and t and t % max(1, T // 10) == 0:
      progress(t, T)
    if np.isnan(P[t, PELVIS]).any() and (x_prev is None or pelvis_gap == "carry"):
      if x_prev is not None:  # no pelvis -> carry previous pose
        Q[t] = mm.pose(x_prev, q_ref)
      continue
    if x_prev is None:
      x, err, q_ref = first_frame(mm, P[t], restarts, rng, outlier=outlier,
                                  mtp_prior=mtp_prior, robust=robust)
    else:
      xs = x_prev.copy()
      if root_init == "pelvis" and not np.isnan(P[t, PELVIS, 0]).any():
        xs[:3], q_ref = root_from_pelvis(mm, P[t])
      else:  # re-centre the rotvec on the previous root orientation
        q_ref = mm.pose(x_prev, q_ref)[3:7]
      xs[3:6] = 0
      ca = contact_args(t) if contact else None
      x, err = mm.solve(P[t], xs, q_ref, x_prev=xs,
                        smooth=smooth_at(xs) if contact else smooth,
                        contact=ca, outlier=outlier, mtp_prior=mtp_prior, robust=robust)
      if recover:
        x, err, _ = recover_frame(mm, P[t], xs, q_ref, x, err, seed, t, contact=ca,
                                  outlier=outlier, mtp_prior=mtp_prior, robust=robust)
    Q[t] = mm.pose(x, q_ref)
    E[t] = err
    x_prev = x
    if contact:
      mm.markers(Q[t])  # sets the pose for foot_points()
      if floor is None:
        floor = float(mm.foot_points()[:, 2].min())
      touch_down(t)
      last_t = t
  # frames before the first solvable one: hold the first solved pose
  solved = np.where(np.linalg.norm(Q[:, 3:7], axis=1) > 0.5)[0]
  if len(solved) and solved[0] > 0:
    Q[: solved[0]] = Q[solved[0]]
  return Q, E


# the selected marker set's names (M2M_MARKER_SET), re-exported for older callers; the
# functions above take the marker set from the MarkerModel instead
ANKLES, HEAD, LABELS, MARKERS = _ms.ANKLES, _ms.HEAD, _ms.LABELS, _ms.MARKERS
PELVIS, PELVIS_LABELS, PELVIS_MASK, SOLE = _ms.PELVIS, _ms.PELVIS_LABELS, _ms.PELVIS_MASK, _ms.SOLE
anat_world = _ms.anat_world
