"""Marker-based IK from the MMH Plug-in-Gait markers onto MyoFullBody.

Data: ~/Projects/datasets/MMH/npz/<S>_<L>.npz (from c3d_to_npz.py): 39 standard
Plug-in-Gait markers at 120 Hz, metres, Z-up.

Pipeline per subject:
  1. scale the marker cloud by MODEL_HEIGHT / subject height (the model is not
     geometrically scaled; per-marker body offsets absorb what remains);
  2. calibrate marker offsets on their bodies: initial offsets from a standing
     frame with the model in its neutral pose, then alternate (IK on K diverse
     frames) <-> (closed-form offset update with a prior toward the previous
     offsets), as in OpenSim/AddBiomechanics marker registration;
  3. per-frame IK, warm-started from the previous frame.
Unknowns: root (3 + 3 rotvec), 14 leg, 3 lumbar, 2 x 7 arm DoF = 37. Every
other joint follows through the model's joint equalities or stays neutral.
"""

from __future__ import annotations

import numpy as np
import mujoco
import scipy.optimize
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as sRot

from mocap2myo import objects as O
from mocap2myo.kinematics import apply_joint_equalities
from mocap2myo.model import MYOFULLBODY_XML

MODEL_HEIGHT = 1.72  # bone height 1.689 m + soft tissue
# marker -> (body, weight). Head is rigid on the thorax in MyoFullBody (no neck
# joint), so head markers get a low weight; thigh/shank wands and the
# asymmetric RBAK marker are soft-tissue/placement noisy -> medium weight.
MARKERS = {
  "LFHD": ("head", 0.3), "RFHD": ("head", 0.3), "LBHD": ("head", 0.3),
  "RBHD": ("head", 0.3),
  "C7": ("thorax", 1.0), "T10": ("thorax", 1.0), "CLAV": ("thorax", 1.0),
  "STRN": ("thorax", 1.0), "RBAK": ("thorax", 0.3),
  "LSHO": ("scapula_l", 1.0), "LUPA": ("humerus_l", 0.5),
  "LELB": ("humerus_l", 1.0), "LFRM": ("radius_l", 0.5),
  "LWRA": ("radius_l", 1.0), "LWRB": ("ulna_l", 1.0), "LFIN": ("thirdmc_l", 1.0),
  "RSHO": ("scapula_r", 1.0), "RUPA": ("humerus_r", 0.5),
  "RELB": ("humerus_r", 1.0), "RFRM": ("radius_r", 0.5),
  "RWRA": ("radius_r", 1.0), "RWRB": ("ulna_r", 1.0), "RFIN": ("thirdmc_r", 1.0),
  "LASI": ("pelvis", 1.0), "RASI": ("pelvis", 1.0), "LPSI": ("pelvis", 1.0),
  "RPSI": ("pelvis", 1.0),
  "LTHI": ("femur_l", 0.5), "LKNE": ("femur_l", 1.0), "LTIB": ("tibia_l", 0.5),
  "LANK": ("tibia_l", 1.0), "LHEE": ("calcn_l", 1.0), "LTOE": ("calcn_l", 1.0),
  "RTHI": ("femur_r", 0.5), "RKNE": ("femur_r", 1.0), "RTIB": ("tibia_r", 0.5),
  "RANK": ("tibia_r", 1.0), "RHEE": ("calcn_r", 1.0), "RTOE": ("calcn_r", 1.0),
}
LABELS = list(MARKERS)
HEAD = np.array([l.endswith("HD") for l in LABELS])
PELVIS_MASK = np.array([l[1:] in ("ASI", "PSI") for l in LABELS])
# Anatomical guesses (world, model at qpos0, root at (0,0,0.99) identity: faces
# -y, left = +x) for the LEFT-side / midline markers; right side mirrors x.
# Built from the model's joint centres (hip 0.077,0.056,0.911; knee z .507;
# ankle z .107; shoulder .17,.10,1.464; elbow .158,.093,1.174; wrist .203,.075,
# .92) plus typical skin-marker offsets. Calibration refines them.
_ANAT_L = {
  "FHD": (0.06, 0.02, 1.68), "BHD": (0.06, 0.19, 1.66),
  "SHO": (0.17, 0.10, 1.51), "UPA": (0.23, 0.10, 1.32), "ELB": (0.193, 0.093, 1.17),
  "FRM": (0.21, 0.09, 1.04), "WRA": (0.205, 0.045, 0.925), "WRB": (0.205, 0.105, 0.925),
  "FIN": (0.215, 0.065, 0.81),
  "ASI": (0.125, -0.014, 0.98), "PSI": (0.05, 0.146, 0.985),
  "THI": (0.147, 0.056, 0.71), "KNE": (0.126, 0.056, 0.507), "TIB": (0.136, 0.06, 0.31),
  "ANK": (0.116, 0.066, 0.10), "HEE": (0.084, 0.135, 0.045), "TOE": (0.08, -0.055, 0.035),
}
_ANAT_MID = {"C7": (0, 0.17, 1.47), "T10": (0, 0.16, 1.23), "CLAV": (0, 0.04, 1.44),
             "STRN": (0, 0.0, 1.26), "RBAK": (-0.08, 0.18, 1.36)}


def anat_world(label):
  if label in _ANAT_MID:
    return np.array(_ANAT_MID[label], float)
  side, key = label[0], label[1:]
  x, y, z = _ANAT_L[key]
  return np.array([x if side == "L" else -x, y, z], float)
FREE = [f"{j}_{s}" for s in ("r", "l") for j in
        ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle",
         "subtalar_angle", "mtp_angle")] + \
       ["flex_extension", "lat_bending", "axial_rotation"] + \
       [f"{j}_{s}" for s in ("r", "l") for j in
        ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flex", "pro_sup",
         "deviation", "flexion")]


LUMBAR = ("flex_extension", "lat_bending", "axial_rotation")


def load_trial(path, max_gap_s=0.25, lowpass_hz=None):
  """Standard markers [T, 39, 3] scaled to the model; short gaps interpolated.
  lowpass_hz (opt-in, default off = MMH / OMOMO unchanged): zero-phase 2nd-order
  Butterworth (filtfilt) per marker coordinate over each gap-free run, the usual
  marker filtering before IK (Winter: ~6 Hz for whole-body motion)."""
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
    for r in runs:
      if len(r) and len(r) <= max_gap_s * rate and r[0] > 0 and r[-1] < len(P) - 1:
        P[r, k] = filled[r]
  if lowpass_hz:
    bb, aa = butter(2, lowpass_hz / (rate / 2))
    pad = 3 * max(len(aa), len(bb))
    for k in range(len(LABELS)):
      idx = np.flatnonzero(~np.isnan(P[:, k, 0]))
      if len(idx):
        for r in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
          if len(r) > pad:
            P[r, k] = filtfilt(bb, aa, P[r, k], axis=0)
  s = MODEL_HEIGHT / float(d["height_m"])
  return P * s, rate, s


class MarkerModel:
  def __init__(self):
    spec = mujoco.MjSpec.from_file(str(MYOFULLBODY_XML))
    for l, (b, _) in MARKERS.items():
      spec.body(b).add_site(name=f"mk_{l}", pos=[0, 0, 0], size=[0.01, 0, 0])
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    m = self.m
    self.sid = np.array([m.site(f"mk_{l}").id for l in LABELS])
    # Sites created at the body origin get site_sameframe=1, and kinematics
    # then copies the body pose and ignores site_pos -- every offset written
    # later would silently do nothing. Clear the flag.
    m.site_sameframe[self.sid] = 0
    self.bid = np.array([m.body(MARKERS[l][0]).id for l in LABELS])
    self.w = np.array([MARKERS[l][1] for l in LABELS])
    self.adr = np.array([m.jnt_qposadr[m.joint(n).id] for n in FREE])
    self.mtp = np.array([FREE.index(f"mtp_angle_{sd}") for sd in ("r", "l")])
    jid = [m.joint(n).id for n in FREE]
    self.lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -np.pi for j in jid])
    self.hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else np.pi for j in jid])
    self._foot_points()
    self.hand_b = [m.body("thirdmc_l").id, m.body("thirdmc_r").id]
    pl, sp = O.hand_palmar_local(), O.hand_sphere_local()
    self.palm_n = [pl["l"], pl["r"]]
    self.palm_c = [sp["l"], sp["r"]]
    mujoco.mj_kinematics(m, self.d)  # qpos0 (set by _foot_points)
    self.palm_d = []  # distal (finger) axis in the thirdmc frame
    for sd, b in zip("lr", self.hand_b):
      v = self.d.xpos[m.body(f"3proxph_{sd}").id] - self.d.xpos[b]
      self.palm_d.append(self.d.xmat[b].reshape(3, 3).T @ (v / np.linalg.norm(v)))

  def palms(self):
    """Hand sphere centres, palm normals, finger axes [2, 3] each (l, r)."""
    c, n, f = np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3))
    for k, b in enumerate(self.hand_b):
      R = self.d.xmat[b].reshape(3, 3)
      c[k] = self.d.xpos[b] + R @ self.palm_c[k]
      n[k] = R @ self.palm_n[k]
      f[k] = R @ self.palm_d[k]
    return c, n, f

  def _foot_points(self):
    """Sole points: both ends of every foot capsule, radius below the axis (feet
    are flat at qpos0). heel = back ends of calcn capsules, toe = front ends of
    toes capsules (the model faces -y). Stored as (body, local offset)."""
    m, d = self.m, self.d
    d.qpos[:] = m.qpos0
    mujoco.mj_kinematics(m, d)
    self.fp_body, self.fp_local, self.fp_kind = [], [], []  # kind: (side, heel/toe/-)
    for g in range(m.ngeom):
      b = m.geom_bodyid[g]
      bn = m.body(b).name
      if not (m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
              and bn[:-2] in ("calcn", "toes")):
        continue
      R = d.geom_xmat[g].reshape(3, 3)
      ends = [d.geom_xpos[g] + s * R[:, 2] * m.geom_size[g][1] for s in (1, -1)]
      ys = [e[1] for e in ends]
      for k, e in enumerate(ends):
        w = e - np.array([0, 0, m.geom_size[g][0]])
        Rb = d.xmat[b].reshape(3, 3)
        kind = "-"
        if bn.startswith("calcn") and ys[k] == max(ys):
          kind = "heel"
        if bn.startswith("toes") and ys[k] == min(ys):
          kind = "toe"
        self.fp_body.append(b)
        self.fp_local.append(Rb.T @ (w - d.xpos[b]))
        self.fp_kind.append((bn[-1], kind))
    self.fp_body = np.array(self.fp_body)
    self.fp_local = np.array(self.fp_local)
    self.heel_pts = {sd: [i for i, k in enumerate(self.fp_kind) if k == (sd, "heel")] for sd in "lr"}
    self.toe_pts = {sd: [i for i, k in enumerate(self.fp_kind) if k == (sd, "toe")] for sd in "lr"}

  def foot_points(self):
    """World sole points for the pose currently in self.d."""
    R = self.d.xmat[self.fp_body].reshape(-1, 3, 3)
    return self.d.xpos[self.fp_body] + np.einsum("nij,nj->ni", R, self.fp_local)

  def set_offsets(self, off):
    self.m.site_pos[self.sid] = off

  def pose(self, x, q_root_ref):
    """x = [root pos 3, root rotvec 3 (relative to q_root_ref), FREE joints]."""
    q = self.m.qpos0.copy()
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
    """IK for one frame. P [39, 3] with NaN = missing.

    robust (optional, m; replaces the outlier re-solve): Cauchy loss on each
    marker's distance, rho = f^2 log(1 + (d / f)^2) with f = robust, so a
    marker's pull fades smoothly as it moves away (influence peaks at d = f and
    decays like 1 / d). The outlier re-solve switches markers fully in or out:
    a marker hovering at the threshold flipped between frames, the fit jumped,
    and since only the joints are smoothed / speed-limited the root took the
    jump (MMH Elevator: root acceleration p99 80 m/s^2, >100 m/s^2 in 0.7 % of
    frames, 10x enriched at outlier-set changes; MMH 8 m/s^2). With robust,
    `outlier` only marks 1-3 markers farther than it as NaN in the returned
    error (mislabels, as the re-solve; more = a bad solve, kept for the QA); the
    solve is untouched. The 4 pelvis markers keep the squared loss (bony
    landmarks on the root body, made rigid in the conversion): with their pull
    capped too, a deep stoop could leave the pelvis behind -- the hips ran to
    their 120 deg limit with the pelvis markers 16 cm off, then the fit snapped
    back, the pelvis turning 50 deg in 0.1 s against the lumbar (MMH Elevator
    IK v9: pelvis markers > 5 cm off in 0.36 % of frames, S10 / S11 / S06 /
    S01 most; 650 frames of pelvis rotation > 300 deg/s).

    outlier (optional, m): if 1-3 markers end up farther than this from the
    model, they are treated as mislabelled for this frame and the frame is
    solved again without them; their error is returned as NaN. More than 3 is
    taken as a bad solve, not bad labels, and left as it is. Head markers are
    never rejected (no neck joint: they are off by design, at low weight).

    mtp_prior (optional, per rad): pulls both toe joints toward 0. No marker
    sits on the toes (TOE is on calcn), so without it mtp drifts to its limit
    (Elevator v1: -30 deg in nearly every frame, toe tips 5-8 cm up while
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
    ok = ~np.isnan(P[:, 0])
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
      if mtp_prior:
        r.append(mtp_prior * x[6 + self.mtp])
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
        if contact.get("grasp2h"):
          # two-handed carry: palms face each other and the fingers run along
          # the side faces (finger axis _|_ the line between the hands).
          # 3 markers per hand leave the hand orientation loose; the IK
          # otherwise splayed the fingers outward (hands in a V, wrists
          # inboard of the palms) -> forearms inside the carried box.
          c, n, f = self.palms()
          u = c[0] - c[1]
          u /= max(np.linalg.norm(u), 1e-6)  # right -> left hand
          r.append(contact["w_grasp"] * np.r_[n[0] + u, n[1] - u, f @ u])
      return np.concatenate(r)

    sol = scipy.optimize.least_squares(res, np.clip(x0, lo + 1e-9, hi - 1e-9),
                                       bounds=(lo, hi), xtol=1e-6, ftol=1e-6,
                                       max_nfev=100)
    err = np.linalg.norm(self.markers(self.pose(sol.x, q_root_ref)) - P, axis=1)
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
  """Marker offsets in body frames from the anatomical world guesses."""
  m, d = mm.m, mm.d
  d.qpos[:] = m.qpos0
  d.qpos[:3] = [0, 0, 0.99]
  d.qpos[3:7] = [1, 0, 0, 0]
  mujoco.mj_kinematics(m, d)
  off = np.zeros((len(LABELS), 3))
  for k, l in enumerate(LABELS):
    R = d.xmat[mm.bid[k]].reshape(3, 3)
    off[k] = R.T @ (anat_world(l) - d.xpos[mm.bid[k]])
  return off


PELVIS = [LABELS.index(l) for l in ("LASI", "RASI", "LPSI", "RPSI")]


def root_from_pelvis(mm, P):
  """Root pose from a rigid (Kabsch) fit of the model pelvis markers."""
  m, d = mm.m, mm.d
  d.qpos[:] = m.qpos0
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


def pelvis_frame(P):
  """Root guess from pelvis markers: yaw + position (model faces -y, left +x)."""
  L = {l: P[LABELS.index(l)] for l in ("LASI", "RASI", "LPSI", "RPSI")}
  asis = (L["LASI"] + L["RASI"]) / 2
  psis = (L["LPSI"] + L["RPSI"]) / 2
  fwd = asis - psis
  fwd[2] = 0
  fwd /= np.linalg.norm(fwd)
  yaw = np.arctan2(fwd[1], fwd[0]) - np.arctan2(-1.0, 0.0)  # model faces -y
  q = sRot.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
  pos = (asis + psis) / 2 + np.array([0, 0, -0.03])
  return pos, q


# Joints held near 0 in the static (anatomical) standing trial, see calibrate(static=...)
STATIC_NEUTRAL = [f"{j}_{s}" for s in ("r", "l") for j in
                  ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle",
                   "subtalar_angle")]


# Segments whose marker placement the static trial decides (the posture prior holds
# for them); see calibrate(static_segments=...)
STATIC_SEGMENTS_LEGS = ("pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l", "calcn_r", "calcn_l")


def calibrate(mm, trials, n_frames=24, iters=5, prior=2.0, max_shift=0.06, seed=0,
              outlier=None, stand=None, flat=10.0, robust=None, static=None,
              static_share=0.5, static_w=1.0, static_segments=None):
  """Fit per-marker body offsets for one subject from several trials.
  robust: MarkerModel.solve (Cauchy marker loss instead of the outlier re-solve).
  max_shift: float or per-marker array [39] (m) bounding the move from the
  anatomical guess. outlier: see MarkerModel.solve; rejected samples are left
  out of the offset update. stand (optional): extra quiet-standing frames
  [k, 39, 3], solved with the flat-foot term (weight `flat`).

  static (optional, default off): frames [k, 39, 3] of the subject's static
  standing trial. They are solved with flat feet and the STATIC_NEUTRAL joints
  pulled toward 0 (weight static_w per rad), and carry static_share of the
  weight of every offset update. Without them the offsets and the pose are not
  separable: a knee marker a few cm further back on the femur fits as well as a
  knee bent 20 deg, and the dynamic frames alone settled on bent knees / tilted
  pelvis in some subjects (MMH Elevator 2026-10-15: quiet-standing knee flexion
  10-24 deg in 16 of 22 trials, confirmed against Xsens on S11). With the
  static trial: knee ~0 deg, residual unchanged or lower (S08, S09) to +2 mm
  (S01). static_segments (optional): only markers on these bodies take their
  placement from the static frames (e.g. STATIC_SEGMENTS_LEGS); with all markers,
  the hand markers were fitted to the loose hanging wrists of the static pose and
  their leave-one-out prediction got worse (finger 33 -> 48 mm)."""
  rng = np.random.default_rng(seed)
  m, d = mm.m, mm.d
  off0 = anat_offsets(mm)
  off = off0.copy()
  mm.set_offsets(off)
  # diverse frames: spread over all trials, require full pelvis markers
  frames = []
  for P in trials:
    idx = rng.choice(len(P), size=max(1, n_frames // len(trials)), replace=False)
    frames += [P[i] for i in idx if not np.isnan(P[i, 23:27]).any()]
  n_rand = len(frames)
  if stand is not None:
    frames += [P for P in stand if not np.isnan(P[23:27]).any()]
  n_dyn = len(frames)
  wt = np.ones(n_dyn)
  posture = None
  if static is not None:
    st = [P for P in static if not np.isnan(P[23:27]).any()]
    frames += st
    wt = np.r_[wt, np.full(len(st), static_share / (1 - static_share) * n_dyn / max(len(st), 1))]
    pidx = np.array([FREE.index(j) for j in STATIC_NEUTRAL])
    posture = (pidx, np.zeros(len(pidx)), static_w)
  use_static = np.ones(len(LABELS), bool)
  if static_segments is not None:
    use_static = np.array([MARKERS[l][0] in static_segments for l in LABELS])
  for it in range(iters):
    mm.set_offsets(off)
    poses, used = [], []
    for fi, P in enumerate(frames):
      fl = flat if fi >= n_rand else 0.0
      is_static = fi >= n_dyn
      pos, q = root_from_pelvis(mm, P)
      x0 = np.r_[pos, np.zeros(3), m.qpos0[mm.adr]]
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
      errs.append(np.nanmean(e))
    print(f"  calib iter {it}: mean marker err {1e3 * np.mean(errs):.1f} mm "
          f"(max shift {1e3 * np.linalg.norm(off - off0, axis=1).max():.0f} mm)")
  return off


def sole_stance(P, rate, v_max=0.1, dz=0.02, min_s=0.1):
  """Stance mask per sole marker {(side, heel|toe): [T] bool} from the markers:
  slow (< v_max m/s) and low (< its 5th-percentile height + dz), runs shorter
  than min_s dropped."""
  out = {}
  for side in ("l", "r"):
    for kind, lab in (("heel", "HEE"), ("toe", "TOE")):
      p = P[:, LABELS.index(side.upper() + lab)]
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


def ik_trial(mm, P, smooth=0.05, restarts=8, seed=0, rate=None, contact=False,
             v_joint=15.0, w_floor=10.0, w_anchor=3.0, grasp2h=None, w_grasp=0.3,
             ankle_smooth=0.3, outlier=None, mtp_prior=0.0, anchor_centroid=False,
             pelvis_gap="carry", robust=None, root_init="prev"):
  """Per-frame IK over a whole trial. Returns qpos [T, nq] and marker err [T, 39]
  (NaN where a marker is missing or was rejected, see MarkerModel.solve).
  anchor_centroid: lock each stance foot at the centroid of its anchored sole
  points instead of every point, so it may turn about it. MyoFullBody's knee
  has no axial rotation: in a pivot turn on a planted foot (MMH Elevator) the
  whole twist went into hip rotation, which sat on its -40 deg limit and
  snapped back when the foot lifted (500-860 deg/s).

  pelvis_gap: what a frame with a missing pelvis marker gets once the first
  frame is solved. "carry" (default, MMH / OMOMO) repeats the previous pose;
  "solve" solves it from the markers that are there (the root then rests on the
  thigh / trunk markers and the warm start). MMH Elevator: the carried box hides
  the front pelvis markers for 0.3-1.4 s per bend, and carrying froze the
  reference there, then jumped it 0.1-0.6 m in one frame (15 of 107 clips).
  Frames before the first full pelvis stay unsolved (zeros, marker_err NaN).

  root_init: warm start of the root in each frame. "prev" (default): the previous
  frame's solution, like the joints. "pelvis": the rigid fit of this frame's 4
  pelvis markers (root_from_pelvis) when all 4 are there, else "prev". Warm-
  starting the root from the previous frame let the fit slide into a wrong
  basin during a deep stoop and stay there (MMH Elevator IK v9: the root sank
  9 cm, the hips ran to their 120 deg limit, pelvis markers 16 cm off for up to
  0.4 s, then a 50 deg snap back; the same frames solved from scratch fit).

  grasp2h [T] bool (optional): two-handed carry frames -> palms face each other.
  contact=True (needs rate): sole points kept above the floor (the lowest sole
  point of the first, standing frame), sole points in marker-detected stance
  locked where they touched down (no skating), joint speed <= v_joint rad/s.
  """
  rng = np.random.default_rng(seed)
  m = mm.m
  T = len(P)
  Q = np.zeros((T, m.nq))
  E = np.full((T, len(LABELS)), np.nan)
  x_prev, q_ref = None, None
  stance = sole_stance(P, rate) if contact else None
  if contact:
    # stronger frame-to-frame smoothing where the IK chattered: ankle/subtalar
    # wobbled at the speed limit in deep squats; elv_angle (shoulder elevation
    # plane) is undefined with the arm hanging and flipped by pi between the
    # two equivalent fits -- so it is held only while shoulder_elv is small
    # (a constant 1.0 lagged the raised arm: elbow marker error 8.7 -> 14 mm;
    # 0.2 on every joint lagged fast clips)
    base = np.array([ankle_smooth if n.startswith(("ankle_angle", "subtalar")) else 0.05
                     for n in FREE])
    elv = [FREE.index(f"elv_angle_{sd}") for sd in ("r", "l")]
    selv = [FREE.index(f"shoulder_elv_{sd}") for sd in ("r", "l")]

    def smooth_at(x):
      w = base.copy()
      for e, se in zip(elv, selv):
        w[e] = 0.05 + 1.0 * np.exp(-(x[6 + se] / 0.35) ** 2)
      return w
  floor, anchors, last_t = None, {}, None

  def contact_args(t):
    if floor is None:
      return None
    for i, (side, kind) in enumerate(mm.fp_kind):
      if kind == "-" or not stance[(side, kind)][t]:
        anchors.pop(i, None)
    return dict(floor=floor, anchors=dict(anchors), w_floor=w_floor, w_grasp=w_grasp,
                centroid=anchor_centroid,
                grasp2h=grasp2h is not None and bool(grasp2h[t]),
                w_anchor=w_anchor, max_step=v_joint * (t - last_t) / rate)

  def touch_down(t):
    fp = mm.foot_points()  # pose of the frame just solved
    for i, (side, kind) in enumerate(mm.fp_kind):
      if kind != "-" and stance[(side, kind)][t] and i not in anchors:
        anchors[i] = np.r_[fp[i, :2], max(fp[i, 2], floor)]

  for t in range(T):
    if np.isnan(P[t, 23:27]).any() and (x_prev is None or pelvis_gap == "carry"):
      if x_prev is not None:  # no pelvis -> carry previous pose
        Q[t] = mm.pose(x_prev, q_ref)
      continue
    if x_prev is None:
      pos, q_ref = root_from_pelvis(mm, P[t])
      cands = []
      for k in range(restarts):
        xs = np.r_[pos, np.zeros(3), m.qpos0[mm.adr]]
        if k:
          xs[6:] = mm.lo + rng.random(len(mm.lo)) * (mm.hi - mm.lo)
        if mtp_prior:
          xs[6 + mm.mtp] = 0.0
        cands.append(mm.solve(P[t], xs, q_ref, outlier=outlier, mtp_prior=mtp_prior,
                              robust=robust))
      x, err = min(cands, key=lambda c: np.nanmean(c[1]))
    else:
      xs = x_prev.copy()
      if root_init == "pelvis" and not np.isnan(P[t, PELVIS, 0]).any():
        xs[:3], q_ref = root_from_pelvis(mm, P[t])
      else:  # re-centre the rotvec on the previous root orientation
        q_ref = mm.pose(x_prev, q_ref)[3:7]
      xs[3:6] = 0
      x, err = mm.solve(P[t], xs, q_ref, x_prev=xs,
                        smooth=smooth_at(xs) if contact else smooth,
                        contact=contact_args(t) if contact else None, outlier=outlier,
                        mtp_prior=mtp_prior, robust=robust)
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
