"""Per-segment scaling: the model scaled to each subject from its static trial (M2M_SCALE=segments).

By default (M2M_SCALE=uniform) the markers are scaled to the model by the ratio of the model's
standing height to the subject's (mocap2myo.ik.load_trial) and the model keeps its proportions.
With segment scaling the model is scaled to the subject instead, segment by segment, as OpenSim's
Scale tool does, and the markers stay in the subject's own size (the IK's "scale" is 1):

  factors   one per scaling group (GROUPS: canonical segments sharing their factors, the left and
            right sides together) and canonical axis (lateral, anterior-posterior, superior of the
            model at its neutral pose): marker distances in the static trial over the same
            distances between the model's marker guesses (the marker set's anatomical placement,
            mocap2myo.ik.anat_offsets) at the neutral pose, averaged over the pairs of MEASURE that
            the marker set has and the static trial saw (both sides). An axis no pair measures
            takes the geometric mean of the group's measured axes; a group none measures (the
            head and neck; a segment whose markers are missing) takes the uniform factor, subject
            height / model height.
  bodies    a body takes the group of its canonical segment ([segments] of the profile; a body in
            several, the first group in GROUPS); a body in none (lumbar vertebrae, talus, patella,
            shoulder helper bodies, fingers, path-point bodies) the group of the segment bodies
            right below it when they share one, else its parent's.
  the model everything a body carries in its own frame is mapped by A_b = R_b^T diag(f) R_b (R_b:
            the body's orientation in the canonical frame at the neutral pose; f its group's
            factors; A_b = f I for one factor): child body and joint positions, sites (the marker
            sites and the muscle attachment points), geoms (positions, sizes, mesh scales; meshes of
            a body scaled unequally along its axes are rebuilt from their vertices), the inertial
            frame and inertia (the mass distribution stretched with the body: second moment
            A S A^T at the same mass). Slide joints (range, reference, their coupling polynomials)
            take the parent's factor along their axis, as OpenSim scales a joint's translation
            functions by its parent. Muscle length ranges (MuJoCo's muscle L0 and tendon slack
            length) and tendon spring lengths change with the path length at the reference pose,
            as OpenSim's muscles after scaling. Orientations, joint axes and masses are not changed;
            mocap2myo.kinetics scales the masses to the subject's (its mass distribution kept).

A subject's factors are a small JSON file (Scaling.save / load; pipeline/solve_ik.py writes
<M2M_IK_DIR>/<subject>_scale.json); every stage rebuilds the scaled model from it
(ik.MarkerModel(scaling=...)), and write_mjcf exports it with the subject's markers.
"""

from __future__ import annotations

import json
import os

import mujoco
import numpy as np

AXES = ("lateral", "anterior", "superior")  # canonical x (left), y (back), z (up); factors are unsigned

# scaling groups, in order of precedence: canonical segments that share their factors
GROUPS = (
  ("pelvis", ("pelvis",)),
  ("torso", ("torso", "clavicle_l", "clavicle_r", "scapula_l", "scapula_r")),
  ("thigh", ("femur_l", "femur_r")),
  ("shank", ("tibia_l", "tibia_r")),
  ("foot", ("foot_l", "foot_r", "toes_l", "toes_r")),
  ("upper_arm", ("humerus_l", "humerus_r")),
  ("forearm", ("ulna_l", "ulna_r", "radius_l", "radius_r")),
  ("hand", ("hand_l", "hand_r")),
  ("head", ("neck", "head")),
)
GROUP_NAMES = tuple(g for g, _ in GROUPS)
UNIFORM = "uniform"  # a body in no group: the uniform factor

# group -> {axis: [(landmark, landmark), ...]}: the marker pairs that measure it, named by the
# markers' landmarks (mocap2myo.landmarks; "{s}" = either side), so that one table serves every
# marker set. "all": one factor for the three axes. Plug-in-Gait has the first pair of each;
# the ISB / Dumas set the others. A marker placed along a segment (a wand, markerset "along")
# measures nothing.
MEASURE = {
  "pelvis": {"lateral": [("asis_l", "asis_r")],
             "anterior": [("asis_{s}", "psis_{s}")]},
  "torso": {"superior": [("c7", "psis_{s}")],
            "lateral": [("acromion_l", "acromion_r"), ("acromion_tip_l", "acromion_tip_r")]},
  "thigh": {"all": [("asis_{s}", "femur_epi_lat_{s}"), ("asis_{s}", "femur_epi_med_{s}")]},
  "shank": {"all": [("femur_epi_lat_{s}", "malleolus_lat_{s}"), ("femur_epi_med_{s}", "malleolus_med_{s}")]},
  "foot": {"all": [("heel_{s}", "mt2_head_{s}"), ("heel_{s}", "mt1_head_{s}"), ("heel_{s}", "mt5_head_{s}")]},
  "upper_arm": {"all": [("acromion_{s}", "epicondyle_lat_{s}"), ("acromion_tip_{s}", "epicondyle_lat_{s}"),
                        ("acromion_tip_{s}", "epicondyle_med_{s}")]},
  "forearm": {"all": [("epicondyle_lat_{s}", "styloid_rad_{s}"), ("epicondyle_med_{s}", "styloid_uln_{s}")]},
  "hand": {"all": [("styloid_rad_{s}", "mc3_head_{s}"), ("styloid_uln_{s}", "mc3_head_{s}"),
                   ("styloid_rad_{s}", "mc2_head_{s}"), ("styloid_uln_{s}", "mc5_head_{s}")]},
  "head": {},  # uniform
}


# --- factors ------------------------------------------------------------------------

def measure(mm, P, height_m, keep_uniform=()):
  """Scale factors of one subject (a dict, see Scaling). mm: an unscaled ik.MarkerModel (its
  marker guesses are set here); P [k, n, 3]: static standing frames in the subject's own size
  (metres, the marker set's order); height_m: the subject's height. keep_uniform: groups left
  at the uniform factor."""
  from mocap2myo import ik
  off = ik.anat_offsets(mm)  # switches off markers whose landmark the model lacks
  mm.set_offsets(off)
  M = mm.markers(mm.qref)
  seen = ~np.isnan(P[..., 0]).all(0)  # markers the static frames saw
  S = np.full(P.shape[1:], np.nan)
  S[seen] = np.nanmedian(P[:, seen], axis=0)
  by_lm = {}  # landmark -> marker index
  for k, l in enumerate(mm.ms.LABELS):
    mk = mm.ms.markers[l]
    if mk.along is None and mm.active[k]:
      by_lm.setdefault(mk.landmark, k)
  uniform = float(height_m) / mm.model_height
  factors, how, pairs = {}, {}, []
  for g in GROUP_NAMES:
    meas = {} if g in keep_uniform else MEASURE.get(g, {})
    got = {}
    for ax, pp in meas.items():
      r = []
      for a, b in pp:
        for s in (("l", "r") if "{s}" in a + b else ("",)):
          la, lb = a.format(s=s), b.format(s=s)
          if la not in by_lm or lb not in by_lm:
            continue
          i, j = by_lm[la], by_lm[lb]
          ds, dm = float(np.linalg.norm(S[i] - S[j])), float(np.linalg.norm(M[i] - M[j]))
          if not np.isfinite(ds):  # not seen in the static frames
            continue
          r.append(ds / dm)
          pairs.append(dict(group=g, axis=ax, markers=[mm.ms.LABELS[i], mm.ms.LABELS[j]],
                            subject_m=round(ds, 4), model_m=round(dm, 4), ratio=round(ds / dm, 4)))
      if r:
        got[ax] = float(np.mean(r))
    if "all" in got:
      f = [got["all"]] * 3
      how[g] = "measured (one factor)"
    elif got:
      gm = float(np.exp(np.mean(np.log(list(got.values())))))
      f = [got.get(ax, gm) for ax in AXES]
      how[g] = {ax: "measured" if ax in got else "geometric mean of the measured axes" for ax in AXES}
    else:
      f = [uniform] * 3
      how[g] = ("uniform (kept)" if g in keep_uniform else
                "uniform (no marker pair)" if not MEASURE.get(g) else
                "uniform (its markers are missing or not on this model)")
    factors[g] = [round(float(x), 6) for x in f]
  return dict(mode="segments", model_profile=mm.profile.name, marker_set=mm.ms.key,
              height_m=float(height_m), model_height_m=mm.model_height, uniform=round(uniform, 6),
              factors=factors, how=how, pairs=pairs, axes=list(AXES))


class Scaling:
  """A subject's scale factors (measure) and, once applied to a model's MjSpec (apply), the
  per-body matrices A [nbody, 3, 3] (each body's frame contents x A[b]) and groups."""

  def __init__(self, data):
    self.data = data
    self.factors = {g: np.asarray(f, float) for g, f in data["factors"].items()}
    self.uniform = float(data["uniform"])
    self.A = None
    self.groups = None

  @classmethod
  def load(cls, path):
    with open(path) as f:
      return cls(json.load(f))

  def save(self, path):
    tmp = path + ".part"
    with open(tmp, "w") as f:
      json.dump(self.data, f, indent=1)
    os.replace(tmp, path)

  def factor(self, group):
    """The (lateral, anterior, superior) factors of a group (UNIFORM: the uniform one)."""
    if group == UNIFORM:
      return np.full(3, self.uniform)
    return self.factors[group]

  def apply(self, spec, profile):
    """Scale the model in `spec` (profile.spec(), not yet changed) in place; sets self.A and
    self.groups. Returns the scaled MjModel (marker sites added later need another compile)."""
    m0 = spec.compile()
    self.groups = body_groups(profile, m0)
    self.A, world = body_matrices(profile, m0, self, self.groups)
    return scale_spec(spec, profile, m0, self.A, world)


# --- bodies -------------------------------------------------------------------------

def body_groups(profile, m):
  """Scaling group of every body (None: not part of the model, e.g. scenery on the world;
  UNIFORM: the uniform factor), by the rule in the module docstring."""
  seg_group = {s: g for g, segs in GROUPS for s in segs}
  own = {}
  for s, sg in profile.segments.items():
    g = seg_group[s]
    for bn in sg.bodies:
      b = m.body(bn).id
      if b not in own or GROUP_NAMES.index(g) < GROUP_NAMES.index(own[b]):
        own[b] = g
  kids = [[] for _ in range(m.nbody)]
  for b in range(1, m.nbody):
    kids[m.body_parentid[b]].append(b)

  def below(b):  # groups of the segment bodies right below b
    out = set()
    for c in kids[b]:
      out |= {own[c]} if c in own else below(c)
    return out
  root = int(m.jnt_bodyid[m.joint(profile.root_joint).id])
  out = [None] * m.nbody
  todo = [(root, UNIFORM)]
  while todo:
    b, up = todo.pop()
    if b in own:
      out[b] = own[b]
    else:
      gs = below(b)
      out[b] = gs.pop() if len(gs) == 1 else up
    todo += [(c, out[b]) for c in kids[b]]
  return out


def body_matrices(profile, m, sc, groups):
  """A [nbody, 3, 3]: each body's scaling in its own frame, and S [nbody, 3, 3] the same in
  the canonical world frame at the neutral pose (diag of its factors, identity off the model)."""
  from mocap2myo import landmarks as L
  nt = L.Neutral(profile, "floor", m)
  A = np.tile(np.eye(3), (m.nbody, 1, 1))
  S = A.copy()
  for b, g in enumerate(groups):
    if g is None:
      continue
    f = sc.factor(g)
    S[b] = np.diag(f)
    if f[0] == f[1] == f[2]:
      A[b] = f[0] * np.eye(3)
    else:
      R = nt.d.xmat[b].reshape(3, 3)
      A[b] = R.T @ np.diag(f) @ R
  return A, S


# --- the model ------------------------------------------------------------------------

def _mat(q):
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, np.asarray(q, float))
  return R.reshape(3, 3)


def _quat(R):
  q = np.zeros(4)
  mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
  return q


def _frame_rot(fr):
  """Rotation of an element's frame (mjsFrame chain) relative to its body; identity for None."""
  R = np.eye(3)
  while fr is not None:
    if fr.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT:
      raise SystemExit(f"scaling: frame {fr.name!r} has an orientation other than a quaternion "
                       "(not supported)")
    R = _mat(fr.quat) @ R
    fr = fr.frame
  return R


def _in_frame(A, fr):
  """A body's matrix A expressed in the coordinates of a frame in that body."""
  if fr is None:
    return A
  R = _frame_rot(fr)
  return R.T @ A @ R


def _scaled(v, A):
  return (A @ np.asarray(v, float)).tolist()


def _iso(A):
  """The factor of an isotropic matrix f I, else None."""
  f = A[0, 0]
  return float(f) if np.allclose(A, f * np.eye(3), rtol=0, atol=1e-12) else None


def _geom_stretch(A, Rg):
  """How much A stretches each axis of a geom with orientation Rg (in the body frame)."""
  return np.linalg.norm(A @ Rg, axis=0)


def scale_spec(spec, profile, m0, A, S):
  """Apply the per-body matrices A (body frames) to spec, whose compiled model is m0; S: the
  same in the canonical world frame (slide joints). Returns the compiled scaled model."""
  T = mujoco.mjtGeom
  for kind in ("skins", "flexes"):
    if len(getattr(spec, kind, [])):
      raise SystemExit(f"scaling: the model has {kind} (not supported)")
  bodies = list(spec.bodies)
  # bodies: position in the parent, inertial frame and inertia
  for i, b in enumerate(bodies):
    if i == 0:
      continue
    p = m0.body_parentid[i]
    b.pos = _scaled(b.pos, _in_frame(A[p], b.frame))
    if np.allclose(A[i], np.eye(3)):
      continue
    R = _mat(m0.body_iquat[i])
    I = R @ np.diag(m0.body_inertia[i]) @ R.T
    Sg = 0.5 * np.trace(I) * np.eye(3) - I  # second moment of the mass about its centre
    s, V = np.linalg.eigh(A[i] @ Sg @ A[i].T)
    s = np.maximum(s, 0.0)  # a flat body (inertia on the triangle bound) stays exactly on it
    if np.linalg.det(V) < 0:
      V[:, 0] = -V[:, 0]
    b.explicitinertial = True
    b.mass = float(m0.body_mass[i])
    b.ipos = _scaled(m0.body_ipos[i], A[i])
    b.inertia = [s[1] + s[2], s[0] + s[2], s[0] + s[1]]  # principal moments, axes V
    b.iquat = _quat(V).tolist()
    b.ialt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    b.fullinertia = [np.nan, 0, 0, 0, 0, 0]
  # frames: position in their body (or outer frame)
  for fr in spec.frames:
    fr.pos = _scaled(fr.pos, _in_frame(A[fr.parent.id], fr.frame))
  # geoms
  mesh_users = {}
  for g in spec.geoms:
    if g.type == T.mjGEOM_MESH:
      mesh_users.setdefault(g.meshname, []).append(g)
  for g in spec.geoms:
    i = g.id
    b = int(m0.geom_bodyid[i])
    Ab = A[b]
    if np.allclose(Ab, np.eye(3)):
      continue
    Al = _in_frame(Ab, g.frame)
    f = _iso(Ab)
    g.pos = _scaled(g.pos, Al)
    if not np.isnan(g.fromto[0]):
      g.fromto = _scaled(g.fromto[:3], Al) + _scaled(g.fromto[3:], Al)
    if g.type == T.mjGEOM_MESH:
      continue  # below
    d = np.full(3, f) if f is not None else _geom_stretch(Ab, _mat(m0.geom_quat[i]))
    sz = np.array(g.size, float)
    if g.type == T.mjGEOM_SPHERE:
      sz[0] *= np.cbrt(d.prod())
    elif g.type in (T.mjGEOM_CAPSULE, T.mjGEOM_CYLINDER):
      sz[0] *= np.sqrt(d[0] * d[1])
      sz[1] *= d[2]
    elif g.type in (T.mjGEOM_ELLIPSOID, T.mjGEOM_BOX):
      sz[:3] *= d
    elif g.type == T.mjGEOM_PLANE:
      sz[:2] *= d[:2]
    else:
      raise SystemExit(f"scaling: geom {g.name!r} of type {g.type} (not supported)")
    g.size = sz.tolist()
  # meshes: a factor for every body -> the mesh's scale; else rebuilt from its vertices
  for name, users in mesh_users.items():
    Ms = [A[int(m0.geom_bodyid[g.id])] for g in users]
    if all(np.allclose(M, np.eye(3)) for M in Ms):
      continue
    fs = [_iso(M) for M in Ms]
    mesh = spec.mesh(name)
    if all(f is not None for f in fs) and np.allclose(fs, fs[0]) and not np.any(mesh.refpos):
      mesh.scale = (np.asarray(mesh.scale) * fs[0]).tolist()
      continue
    for k, g in enumerate(users):  # each user its own copy, in its body's frame
      if g.frame is not None:
        raise SystemExit(f"scaling: mesh geom {g.name!r} in a frame, scaled unequally along its "
                         "axes (not supported)")
      i = g.id
      mid = int(m0.geom_dataid[i])
      V = m0.mesh_vert[m0.mesh_vertadr[mid]:m0.mesh_vertadr[mid] + m0.mesh_vertnum[mid]].astype(float)
      F = m0.mesh_face[m0.mesh_faceadr[mid]:m0.mesh_faceadr[mid] + m0.mesh_facenum[mid]]
      Vb = (m0.geom_pos[i] + V @ _mat(m0.geom_quat[i]).T) @ Ms[k].T  # body frame, scaled
      new = name if len(users) == 1 else f"{name}_{k}"
      if len(users) == 1:
        spec.delete(mesh)
      spec.add_mesh(name=new, uservert=Vb.ravel().tolist(), userface=F.ravel().tolist())
      g.meshname = new
      g.pos = [0.0, 0.0, 0.0]
      g.quat = [1.0, 0.0, 0.0, 0.0]
      g.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
      g.fromto = [np.nan] * 6
    if len(users) > 1:
      spec.delete(mesh)
  # sites, cameras, lights: positions
  for kind in ("sites", "cameras", "lights"):
    for e in getattr(spec, kind):
      b = int({"sites": m0.site_bodyid, "cameras": m0.cam_bodyid, "lights": m0.light_bodyid}[kind][e.id])
      if not np.allclose(A[b], np.eye(3)):
        e.pos = _scaled(e.pos, _in_frame(A[b], e.frame))
  # joints: positions; slide joints by the parent's factor along the axis (world, neutral pose)
  from mocap2myo import landmarks as L
  nt = L.Neutral(profile, "floor", m0)
  jf = np.ones(m0.njnt)
  for j in spec.joints:
    i = j.id
    b = int(m0.jnt_bodyid[i])
    if not np.allclose(A[b], np.eye(3)):
      j.pos = _scaled(j.pos, _in_frame(A[b], j.frame))
    if j.type == mujoco.mjtJoint.mjJNT_SLIDE:
      p = int(m0.body_parentid[b])
      jf[i] = float(np.linalg.norm(S[p] @ nt.d.xaxis[i]))
      j.range = (np.asarray(j.range) * jf[i]).tolist()
      j.ref *= jf[i]
      j.springref *= jf[i]
      j.margin *= jf[i]
  # joint couplings q1 - q1_0 = sum c_k (q2 - q2_0)^k: lengths change with their joints' factors
  for e in spec.equalities:
    if e.type != mujoco.mjtEq.mjEQ_JOINT:
      raise SystemExit(f"scaling: equality {e.name!r} of type {e.type} (only joint couplings are supported)")
    f1 = jf[m0.joint(e.name1).id]
    f2 = jf[m0.joint(e.name2).id] if e.name2 else 1.0
    c = np.array(e.data, float)
    c[:5] = f1 * c[:5] / f2 ** np.arange(5)
    e.data = c.tolist()
  for k in spec.keys:
    if len(k.qpos) == m0.nq:
      q = np.array(k.qpos, float)
      for i in np.flatnonzero(m0.jnt_type == mujoco.mjtJoint.mjJNT_SLIDE):
        q[m0.jnt_qposadr[i]] *= jf[i]
      k.qpos = q.tolist()
  # fixed tendons over slide joints would need their coefficients scaled
  for t in range(m0.ntendon):
    for w in range(m0.tendon_adr[t], m0.tendon_adr[t] + m0.tendon_num[t]):
      if m0.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT and jf[m0.wrap_objid[w]] != 1.0:
        raise SystemExit(f"scaling: tendon {m0.tendon(t).name!r} couples a slide joint (not supported)")
  # path lengths at the reference pose, before / after: muscle length ranges, spring lengths
  m1 = spec.compile()
  r_ten, r_act = _length_ratios(profile, m0, m1)
  for t in spec.tendons:
    r = r_ten[t.id]
    if r == 1.0:
      continue
    sl = np.asarray(t.springlength, float)
    if np.any(sl > 0):
      t.springlength = (sl * r).tolist()
    if np.any(np.asarray(t.range) != 0):
      t.range = (np.asarray(t.range) * r).tolist()
  for a in spec.actuators:
    lr = np.asarray(a.lengthrange, float)
    if lr[1] > lr[0]:
      i = a.id
      r = r_act[i] if m0.actuator_trntype[i] in (mujoco.mjtTrn.mjTRN_TENDON,) else (
        jf[m0.actuator_trnid[i][0]] if m0.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT else 1.0)
      a.lengthrange = (lr * r).tolist()
  return spec.compile()


def _length_ratios(profile, m0, m1):
  """Tendon and actuator lengths at the reference pose, scaled over unscaled (1 where the
  unscaled length is 0)."""
  out = []
  for m in (m0, m1):
    d = mujoco.MjData(m)
    d.qpos[:] = profile.reference_qpos(m)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_tendon(m, d)
    mujoco.mj_transmission(m, d)
    out.append((d.ten_length.copy(), d.actuator_length.copy()))
  (t0, a0), (t1, a1) = out
  with np.errstate(all="ignore"):
    rt = np.where(np.abs(t0) > 1e-9, t1 / t0, 1.0)
    ra = np.where(np.abs(a0) > 1e-9, a1 / a0, 1.0)
  return rt, ra


def write_mjcf(profile, sc, path, markers=None):
  """The scaled model as one MJCF file (mesh paths absolute), with the subject's markers as
  sites mk_<label> if given: markers = (marker set, offsets [n, 3] in the body frames, active)."""
  spec = profile.spec()
  sc.apply(spec, profile)
  if markers is not None:
    ms, off, active = markers
    for l, o, a in zip(ms.LABELS, off, active):
      body = profile.segment_body(ms.segment(l))
      if a and body is not None:
        spec.body(body).add_site(name=f"mk_{l}", pos=np.asarray(o, float).tolist(),
                                 size=[0.01, 0, 0], rgba=[0.9, 0.2, 0.2, 1])
  base = profile.xml.parent
  for attr in ("meshdir", "texturedir"):
    v = getattr(spec, attr)
    setattr(spec, attr, str((base / v).resolve()) if v and not os.path.isabs(v) else (v or str(base)))
  spec.compile()
  tmp = path + ".part"
  with open(tmp, "w") as f:
    f.write(spec.to_xml())
  os.replace(tmp, path)
