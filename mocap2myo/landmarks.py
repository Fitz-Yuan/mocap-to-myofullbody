"""Anatomical landmarks of a musculoskeletal model, computed from its own geometry.

For a model profile (mocap2myo.profile) the model is put in its neutral standing pose in
the canonical frame (z up, facing -y, left = +x; feet on z = 0 and the hip centres'
midpoint on x = y = 0 with place="floor", or the root where the profile puts it with
place="root"), and each landmark is found as

  joint centres   the anchor of a joint or the origin of a body ([joint_centres]);
  surface points  the extreme vertex of a segment's bone meshes in an anatomical
                  direction, within a region set by the joint centres (e.g. ASIS = most
                  anterior pelvis vertex lateral to 0.6 x the hip half-width and 3 cm
                  above the hip centres). Bodies without meshes (primitive-geom models)
                  use points sampled on their solid geoms.

Thresholds are for a 1.72 m model and scale with the model's bone height. A profile can
narrow the geometry of a landmark ([landmarks] <name> = {geoms = [globs], bodies = [..]};
"{s}" in a glob is the side letter), set the search direction of the styloids of a
one-body forearm (dir = "anterior" / "posterior" / "lateral" / "medial" / ...), place the
3rd metacarpal head of a one-mesh hand (fraction = 0.5 of the way wrist -> fingertips),
turn off a height band (band = false, for c7 / t10 when the geoms are one vertebra), or give
the point outright (point = {body = "..", pos = [x, y, z]} in that body's frame).

Every landmark comes with the body it moves with and its position in that body's frame
(at the neutral pose), which is what a marker set needs: marker = body point of a
landmark + the marker's offset from it.

Landmarks (LANDMARKS; _l / _r: left / right):
  joint centres     hip, knee, ankle, mtp, shoulder (glenohumeral), elbow, wrist; any
                    other [joint_centres] entry (e.g. lumbar)
  femur_epi_lat/med femoral epicondyles: extremes of the distal femur along the knee axis
  malleolus_lat/med extremes of the shank (tibia, fibula) at the ankle-centre height
  heel, toe         most posterior calcaneus / most anterior toe vertex
  asis, psis        most anterior pelvis / most posterior ilium vertex above the hips
  sacrum            most posterior midline pelvis vertex at the PSIS height
  c7, t10           spinous processes: most posterior midline vertex 1.5-10 cm above the
                    jugular notch / at the xiphoid height (or of a named vertebra mesh)
  jugular, xiphoid  top / bottom of the sternum (midline, within 7 / 3 cm of its front)
  head_top, forehead, back_head   head top; support points 45 deg from the midline,
                    anterior / posterior, 5-10 cm below the head top (head-band level)
  acromion          highest scapula (or clavicle, or trunk) vertex lateral of the shoulder
  epicondyle_lat/med humeral epicondyles: extremes of the distal humerus along the elbow axis
  styloid_rad/uln   most distal radius / ulna vertex along the forearm
  mc3_head          most distal vertex of the 3rd metacarpal along wrist -> hand
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from mocap2myo.profile import Profile, geom_matches

REF_BONE_HEIGHT = 1.689   # MyoFullBody, head top to the lowest foot-bone point
GH_FRAC = 0.8715          # MyoFullBody: (glenohumeral centre - foot-bone sole) / bone height
DIRS = {"anterior": (0, -1, 0), "posterior": (0, 1, 0), "superior": (0, 0, 1),
        "inferior": (0, 0, -1), "left": (1, 0, 0), "right": (-1, 0, 0)}
SIDE_SIGN = {"l": 1.0, "r": -1.0}

LANDMARKS = (
  [f"{j}_{s}" for j in ("hip", "knee", "ankle", "mtp") for s in "lr"]
  + [f"{j}_{s}" for j in ("femur_epi_lat", "femur_epi_med", "malleolus_lat", "malleolus_med",
                          "heel", "toe") for s in "lr"]
  + [f"{j}_{s}" for j in ("asis", "psis") for s in "lr"] + ["sacrum"]
  + ["c7", "t10", "jugular", "xiphoid", "head_top"]
  + [f"{j}_{s}" for j in ("forehead", "back_head") for s in "lr"]
  + [f"{j}_{s}" for j in ("shoulder", "acromion", "elbow", "epicondyle_lat", "epicondyle_med",
                          "wrist", "styloid_rad", "styloid_uln", "mc3_head") for s in "lr"])
# further landmarks for skin-marker sets (e.g. the ISB / Dumas set of Moissenet & Schreiber 2019)
LANDMARKS_EXTRA = [f"{j}_{s}" for j in (
  "trochanter", "fibula_head", "tibial_tuberosity", "mt1_head", "mt2_head", "mt5_head",
  "scap_inf_angle", "scap_spine_root", "acromial_angle", "acromion_tip", "olecranon",
  "mc2_head", "mc5_head") for s in "lr"]
JOINT_CENTRES = ("hip", "knee", "ankle", "mtp", "shoulder", "elbow", "wrist")


@dataclass
class Landmark:
  name: str
  pos: np.ndarray            # canonical frame, neutral pose
  body: str                  # body it moves with
  segment: str | None        # canonical segment of that body
  local: np.ndarray          # position in that body's frame
  how: str                   # rule that produced it


@dataclass
class Result:
  profile: str
  landmarks: dict[str, Landmark]
  sole_z: float              # lowest foot-bone point
  head_top_z: float | None
  bone_height: float | None  # head top - foot-bone sole
  model_height: float | None  # bone height + profile soft tissue (MyoFullBody: ik.MODEL_HEIGHT)
  qpos: np.ndarray           # the neutral pose used
  missing: dict[str, str] = field(default_factory=dict)  # landmark -> why not found


def _sphere_dirs(n=162):
  i = np.arange(n) + 0.5
  phi = np.arccos(1 - 2 * i / n)
  th = np.pi * (1 + 5 ** 0.5) * i
  return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_SPH = _sphere_dirs()


def primitive_points(gtype, size) -> np.ndarray:
  """Points on the surface of a primitive geom, in its frame."""
  T = mujoco.mjtGeom
  if gtype == T.mjGEOM_SPHERE:
    return _SPH * size[0]
  if gtype == T.mjGEOM_ELLIPSOID:
    return _SPH * size[:3]
  if gtype == T.mjGEOM_CAPSULE:
    r, h = size[0], size[1]
    top = _SPH[_SPH[:, 2] >= 0] * r + [0, 0, h]
    bot = _SPH[_SPH[:, 2] <= 0] * r - [0, 0, h]
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    z = np.linspace(-h, h, 9)
    A, Z = np.meshgrid(a, z)
    side = np.stack([r * np.cos(A).ravel(), r * np.sin(A).ravel(), Z.ravel()], 1)
    return np.concatenate([top, bot, side])
  if gtype == T.mjGEOM_CYLINDER:
    r, h = size[0], size[1]
    a = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    z = np.linspace(-h, h, 9)
    A, Z = np.meshgrid(a, z)
    return np.stack([r * np.cos(A).ravel(), r * np.sin(A).ravel(), Z.ravel()], 1)
  if gtype == T.mjGEOM_BOX:
    g = np.linspace(-1, 1, 7)
    pts = []
    for ax in range(3):
      o = [i for i in range(3) if i != ax]
      U, V = np.meshgrid(g, g)
      for sgn in (-1, 1):
        p = np.zeros((U.size, 3))
        p[:, ax] = sgn
        p[:, o[0]], p[:, o[1]] = U.ravel(), V.ravel()
        pts.append(p)
    return np.unique(np.concatenate(pts), axis=0) * size[:3]
  return np.zeros((0, 3))


class Neutral:
  """A profile's model at its neutral pose in the canonical frame, with its geometry."""

  def __init__(self, profile: Profile, place: str = "floor", m: mujoco.MjModel | None = None):
    self.p = profile
    self.m = m if m is not None else profile.compile()
    self.d = mujoco.MjData(self.m)
    q = profile.neutral_qpos(self.m)
    self.set(q)
    if place == "floor" or (place == "root" and "root_pos" not in profile.neutral_cfg):
      sole = self.foot_points()[:, 2].min()
      hips = [self.joint_centre(f"hip_{s}")[0] for s in "lr" if f"hip_{s}" in profile.joint_centres]
      mid = np.mean(hips, 0) if hips else np.zeros(3)
      q = q.copy()
      q[:3] -= [mid[0], mid[1], sole]
      self.set(q)
    self.qpos = q

  def set(self, q):
    self.d.qpos[:] = q
    mujoco.mj_kinematics(self.m, self.d)
    mujoco.mj_comPos(self.m, self.d)

  # --- geometry ------------------------------------------------------------
  def body_geoms(self, b: int, globs=(), exclude=(), meshes: bool | None = None) -> list[int]:
    """Bone geoms of a body: its mesh geoms, or (meshes=False: a primitive-geom model) its
    solid primitives (not wrapping geoms, group 3; not planes). meshes=None: meshes if
    the body has any."""
    m = self.m
    gs = list(range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
    mesh = [g for g in gs if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if meshes is None:
      meshes = bool(mesh)
    if meshes:
      gs = mesh
    else:
      gs = [g for g in gs if m.geom_group[g] != 3 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    if globs:
      gs = [g for g in gs if geom_matches(m, g, globs)]
    if exclude:
      gs = [g for g in gs if not geom_matches(m, g, exclude)]
    return gs

  def geom_points(self, g: int) -> np.ndarray:
    m, d = self.m, self.d
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mid = m.geom_dataid[g]
      v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    else:
      v = primitive_points(m.geom_type[g], m.geom_size[g])
    return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T

  def points(self, bodies, globs=(), exclude=()) -> tuple[np.ndarray, np.ndarray]:
    """World points [n, 3] of these bodies' bone geometry and the body of each point."""
    P, B = [np.zeros((0, 3))], [np.zeros(0, int)]
    ids = [bn if isinstance(bn, (int, np.integer)) else self.m.body(bn).id for bn in bodies]
    # meshes when any of the bodies has one (a mesh-less helper body welded to a bone, e.g.
    # MyoFullBody's thorax, must not add its collision primitives)
    meshes = any(self.body_geoms(b, meshes=True) for b in ids)
    for b in ids:
      for g in self.body_geoms(b, globs, exclude, meshes):
        p = self.geom_points(g)
        P.append(p)
        B.append(np.full(len(p), b))
    return np.concatenate(P), np.concatenate(B)

  def segment_points(self, names, side=None, hint=None):
    """Points of canonical segments (those the profile has), or of a hint's bodies /
    geoms. Returns (P, B) or None when there is no geometry."""
    hint = hint or {}
    sub = (lambda s: s.replace("{s}", side)) if side else (lambda s: s)
    globs = [sub(g) for g in hint.get("geoms", [])]
    excl = [sub(g) for g in hint.get("exclude_geoms", [])]
    if "bodies" in hint:
      P, B = self.points([sub(b) for b in hint["bodies"]], globs, excl)
      return (P, B) if len(P) else None
    Ps, Bs = [], []
    for n in names:
      seg = self.p.segments.get(n)
      if seg is None:
        continue
      gl = globs or list(seg.geoms)
      if globs:  # a hint's globs may name geoms of any body of the segments
        P, B = self.points(seg.bodies, gl, excl or seg.exclude)
      else:
        P, B = self.points(seg.bodies, gl, seg.exclude)
      Ps.append(P)
      Bs.append(B)
    if not Ps or not sum(len(p) for p in Ps):
      return None
    return np.concatenate(Ps), np.concatenate(Bs)

  def foot_points(self):
    """Bone points of the feet (for the floor and the height)."""
    names = [n for n in ("foot_l", "toes_l", "foot_r", "toes_r") if n in self.p.segments]
    r = self.segment_points(names)
    if r is None:
      raise ValueError(f"{self.p.name}: no foot geometry (segments foot_l / foot_r)")
    return r[0]

  # --- points ----------------------------------------------------------------
  def joint_centre(self, name) -> tuple[np.ndarray, int, str]:
    spec = self.p.joint_centres[name]
    kind, ref = spec.split(":", 1)
    m, d = self.m, self.d
    if kind == "jnt":
      j = m.joint(ref).id
      return d.xanchor[j].copy(), int(m.jnt_bodyid[j]), f"anchor of joint {ref}"
    if kind == "body":
      b = m.body(ref).id
      return d.xpos[b].copy(), b, f"origin of body {ref}"
    raise ValueError(f"{self.p.name}: joint centre {name} = {spec!r}: expected jnt: or body:")

  def joint_axis(self, name) -> np.ndarray | None:
    spec = self.p.joint_centres.get(name, "")
    if spec.startswith("jnt:"):
      return self.d.xaxis[self.m.joint(spec[4:]).id].copy()
    return None

  def landmark(self, name, pos, body, how) -> Landmark:
    R = self.d.xmat[body].reshape(3, 3)
    segs = [s.name for s in self.p.segments.values() if self.m.body(body).name in s.bodies]
    return Landmark(name, np.asarray(pos, float).copy(), self.m.body(body).name,
                    "/".join(segs) or None, R.T @ (pos - self.d.xpos[body]), how)


def _pick(P, B, score, mask=None):
  idx = np.arange(len(P)) if mask is None else np.flatnonzero(mask)
  if not len(idx):
    return None
  i = idx[np.argmax(score[idx])]
  return P[i], int(B[i])


def compute(profile: Profile | str, place: str = "floor", m: mujoco.MjModel | None = None,
            neutral: Neutral | None = None) -> Result:
  """All landmarks of a profile's model (those its segments allow)."""
  if isinstance(profile, str):
    from mocap2myo.profile import load
    profile = load(profile)
  nt = neutral or Neutral(profile, place, m)
  p = profile
  H = p.landmark_hints
  out: dict[str, Landmark] = {}
  missing: dict[str, str] = {}

  def hint(name):
    base = name[:-2] if name[-2:] in ("_l", "_r") else name
    return H.get(name) or H.get(base) or {}

  def put(name, r, how):
    if r is None:
      missing[name] = how
      return
    out[name] = nt.landmark(name, r[0], r[1], how)

  def explicit(name, side):
    h = hint(name)
    if "point" in h:
      bn = h["point"]["body"].replace("{s}", side or "")
      b = nt.m.body(bn).id
      R = nt.d.xmat[b].reshape(3, 3)
      out[name] = nt.landmark(name, nt.d.xpos[b] + R @ np.asarray(h["point"]["pos"], float), b,
                              "explicit point (profile)")
      return True
    return False

  # joint centres (also any extra ones the profile names, e.g. "lumbar")
  for n in p.joint_centres:
    pos, b, how = nt.joint_centre(n)
    out[n] = nt.landmark(n, pos, b, how)
  # feet, height, scale
  F = nt.segment_points([n for n in ("foot_l", "toes_l", "foot_r", "toes_r") if n in p.segments])
  sole_z = float(F[0][:, 2].min())
  headg = nt.segment_points(["head"], hint=hint("head_top"))
  head_top = None
  if headg is not None:
    r = _pick(*headg, headg[0][:, 2])
    put("head_top", r, "highest head vertex")
    head_top = float(r[0][2])
  bone_h = head_top - sole_z if head_top is not None else None
  if bone_h:
    s = bone_h / REF_BONE_HEIGHT
  elif "hip_l" in out:
    s = (out["hip_l"].pos[2] - sole_z) / 0.915  # MyoFullBody hip centre above the bone sole
  else:
    s = 1.0
  x0 = np.mean([out[f"hip_{d}"].pos[0] for d in "lr"]) if "hip_l" in out and "hip_r" in out else 0.0
  w = abs(out["hip_l"].pos[0] - out["hip_r"].pos[0]) / 2 if "hip_l" in out and "hip_r" in out else 0.08 * s
  zh = np.mean([out[f"hip_{d}"].pos[2] for d in "lr" if f"hip_{d}" in out]) if "hip_l" in out else None
  if "shoulder_l" in out and "shoulder_r" in out:
    zsh = (out["shoulder_l"].pos[2] + out["shoulder_r"].pos[2]) / 2
  else:
    zsh = sole_z + GH_FRAC * (bone_h if bone_h else REF_BONE_HEIGHT * s)

  for sd in "lr":
    sg = SIDE_SIGN[sd]
    # heel / toe tip
    for name, segs, score, how in (
        (f"heel_{sd}", [f"foot_{sd}"], lambda P: P[:, 1], "most posterior calcaneus vertex"),
        (f"toe_{sd}", [f"toes_{sd}"] if f"toes_{sd}" in p.segments else [f"foot_{sd}"],
         lambda P: -P[:, 1], "most anterior toe vertex")):
      if explicit(name, sd):
        continue
      g = nt.segment_points(segs, sd, hint(name))
      put(name, None if g is None else _pick(*g, score(g[0])), how)

  # femoral epicondyles (along the knee axis) and malleoli (along the canonical lateral)
  for sd in "lr":
    sg = SIDE_SIGN[sd]
    latv = np.array([sg, 0.0, 0.0])
    hp, kn, an = out.get(f"hip_{sd}"), out.get(f"knee_{sd}"), out.get(f"ankle_{sd}")
    if hp is not None and kn is not None:
      u = (kn.pos - hp.pos) / np.linalg.norm(kn.pos - hp.pos)
      e = nt.joint_axis(f"knee_{sd}")
      e = latv if e is None else e * np.sign(e @ latv or 1.0)
      for name, sgn, what in ((f"femur_epi_lat_{sd}", 1, "lateral"), (f"femur_epi_med_{sd}", -1, "medial")):
        if explicit(name, sd):
          continue
        g = nt.segment_points([f"femur_{sd}"], sd, hint(name))
        if g is None:
          put(name, None, "no femur geometry")
          continue
        P, B = g
        put(name, _pick(P, B, sgn * ((P - kn.pos) @ e), np.abs((P - kn.pos) @ u) < 0.04 * s),
            f"{what} extreme of the distal femur (+-4 cm of the knee centre) along the knee axis")
    if an is not None:
      for name, sgn, what in ((f"malleolus_lat_{sd}", 1, "lateral"), (f"malleolus_med_{sd}", -1, "medial")):
        if explicit(name, sd):
          continue
        g = nt.segment_points([f"tibia_{sd}"], sd, hint(name))
        if g is None:
          put(name, None, "no tibia geometry")
          continue
        P, B = g
        put(name, _pick(P, B, sgn * (P @ latv), np.abs(P[:, 2] - an.pos[2]) < 0.03 * s),
            f"{what} extreme of the shank (tibia / fibula) within 3 cm of the ankle centre height")

  # pelvis
  for sd in "lr":
    sg = SIDE_SIGN[sd]
    for name in (f"asis_{sd}", f"psis_{sd}"):
      if explicit(name, sd):
        continue
      h = hint(name)
      if name.startswith("psis") and not h:  # the PSIS is on the ilium: sacrum meshes out
        h = {"exclude_geoms": ["*sacrum*"]}
      g = nt.segment_points(["pelvis"], sd, h)
      if g is None and name.startswith("psis"):  # one pelvis mesh including the sacrum
        g = nt.segment_points(["pelvis"], sd, None)
      if g is None or zh is None:
        put(name, None, "no pelvis geometry / hip centres")
        continue
      P, B = g
      lat = sg * (P[:, 0] - x0)
      if name.startswith("asis"):
        put(name, _pick(P, B, -P[:, 1], (lat > 0.6 * w) & (P[:, 2] > zh + 0.03 * s)),
            "most anterior pelvis vertex, lateral > 0.6 hip half-width, > 3 cm above hip centres")
      else:
        put(name, _pick(P, B, P[:, 1], (lat > 0.45 * w) & (lat < 0.9 * w) & (P[:, 2] > zh + 0.02 * s)),
            "most posterior ilium vertex (sacrum meshes excluded), lateral 0.45-0.9 hip half-width, > 2 cm above hip centres")
  if not explicit("sacrum", None):
    g = nt.segment_points(["pelvis"], None, hint("sacrum"))
    if g is not None and "psis_l" in out and "psis_r" in out:
      P, B = g
      zp = (out["psis_l"].pos[2] + out["psis_r"].pos[2]) / 2
      put("sacrum", _pick(P, B, P[:, 1], (np.abs(P[:, 0] - x0) < 0.012 * s) & (np.abs(P[:, 2] - zp) < 0.03 * s)),
          "most posterior midline pelvis vertex at the PSIS height (+-3 cm)")
    else:
      put("sacrum", None, "needs pelvis geometry and both PSIS")

  # trunk
  T = nt.segment_points(["torso"], None, None)
  if T is not None:
    P, B = T
    mid = np.abs(P[:, 0] - x0) < 0.012 * s
    below = mid & (P[:, 2] < zsh)
    if below.any():
      ymin = P[below, 1].min()
      if not explicit("jugular", None):
        g = nt.segment_points(["torso"], None, hint("jugular")) if hint("jugular") else T
        Pj, Bj = g
        mj = (np.abs(Pj[:, 0] - x0) < 0.008 * s) & (Pj[:, 1] < ymin + 0.07 * s) & (Pj[:, 2] < zsh + 0.01 * s)
        put("jugular", _pick(Pj, Bj, Pj[:, 2], mj),
            "highest midline (+-8 mm) trunk vertex within 7 cm of the sternum front, below the shoulder centres")
      if not explicit("xiphoid", None):
        g = nt.segment_points(["torso"], None, hint("xiphoid")) if hint("xiphoid") else T
        Px, Bx = g
        mx = (np.abs(Px[:, 0] - x0) < 0.012 * s) & (Px[:, 1] < ymin + 0.03 * s) & (Px[:, 2] < zsh)
        put("xiphoid", _pick(Px, Bx, -Px[:, 2], mx), "lowest midline trunk vertex within 3 cm of the sternum front")
    if not explicit("c7", None):
      h = hint("c7")
      g = nt.segment_points(["torso", "neck"], None, h)
      if g is not None:
        Pc, Bc = g
        mc = np.abs(Pc[:, 0] - x0) < 0.012 * s
        how = "most posterior midline vertex of " + (str(h.get("geoms") or h.get("bodies")) if h else "trunk / neck")
        if h.get("band", True):
          # C7's spinous process: above the jugular notch (~T2 level), below the cervical
          # spinous processes (shorter, further forward); without a notch, near the shoulders
          if "jugular" in out:
            zj = out["jugular"].pos[2]
            mc &= (Pc[:, 2] > zj + 0.015 * s) & (Pc[:, 2] < zj + 0.10 * s)
            how += ", 1.5-10 cm above the jugular notch"
          else:
            mc &= (Pc[:, 2] > zsh - 0.01 * s) & (Pc[:, 2] < zsh + 0.08 * s)
            how += ", 1 cm below to 8 cm above the shoulder centres"
        put("c7", _pick(Pc, Bc, Pc[:, 1], mc), how)
    if not explicit("t10", None):
      h = hint("t10")
      g = nt.segment_points(["torso"], None, h)
      if g is not None:
        Pt, Bt = g
        mt = np.abs(Pt[:, 0] - x0) < 0.012 * s
        how = "most posterior midline vertex of " + (str(h.get("geoms") or h.get("bodies")) if h else "the trunk")
        if h.get("band", True):
          if "xiphoid" in out:
            mt &= np.abs(Pt[:, 2] - out["xiphoid"].pos[2]) < 0.03 * s
            how += " at the xiphoid height (+-3 cm)"
          else:
            mt &= False
        put("t10", _pick(Pt, Bt, Pt[:, 1], mt), how)
  else:
    for n in ("jugular", "xiphoid", "c7", "t10"):
      if not explicit(n, None):
        missing[n] = "no torso geometry"

  # head: the support point 45 deg from the midline (over the temples, as the
  # Plug-in-Gait head band), in a height band below the head top
  if headg is not None:
    for sd in "lr":
      sg = SIDE_SIGN[sd]
      for name, lo, hi, ap, how in (
          (f"forehead_{sd}", 0.08, 0.05, -1.0, "anterior"),
          (f"back_head_{sd}", 0.10, 0.06, 1.0, "posterior")):
        if explicit(name, sd):
          continue
        g = nt.segment_points(["head"], sd, hint(name)) if hint(name) else headg
        P, B = g
        lat = sg * (P[:, 0] - x0)
        msk = (lat > 0) & (P[:, 2] > head_top - lo * s) & (P[:, 2] < head_top - hi * s)
        put(name, _pick(P, B, ap * P[:, 1] + lat, msk),
            f"head support point 45 deg {how}-lateral, {hi * 100:.0f}-{lo * 100:.0f} cm below the head top")

  # arms
  for sd in "lr":
    sg = SIDE_SIGN[sd]
    latv = np.array([sg, 0.0, 0.0])
    sh, el, wr = (out.get(f"{j}_{sd}") for j in ("shoulder", "elbow", "wrist"))
    name = f"acromion_{sd}"
    if not explicit(name, sd):
      segs = [f"scapula_{sd}"] if f"scapula_{sd}" in p.segments else (
        [f"clavicle_{sd}"] if f"clavicle_{sd}" in p.segments else ["torso"])
      g = nt.segment_points(segs, sd, hint(name))
      if sh is None:
        pass  # no arm: absent
      elif g is None:
        put(name, None, "no shoulder geometry")
      else:
        P, B = g
        lat = sg * (P[:, 0] - x0)
        reach = 0.05 if segs != ["torso"] else 0.035
        msk = lat > sg * (sh.pos[0] - x0) - reach * s
        if segs == ["torso"]:
          msk &= P[:, 2] > sh.pos[2]
        put(name, _pick(P, B, P[:, 2], msk),
            f"highest {'/'.join(segs)} vertex lateral of the shoulder centre - {reach * 100:.1f} cm")
    if sh is not None and el is not None:
      u = (el.pos - sh.pos) / np.linalg.norm(el.pos - sh.pos)
      e = nt.joint_axis(f"elbow_{sd}")
      if e is None:
        e = latv
      e = e * np.sign(e @ latv or 1.0)
      for name, sgn, what in ((f"epicondyle_lat_{sd}", 1, "lateral"), (f"epicondyle_med_{sd}", -1, "medial")):
        if explicit(name, sd):
          continue
        g = nt.segment_points([f"humerus_{sd}"], sd, hint(name))
        if g is None:
          put(name, None, "no humerus geometry")
          continue
        P, B = g
        msk = np.abs((P - el.pos) @ u) < 0.03 * s
        put(name, _pick(P, B, sgn * ((P - el.pos) @ e), msk),
            f"{what} extreme of the distal humerus (+-3 cm of the elbow centre) along the elbow axis")
    if el is not None and wr is not None:
      f = (wr.pos - el.pos) / np.linalg.norm(wr.pos - el.pos)
      same = (f"radius_{sd}" in p.segments and f"ulna_{sd}" in p.segments
              and p.segments[f"radius_{sd}"].body == p.segments[f"ulna_{sd}"].body)
      for name, seg, dflt in ((f"styloid_rad_{sd}", f"radius_{sd}", "anterior"),
                              (f"styloid_uln_{sd}", f"ulna_{sd}", "posterior")):
        if explicit(name, sd):
          continue
        h = hint(name)
        g = nt.segment_points([seg], sd, h)
        if g is None:
          put(name, None, f"no {seg} geometry")
          continue
        P, B = g
        pr = P @ f
        if not same:
          put(name, _pick(P, B, pr), f"most distal {seg} vertex along the forearm")
        else:
          dn = h.get("dir", dflt)
          dv = np.asarray(DIRS[dn] if dn in DIRS else
                          {"lateral": latv, "medial": -latv}[dn], float)
          put(name, _pick(P, B, P @ dv, pr > pr.max() - 0.02 * s),
              f"{dn} extreme of the distal 2 cm of the one-body forearm")
    if wr is not None:
      name = f"mc3_head_{sd}"
      if not explicit(name, sd):
        h = hint(name)
        g = nt.segment_points([f"hand_{sd}"], sd, h)
        if g is None:
          put(name, None, "no hand geometry")
        else:
          P, B = g
          hv = P.mean(0) - wr.pos
          hv /= np.linalg.norm(hv)
          pr = P @ hv
          if h.get("geoms"):
            put(name, _pick(P, B, pr), f"most distal vertex of {h['geoms']} along wrist -> hand")
          elif "fraction" in h:  # one mesh with the fingers: a fraction of the way to their tips
            f = float(h["fraction"])
            sel = pr > pr.max() - 0.01 * s
            tip = P[sel].mean(0)
            c = wr.pos + f * (tip - wr.pos)
            b = int(np.bincount(B).argmax())
            out[name] = nt.landmark(name, c, b, f"{f:g} of the way from the wrist centre to the centre "
                                                f"of the hand's distal end (inside the hand)")
          else:  # whole hand: centre of its distal end
            sel = pr > pr.max() - 0.01 * s
            c = P[sel].mean(0)
            b = np.bincount(B[sel]).argmax()
            out[name] = nt.landmark(name, c, b, "centre of the hand's distal 1 cm along wrist -> hand")

  _extra_landmarks(nt, p, out, put, explicit, hint, s, x0)

  bone = head_top - sole_z if head_top is not None else None
  return Result(p.name, out, sole_z, head_top, bone,
                bone + p.soft_tissue if bone is not None else None, nt.qpos.copy(), missing)


def _extra_landmarks(nt, p, out, put, explicit, hint, s, x0):
  """LANDMARKS_EXTRA: trochanter, fibula head, tibial tuberosity, metatarsal heads 1 / 2 / 5,
  scapula (inferior angle, root of the spine, acromial angle, acromion tip), olecranon,
  metacarpal heads 2 / 5. Rules as in compute (canonical frame, neutral pose)."""
  def unit(v):
    return v / np.linalg.norm(v)
  for sd in "lr":
    sg = SIDE_SIGN[sd]
    lat = lambda P: sg * (P[:, 0] - x0)
    hp, kn, mtp = (out.get(f"{j}_{sd}") for j in ("hip", "knee", "mtp"))
    # thigh / shank
    if hp is not None and not explicit(f"trochanter_{sd}", sd):
      g = nt.segment_points([f"femur_{sd}"], sd, hint(f"trochanter_{sd}"))
      if g is not None:
        P, B = g
        msk = (P[:, 2] > hp.pos[2] - 0.10 * s) & (P[:, 2] < hp.pos[2] + 0.03 * s)
        put(f"trochanter_{sd}", _pick(P, B, lat(P), msk),
            "most lateral femur vertex from 10 cm below to 3 cm above the hip centre")
    if kn is not None:
      g = nt.segment_points([f"tibia_{sd}"], sd, None)
      if g is not None:
        P, B = g
        band = (P[:, 2] > kn.pos[2] - 0.10 * s) & (P[:, 2] < kn.pos[2] - 0.03 * s)
        if not explicit(f"tibial_tuberosity_{sd}", sd):
          put(f"tibial_tuberosity_{sd}", _pick(P, B, -P[:, 1], band),
              "most anterior shank vertex 3-10 cm below the knee centre")
        if not explicit(f"fibula_head_{sd}", sd):
          band2 = (P[:, 2] > kn.pos[2] - 0.09 * s) & (P[:, 2] < kn.pos[2] - 0.02 * s)
          put(f"fibula_head_{sd}", _pick(P, B, lat(P) + 0.7 * P[:, 1], band2),
              "shank support point lateral-posterior (1 : 0.7), 2-9 cm below the knee centre")
    # forefoot
    if mtp is not None:
      g = nt.segment_points([f"foot_{sd}"], sd, None)
      if g is not None:
        P, B = g
        dy = P[:, 1] - mtp.pos[1]
        if not explicit(f"mt1_head_{sd}", sd):
          put(f"mt1_head_{sd}", _pick(P, B, -lat(P), np.abs(dy) < 0.02 * s),
              "most medial foot vertex within 2 cm (front-back) of the mtp centre")
        if not explicit(f"mt5_head_{sd}", sd):
          put(f"mt5_head_{sd}", _pick(P, B, lat(P), (dy > -0.005 * s) & (dy < 0.05 * s)),
              "most lateral foot vertex 0.5 cm in front of to 5 cm behind the mtp centre")
        if not explicit(f"mt2_head_{sd}", sd):
          msk = (np.abs(dy) < 0.015 * s) & (np.abs(lat(P) - sg * (mtp.pos[0] - x0)) < 0.012 * s)
          put(f"mt2_head_{sd}", _pick(P, B, P[:, 2], msk),
              "highest foot vertex over the mtp centre (+-1.5 cm front-back, +-1.2 cm sideways)")
    # scapula
    sh, acr = out.get(f"shoulder_{sd}"), out.get(f"acromion_{sd}")
    if sh is not None and acr is not None:
      on_scap = f"scapula_{sd}" in p.segments
      g = nt.segment_points([f"scapula_{sd}"] if on_scap else ["torso"], sd, None)
      if g is not None:
        P, B = g
        L = lat(P)
        za = acr.pos[2]
        if not explicit(f"scap_inf_angle_{sd}", sd):
          if on_scap:
            put(f"scap_inf_angle_{sd}", _pick(P, B, -P[:, 2]), "lowest scapula vertex")
          else:
            zx = out["xiphoid"].pos[2] if "xiphoid" in out else za - 0.25 * s
            msk = (L > 0.05 * s) & (L < 0.13 * s) & (P[:, 2] > zx) & (P[:, 2] < za)
            put(f"scap_inf_angle_{sd}", _pick(P, B, P[:, 1] - 0.3 * P[:, 2], msk),
                "trunk support point posterior-inferior (1 : 0.3), 5-13 cm lateral (no scapula body)")
        if not explicit(f"scap_spine_root_{sd}", sd):
          band = (P[:, 2] > za - 0.07 * s) & (P[:, 2] < za - 0.02 * s)
          if on_scap:  # behind the shoulder centre: the scapula, not a clavicle in the same mesh
            put(f"scap_spine_root_{sd}", _pick(P, B, -L, band & (P[:, 1] > sh.pos[1])),
                "most medial scapula vertex 2-7 cm below the acromion, behind the shoulder centre")
          else:
            put(f"scap_spine_root_{sd}", _pick(P, B, P[:, 1], band & (L > 0.05 * s) & (L < 0.10 * s)),
                "most posterior trunk vertex 5-10 cm lateral, 2-7 cm below the acromion (no scapula body)")
        top = (P[:, 2] > za - 0.03 * s) & (L > sg * (sh.pos[0] - x0) - 0.04 * s)
        if not explicit(f"acromial_angle_{sd}", sd):
          put(f"acromial_angle_{sd}", _pick(P, B, P[:, 1] + L, top),
              "support point posterior-lateral of the acromion region (within 3 cm of its top)")
        if not explicit(f"acromion_tip_{sd}", sd):
          put(f"acromion_tip_{sd}", _pick(P, B, L, top), "most lateral vertex of the acromion region")
    # elbow / hand
    el, wr = out.get(f"elbow_{sd}"), out.get(f"wrist_{sd}")
    if sh is not None and el is not None and not explicit(f"olecranon_{sd}", sd):
      u = unit(el.pos - sh.pos)
      g = nt.segment_points([f"ulna_{sd}"], sd, hint(f"olecranon_{sd}"))
      if g is not None:
        P, B = g
        msk = np.abs((P - el.pos) @ u) < 0.04 * s
        put(f"olecranon_{sd}", _pick(P, B, P[:, 1] - 0.5 * ((P - el.pos) @ u), msk),
            "ulna support point posterior-proximal (1 : 0.5) within 4 cm of the elbow centre")
    mc3 = out.get(f"mc3_head_{sd}")
    if wr is not None and mc3 is not None:
      rad, uln = out.get(f"styloid_rad_{sd}"), out.get(f"styloid_uln_{sd}")
      for name, sgn, gap in ((f"mc2_head_{sd}", 1.0, 0.02), (f"mc5_head_{sd}", -1.0, 0.025)):
        if explicit(name, sd):
          continue
        h = hint(name)
        if h.get("geoms"):
          g = nt.segment_points([f"hand_{sd}"], sd, h)
          if g is not None:
            P, B = g
            put(name, _pick(P, B, P @ unit(mc3.pos - wr.pos)),
                f"most distal vertex of {h['geoms']} along wrist -> hand")
            continue
        if rad is not None and uln is not None:  # one-piece hand: beside the 3rd metacarpal head
          across = rad.pos - uln.pos
          across -= (across @ unit(mc3.pos - wr.pos)) * unit(mc3.pos - wr.pos)
          c = mc3.pos + sgn * gap * s * unit(across)
          out[name] = nt.landmark(name, c, nt.m.body(mc3.body).id,
                                  f"{gap * 100:g} cm to the {'radial' if sgn > 0 else 'ulnar'} side of "
                                  "the 3rd metacarpal head (one-piece hand)")


# --- contact points ---------------------------------------------------------------

def sole_points(nt: Neutral) -> list[dict]:
  """Foot contact points at the neutral pose: dict(body, local, pos, side, kind), kind heel /
  toe / "-". profile [contact] sole_bodies = {l = [heel body, ..., toe body], r = [...]};
  sole_geoms = "collision": the lowest points of those bodies' collision geoms
  (sole_geom_types, default capsule / sphere / ellipsoid / box: both capsule ends with the
  radius below the axis, sphere / ellipsoid bottoms, the 4 lowest box corners), in model
  geom order, heel = rear end of a heel-body geom, toe = front end of a toe-body geom
  (exactly ik.MarkerModel._foot_points for MyoFullBody's capsules); "mesh": from the foot
  bones (heel = lowest vertex of the rearmost 20 %, toe = lowest of the front 10 %, first /
  fifth metatarsal heads = lowest medial / lateral vertex at 65-80 % of the foot length)."""
  m, d = nt.m, nt.d
  c = nt.p.contact
  mode = c.get("sole_geoms", "collision")
  sb = c.get("sole_bodies", {})
  out = []
  if mode == "collision":
    types = c.get("sole_geom_types", ["capsule", "sphere", "ellipsoid", "box"])
    T = mujoco.mjtGeom
    tnames = {T.mjGEOM_CAPSULE: "capsule", T.mjGEOM_SPHERE: "sphere",
              T.mjGEOM_ELLIPSOID: "ellipsoid", T.mjGEOM_BOX: "box"}
    where = {}  # body id -> (side, heel body?, toe body?)
    for sd, bodies in sb.items():
      bid = [m.body(bn).id for bn in bodies]
      for b in bid:
        where[b] = (sd, b == bid[0], b == bid[-1] and len(bid) > 1)
    for g in range(m.ngeom):
      b = int(m.geom_bodyid[g])
      t = m.geom_type[g]
      if b not in where or not (m.geom_contype[g] or m.geom_conaffinity[g]) or tnames.get(t) not in types:
        continue
      sd, heel_body, toe_body = where[b]
      Rb = d.xmat[b].reshape(3, 3)
      R = d.geom_xmat[g].reshape(3, 3)
      sz, x = m.geom_size[g], d.geom_xpos[g]
      if t == T.mjGEOM_CAPSULE:
        pts = [x + sgn * R[:, 2] * sz[1] - np.array([0, 0, sz[0]]) for sgn in (1, -1)]
      elif t == T.mjGEOM_SPHERE:
        pts = [x - np.array([0, 0, sz[0]])]
      elif t == T.mjGEOM_ELLIPSOID:
        pts = [x - np.array([0, 0, np.sqrt(((R[2] * sz) ** 2).sum())])]
      else:
        cs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * sz
        W = x + cs @ R.T
        pts = list(W[np.argsort(W[:, 2])[:4]])
      ys = [w[1] for w in pts]
      for w, y in zip(pts, ys):
        kind = "-"
        if heel_body and len(pts) > 1 and y == max(ys):
          kind = "heel"
        if toe_body and len(pts) > 1 and y == min(ys):
          kind = "toe"
        out.append(dict(body=m.body(b).name, local=Rb.T @ (w - d.xpos[b]), pos=np.asarray(w, float),
                        side=sd, kind=kind))
    return out
  for sd, bodies in sb.items():
    P, B = nt.points(bodies)
    y = P[:, 1]
    fr = (y.max() - y) / (y.max() - y.min())  # 0 at the heel, 1 at the toe tip
    lat = SIDE_SIGN[sd] * P[:, 0]
    xm = np.median(lat[(fr > 0.6) & (fr < 0.85)])
    for kind, msk in (("heel", fr < 0.2), ("toe", fr > 0.9),
                      ("-", (fr > 0.65) & (fr < 0.8) & (lat < xm)),
                      ("-", (fr > 0.65) & (fr < 0.8) & (lat >= xm))):
      i = np.flatnonzero(msk)[np.argmin(P[msk, 2])]
      b = int(B[i])
      Rb = d.xmat[b].reshape(3, 3)
      out.append(dict(body=m.body(b).name, local=Rb.T @ (P[i] - d.xpos[b]), pos=P[i].copy(),
                      side=sd, kind=kind))
  return out
