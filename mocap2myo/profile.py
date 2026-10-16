"""Model profiles: what the pipeline needs to know about a musculoskeletal model.

A profile is a TOML file, mocap2myo/profiles/<name>.toml. It says

  [model]     where the MJCF file is ("pkg:<path>" inside this package, or a path under
              the models folder M2M_MODELS_DIR, default ~/data/mocap2myo/models, filled
              by models/fetch.py), optional variants selected by an environment variable,
              optional patch = [[old, new], ...] text replacements before parsing (MuJoCo
              2.x attributes 3.x rejects); top-level sources = [...] (profiles/sources.toml);
  [build]     how models/fetch.py makes that file from the upstream sources
              (profiles/sources.toml: repository, pinned commit, licence);
  [root]      the floating base: an existing free joint, or a body whose joints (e.g. the
              OpenSim pelvis_tx ... pelvis_rotation) are replaced by a free joint "root"
              (the IK expects qpos[0:3] position, qpos[3:7] quaternion);
  [frame]     the direction the model faces and the up direction at its neutral pose
              (root at its rest orientation); canonical frame of this repository: z up,
              facing -y, left = +x (MyoFullBody's own world);
  [neutral]   the neutral (anatomical standing) pose: a keyframe and / or joint values;
              the root position of the repository's MyoFullBody convention, if any;
  [ik]        the joints the IK solves: an explicit list, or include / exclude globs
              (joints driven by an equality constraint are never solved);
  [segments]  canonical segment -> model bodies. The first body is the one markers attach
              to; all listed bodies (rigid with it) contribute geometry for the landmarks.
              A table {bodies = [...], geoms = [globs]} restricts the geometry to some
              geoms (glob on geom or mesh name), e.g. the skull of a one-body trunk;
  [joint_centres] landmark -> "jnt:<joint>" (anchor) or "body:<body>" (origin);
  [kinetics]  the joints of the inverse dynamics, as mocap2myo.kinetics.JOINTS: distal body,
              point ("jnt:" / "body:"), frame body, components [name, axis, sign] with axis
              "x" / "y" / "z" of the frame body, "jnt:<joint>" (its axis) or, new here,
              "anat:<anterior|posterior|superior|inferior|left|right>": that canonical
              direction fixed in the frame body at the neutral pose (model independent;
              kinetics.py does not read it yet);
  [contact]   sole_bodies {l = [heel body, ..., toe body], r = [...]}, sole_geoms
              "collision" (the bodies' collision geoms, sole_geom_types) or "mesh" (from the
              foot bones), palm_body {l, r} (mocap2myo.landmarks.sole_points);
  [height]    soft_tissue added to the bone height (head top - foot-bone sole) for the
              model's standing height (MyoFullBody: 1.72 = 1.689 + 0.031, ik.MODEL_HEIGHT);
  [landmarks] per-landmark hints for mocap2myo.landmarks (geometry subsets, directions,
              or an explicit body point).

load(name) returns a Profile; Profile.spec() / compile() give the model with the free
root, neutral_qpos() the neutral pose in the canonical frame. mocap2myo.landmarks
computes the anatomical landmarks from it.
"""

from __future__ import annotations

import fnmatch
import os
from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

try:
  import tomllib
except ModuleNotFoundError:  # Python 3.10
  import tomli as tomllib

from mocap2myo.kinematics import apply_joint_equalities

PKG_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PKG_DIR / "profiles"
MODELS_DIR = Path(os.environ.get("M2M_MODELS_DIR", "~/data/mocap2myo/models")).expanduser()

# Canonical segments. Every profile maps the ones its model has; the others are absent
# (e.g. MyoLeg: no arms). radius / ulna name the same body when the forearm is one body.
SIDES = ("l", "r")
CANONICAL_SEGMENTS = (
  ("pelvis", "torso", "neck", "head")
  + tuple(f"{s}_{d}" for s in ("femur", "tibia", "foot", "toes", "clavicle", "scapula",
                               "humerus", "ulna", "radius", "hand") for d in SIDES))
CANONICAL_UP = np.array([0.0, 0.0, 1.0])
CANONICAL_FORWARD = np.array([0.0, -1.0, 0.0])
CANONICAL_LEFT = np.array([1.0, 0.0, 0.0])

_AXES = {"x": 0, "y": 1, "z": 2}


def _axis(s: str) -> np.ndarray:
  """'+z' / '-y' / 'x' -> unit vector."""
  s = s.strip()
  sign = -1.0 if s.startswith("-") else 1.0
  v = np.zeros(3)
  v[_AXES[s.lstrip("+-")]] = sign
  return v


def profile_names() -> list[str]:
  return sorted(p.stem for p in PROFILE_DIR.glob("*.toml") if p.stem != "sources")


def sources() -> dict:
  with open(PROFILE_DIR / "sources.toml", "rb") as f:
    return tomllib.load(f)


def load(name: str) -> Profile:
  path = PROFILE_DIR / f"{name}.toml"
  if not path.exists():
    raise ValueError(f"no model profile {name!r}; known: {', '.join(profile_names())}")
  with open(path, "rb") as f:
    return Profile(name, tomllib.load(f))


def resolve_path(p: str) -> Path:
  """'pkg:<path>' -> inside mocap2myo/; anything else -> under MODELS_DIR (or absolute)."""
  if p.startswith("pkg:"):
    return PKG_DIR / p[4:]
  q = Path(p).expanduser()
  return q if q.is_absolute() else MODELS_DIR / q


class Segment:
  """A canonical segment: the body markers attach to, and the geometry it contributes."""

  def __init__(self, name, spec):
    if isinstance(spec, str):
      spec = [spec]
    if isinstance(spec, list):
      spec = {"bodies": spec}
    self.name = name
    self.bodies = tuple(spec["bodies"])
    self.body = spec.get("body", self.bodies[0])
    self.geoms = tuple(spec.get("geoms", ()))  # globs on geom / mesh names; () = all
    self.exclude = tuple(spec.get("exclude_geoms", ()))

  def __repr__(self):
    g = f", geoms={list(self.geoms)}" if self.geoms else ""
    return f"Segment({self.name}: {self.body} <- {list(self.bodies)}{g})"


def geom_matches(m: mujoco.MjModel, g: int, globs) -> bool:
  names = [m.geom(g).name]
  if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_dataid[g] >= 0:
    names.append(m.mesh(m.geom_dataid[g]).name)
  return any(fnmatch.fnmatchcase(n, pat) for n in names if n for pat in globs)


class Profile:
  def __init__(self, name: str, data: dict):
    self.name = name
    self.data = data
    self.title = data.get("title", name)
    self.model_cfg = data.get("model", {})
    self.root_cfg = data.get("root", {})
    self.frame_cfg = data.get("frame", {"up": "+z", "forward": "-y"})
    self.neutral_cfg = data.get("neutral", {})
    self.ik_cfg = data.get("ik", {})
    self.segments = {k: Segment(k, v) for k, v in data.get("segments", {}).items()}
    unknown = set(self.segments) - set(CANONICAL_SEGMENTS)
    if unknown:
      raise ValueError(f"{name}: unknown canonical segments {sorted(unknown)}")
    self.joint_centres = dict(data.get("joint_centres", {}))
    self.kinetics = dict(data.get("kinetics", {}))
    self.contact = dict(data.get("contact", {}))
    self.landmark_hints = dict(data.get("landmarks", {}))
    self.soft_tissue = float(data.get("height", {}).get("soft_tissue", 0.031))

  # --- files -----------------------------------------------------------------
  @property
  def variant(self) -> str | None:
    var = self.model_cfg.get("variants")
    if not var:
      return None
    env = self.model_cfg.get("variant_env")
    v = os.environ.get(env, self.model_cfg["default_variant"]) if env else self.model_cfg["default_variant"]
    if v not in var:
      raise SystemExit(f"{env}={v!r}: expected one of {', '.join(var)}")
    return v

  @property
  def xml(self) -> Path:
    v = self.variant
    p = self.model_cfg["variants"][v] if v else self.model_cfg["xml"]
    return resolve_path(p)

  def check_available(self) -> None:
    if not self.xml.exists():
      raise FileNotFoundError(
        f"{self.name}: model file {self.xml} not found. Fetch / build it with "
        f"`python models/fetch.py {self.name}` (M2M_MODELS_DIR={MODELS_DIR}).")

  # --- model -----------------------------------------------------------------
  def spec(self) -> mujoco.MjSpec:
    """The model as a MjSpec with a free joint at the root (qpos[0:7]), keyframes removed
    when joints were replaced (their layout no longer fits; neutral_qpos keeps their
    values by joint name)."""
    self.check_available()
    spec = self._load_spec()
    rc = self.root_cfg
    if "body" in rc:  # replace the root body's joints by one free joint
      body = spec.body(rc["body"])
      if body is None:
        raise ValueError(f"{self.name}: root body {rc['body']!r} not found")
      if body.parent.name != "world":
        raise ValueError(f"{self.name}: root body {rc['body']!r} is not a child of the world")
      drop = set(rc.get("replace", []))
      for j in list(body.joints):
        if j.name in drop or not drop:
          spec.delete(j)
      for j in body.joints:
        raise ValueError(f"{self.name}: root body keeps joint {j.name!r} (list it in root.replace)")
      body.add_freejoint(name=rc.get("joint", "root"))
      for k in list(spec.keys):
        spec.delete(k)
    for k in self.model_cfg.get("delete_keys", []):
      spec.delete(spec.key(k))
    return spec

  def _load_spec(self) -> mujoco.MjSpec:
    """The file as it is, or with [model] patch = [[old, new], ...] text replacements
    (e.g. MuJoCo 2.x attributes that 3.x rejects); asset folders stay those of the file."""
    patch = self.model_cfg.get("patch")
    if not patch:
      return mujoco.MjSpec.from_file(str(self.xml))
    text = self.xml.read_text()
    for old, new in patch:
      if old not in text:
        raise ValueError(f"{self.name}: patch target {old!r} not in {self.xml}")
      text = text.replace(old, new)
    spec = mujoco.MjSpec.from_string(text)
    base = self.xml.parent
    spec.meshdir = str(base / spec.meshdir) if spec.meshdir else str(base)
    spec.texturedir = str(base / spec.texturedir) if spec.texturedir else str(base)
    return spec

  def compile(self) -> mujoco.MjModel:
    m = self.spec().compile()
    j = m.joint(self.root_joint)
    if m.jnt_type[j.id] != mujoco.mjtJoint.mjJNT_FREE or m.jnt_qposadr[j.id] != 0:
      raise ValueError(f"{self.name}: root joint {self.root_joint!r} is not a free joint at qpos[0:7]")
    return m

  @property
  def root_joint(self) -> str:
    return self.root_cfg.get("joint", "root")

  @property
  def root_body(self) -> str:
    return self.root_cfg.get("body") or self.root_cfg.get("root_body", "")

  @cached_property
  def rotation(self) -> np.ndarray:
    """R (3x3): model world at neutral -> canonical frame (z up, facing -y, left +x)."""
    up = _axis(self.frame_cfg["up"])
    fwd = _axis(self.frame_cfg["forward"])
    if abs(up @ fwd) > 1e-9:
      raise ValueError(f"{self.name}: frame.up and frame.forward must be perpendicular")
    left = np.cross(up, fwd)
    return np.stack([left, -fwd, up])  # rows: the model's left, back, up

  @cached_property
  def _keyframe_values(self) -> dict[str, np.ndarray]:
    """Values of the neutral keyframe by joint name (read from the unmodified file)."""
    key = self.neutral_cfg.get("keyframe")
    if key is None:
      return {}
    m0 = self._load_spec().compile()
    kid = m0.key(key).id if isinstance(key, str) else int(key)
    q = m0.key_qpos[kid]
    out = {}
    for j in range(m0.njnt):
      a = m0.jnt_qposadr[j]
      n = {0: 7, 1: 4, 2: 1, 3: 1}[int(m0.jnt_type[j])]
      out[m0.joint(j).name] = q[a:a + n].copy()
    return out

  def neutral_qpos(self, m: mujoco.MjModel) -> np.ndarray:
    """Neutral pose (keyframe, then [neutral] qpos, then the joint equalities) with the
    root turned into the canonical frame; the root at neutral.root_pos if the profile
    gives one (MyoFullBody: (0, 0, 0.99), the frame of markerset.py's anatomical
    guesses), else where the file puts it (mocap2myo.landmarks.Neutral then stands the
    model on the floor)."""
    q = m.qpos0.copy()
    for name, v in self._keyframe_values.items():
      jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid < 0 or m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
        continue
      a = m.jnt_qposadr[jid]
      q[a:a + len(v)] = v
    for name, v in self.neutral_cfg.get("qpos", {}).items():
      jid = m.joint(name).id
      q[m.jnt_qposadr[jid]] = v
    apply_joint_equalities(m, q)
    # root: rotate the rest orientation into the canonical frame
    from scipy.spatial.transform import Rotation as sRot
    r0 = sRot.from_quat(q[[4, 5, 6, 3]])
    r = sRot.from_matrix(self.rotation) * r0
    q[3:7] = r.as_quat()[[3, 0, 1, 2]]
    if "root_pos" in self.neutral_cfg:
      q[:3] = self.neutral_cfg["root_pos"]
    else:
      q[:3] = self.rotation @ q[:3]
    return q

  # --- IK joints ---------------------------------------------------------------
  def equality_followers(self, m: mujoco.MjModel) -> set[int]:
    return {int(m.eq_obj1id[e]) for e in range(m.neq)
            if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT}

  def ik_joints(self, m: mujoco.MjModel) -> list[str]:
    """Joints the IK solves, in order (the root's 6 DoF come first, separately)."""
    if "joints" in self.ik_cfg:
      names = list(self.ik_cfg["joints"])
      for n in names:
        m.joint(n)  # KeyError if missing
      return names
    inc = self.ik_cfg.get("include", ["*"])
    exc = self.ik_cfg.get("exclude", [])
    follow = self.equality_followers(m)
    out = []
    for j in range(m.njnt):
      n = m.joint(j).name
      if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE or j in follow:
        continue
      if not any(fnmatch.fnmatchcase(n, p) for p in inc):
        continue
      if any(fnmatch.fnmatchcase(n, p) for p in exc):
        continue
      out.append(n)
    return out

  def __repr__(self):
    return f"Profile({self.name}: {self.xml})"
