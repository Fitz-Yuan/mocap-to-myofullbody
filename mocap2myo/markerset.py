"""Marker sets: where each optical marker sits, as data (mocap2myo/markersets/<name>.toml).

A marker is described relative to the anatomy, not to a model: the canonical segment it
moves with (mocap2myo.profile.CANONICAL_SEGMENTS: pelvis, torso, femur_l, ...), the
anatomical landmark under it (mocap2myo.landmarks: asis_l, femur_epi_lat_l, c7, ...) and
its offset from that landmark in the segment's anatomical axes at the model's neutral pose
(anterior, superior, lateral; lateral = away from the midline on the landmark's side, to
the left for a midline landmark), so the same file serves every model profile. The model
profile maps the segment to a body (a segment the model lacks falls back, e.g. scapula ->
torso; without one the marker is left out of that model's IK) and computes the landmark
from its bones. A marker set file:

  name, description, sides = {l = "L", r = "R"}      label prefixes of the two sides
  [markers]          in array order: LABEL = {segment, landmark, offset_mm = [ant, sup, lat],
                     weight, group, along}. along = [landmark, f]: the point a fraction f
                     of the way from `landmark` to that other landmark (wand / mid-segment
                     markers). group: "loose" (placement varies: the calibration may move
                     the marker 15 cm from its guess), "foot" (shoe markers: M2M_FOOT_CAP),
                     else 6 cm.
  [roles]            head = {label = [front, left]} (head markers: low weight, never
                     rejected as mislabels, assigned by position; may be empty), pelvis =
                     [L ASIS, R ASIS, L PSIS, R PSIS] (root fit and rigid refill), heel /
                     toe = {l, r} (stance and flat-foot calibration), ankles (quiet-standing
                     detection), prefer (when two take labels map onto one marker)
  [aliases.<name>]   other naming schemes of the same markers: take label -> marker label
                     ("HEAD" = a head marker assigned by position); M2M_MARKER_ALIASES
                     picks which apply (auto: the one that matches the take's labels best,
                     if it beats the set's own labels)
  [override]         explicit anatomical guesses for some model profiles (profiles = [...]):
                     world positions of the model at its qpos0 with the root at its
                     neutral.root_pos (left = by label without the side letter, the right
                     side mirrors x; mid = by full label). Plug-in-Gait on MyoFullBody keeps
                     the hand-tuned guesses it was calibrated with this way.

load(name) returns a MarkerSet. The module-level names (LABELS, MARKERS, HEAD, PELVIS,
SOLE, ANKLES, ALIASES, PREFER, anat_world, ...) are those of the set named by
M2M_MARKER_SET (default plug_in_gait; pipeline/config.py sets it from the dataset
profile), for code that imports them directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
  import tomllib
except ModuleNotFoundError:  # Python 3.10
  import tomli as tomllib

SET_DIR = Path(__file__).resolve().parent / "markersets"
GROUPS = ("", "loose", "foot")


def names() -> list[str]:
  return sorted(p.stem for p in SET_DIR.glob("*.toml"))


@dataclass
class Marker:
  label: str
  segment: str                  # canonical segment
  landmark: str                 # mocap2myo.landmarks name
  offset: np.ndarray            # m, [anterior, superior, lateral]
  weight: float
  group: str = ""               # "", "loose", "foot"
  along: tuple | None = None    # (landmark, fraction)

  @property
  def lateral_sign(self) -> float:
    """+1: lateral = +x (left side or midline), -1: right side."""
    for n in (self.landmark, self.segment):
      if n.endswith("_r"):
        return -1.0
      if n.endswith("_l"):
        return 1.0
    return 1.0


class MarkerSet:
  def __init__(self, name: str, data: dict):
    self.key = name
    self.data = data
    self.name = data.get("name", name)
    self.sides = data.get("sides", {"l": "L", "r": "R"})
    self.markers: dict[str, Marker] = {}
    for label, v in data["markers"].items():
      g = v.get("group", "")
      if g not in GROUPS:
        raise ValueError(f"marker set {name}: {label}: group {g!r} (expected one of {GROUPS})")
      along = v.get("along")
      self.markers[label] = Marker(label, v["segment"], v["landmark"],
                                   np.asarray(v.get("offset_mm", [0, 0, 0]), float) / 1000.0,
                                   float(v.get("weight", 1.0)), g,
                                   (str(along[0]), float(along[1])) if along else None)
    roles = data.get("roles", {})
    self.LABELS = list(self.markers)
    self.MARKERS = {l: (m.segment, m.weight) for l, m in self.markers.items()}
    self.head = {l: tuple(v) for l, v in roles.get("head", {}).items()}
    # False: the head markers keep their labels (no reordering by position; Motive's skeleton
    # labels them reliably, and a single side marker has no fixed side)
    self.HEAD_REORDER = bool(roles.get("head_reorder", True))
    self.HEAD_LABELS = tuple(self.head)
    self.HEAD = np.array([l in self.head for l in self.LABELS])
    self.PELVIS_LABELS = tuple(roles["pelvis"])
    self.PELVIS = [self.LABELS.index(l) for l in self.PELVIS_LABELS]
    self.PELVIS_MASK = np.array([l in self.PELVIS_LABELS for l in self.LABELS])
    self.SOLE = {(s, k): roles[k][s] for s in ("l", "r") for k in ("heel", "toe")}
    self.ANKLES = tuple(roles["ankles"])
    self.PREFER = tuple(roles.get("prefer", ()))
    self.ALIASES = {k: dict(v) for k, v in data.get("aliases", {}).items()}
    self.LOOSE = np.array([m.group == "loose" for m in self.markers.values()])
    self.FOOT = np.array([m.group == "foot" for m in self.markers.values()])
    for l in (*self.HEAD_LABELS, *self.PELVIS_LABELS, *self.SOLE.values(), *self.ANKLES):
      if l not in self.markers:
        raise ValueError(f"marker set {name}: role marker {l!r} is not in [markers]")
    ov = data.get("override")
    self._override = None
    if ov:
      self._override = (tuple(ov["profiles"]), {k: tuple(v) for k, v in ov.get("left", {}).items()},
                        {k: tuple(v) for k, v in ov.get("mid", {}).items()})

  def __repr__(self):
    return f"MarkerSet({self.name}: {len(self.LABELS)} markers)"

  def segment(self, label: str) -> str:
    return self.markers[label].segment

  # --- anatomical guesses ---------------------------------------------------------
  def has_override(self, profile_name: str) -> bool:
    return self._override is not None and profile_name in self._override[0]

  def anat_world(self, label: str) -> np.ndarray:
    """The explicit guess of a marker ([override]; world, see the module docstring)."""
    _, left, mid = self._override
    if label in mid:
      return np.array(mid[label], float)
    side = next(s for s, p in self.sides.items() if label.startswith(p))
    x, y, z = left[label[len(self.sides[side]):]]
    return np.array([x if side == "l" else -x, y, z], float)

  def guess_world(self, label: str, landmarks: dict) -> np.ndarray | None:
    """Where the marker is on the model at its neutral pose in the canonical frame (z up,
    facing -y, left +x): its landmark (or the point `along` it), plus the offset in the
    anatomical axes. landmarks: mocap2myo.landmarks.Result.landmarks. None if a landmark
    it needs is missing."""
    mk = self.markers[label]
    if mk.landmark not in landmarks:
      return None
    p = landmarks[mk.landmark].pos.copy()
    if mk.along is not None:
      if mk.along[0] not in landmarks:
        return None
      p = p + mk.along[1] * (landmarks[mk.along[0]].pos - p)
    a, s, lt = mk.offset
    return p + np.array([mk.lateral_sign * lt, -a, s])


def load(name: str | MarkerSet) -> MarkerSet:
  """The marker set mocap2myo/markersets/<name>.toml (a MarkerSet passes through)."""
  if isinstance(name, MarkerSet):
    return name
  path = SET_DIR / f"{name}.toml"
  if not path.exists():
    raise ValueError(f"no marker set {name!r}; known: {', '.join(names())}")
  with open(path, "rb") as f:
    return MarkerSet(name, tomllib.load(f))


def selected() -> MarkerSet:
  """The marker set named by M2M_MARKER_SET (default plug_in_gait)."""
  return load(os.environ.get("M2M_MARKER_SET", "plug_in_gait"))


# --- the selected set's names, for code that imports them ------------------------------
MARKER_SET = selected()
MARKERS = MARKER_SET.MARKERS
LABELS = MARKER_SET.LABELS
HEAD_LABELS = MARKER_SET.HEAD_LABELS
HEAD = MARKER_SET.HEAD
PELVIS_LABELS = MARKER_SET.PELVIS_LABELS
PELVIS = MARKER_SET.PELVIS
PELVIS_MASK = MARKER_SET.PELVIS_MASK
SOLE = MARKER_SET.SOLE
ANKLES = MARKER_SET.ANKLES
ALIASES = MARKER_SET.ALIASES
PREFER = MARKER_SET.PREFER


def anat_world(label):
  return MARKER_SET.anat_world(label)
