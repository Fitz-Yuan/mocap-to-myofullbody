"""Box lifting on any model with hands: the task's bodies through the model profile.

The task needs, for each hand, a body and a palm point on it (where the box rests in the
hand: grasp detection, the box's load on the hands, the QA's hand-box gap), the feet
(sole height, where the subject stands) and, for the motion dataset, a set of tracked
bodies. They come from the model profile (mocap2myo/profiles/): its canonical segments
([segments] hand_l, foot_l, ...) and its hand and sole definitions ([contact] palm_body,
sole_bodies), through mocap2myo.ik.MarkerModel. MyoFullBody: the thirdmc bodies and the
hand sphere of mocap2myo/objects.py, the foot collision capsules (as before). Another model:
its palm landmarks (65 % of the way wrist -> third metacarpal head, MarkerModel._palms) and
its sole points (MarkerModel.foot_points). A model without hands the IK moves (no hand
segments or no palm_body: MyoLeg, Gait2354, MS-Human-700 Locomotion with its rigid arms)
is refused (check_model): run it with M2M_TASK=none.
"""

import numpy as np

from mocap2myo import ik
from mocap2myo.kinematics import foot_collision_geoms, sole_height

# the motion dataset's tracked bodies (MyoFullBody's names, motion_io.TRACKED) as canonical
# segments; "ankle_*": the body of the profile's ankle joint centre (MyoFullBody's talus)
TRACKED_SEGMENTS = ("pelvis", "tibia_l", "tibia_r", "ankle_l", "ankle_r", "toes_l", "toes_r",
                    "torso", "head", "humerus_l", "ulna_l", "hand_l", "humerus_r", "ulna_r", "hand_r")


def is_myofullbody(profile):
  return profile.name.startswith("myofullbody")


def check_model(profile):
  """SystemExit unless the model has both hands with a palm (the profile's [segments]
  hand_l / hand_r and [contact] palm_body l / r)."""
  pb = profile.contact.get("palm_body", {})
  why = []
  if any(s not in profile.segments for s in ("hand_l", "hand_r")):
    why.append("no hand segments")
  elif any(s not in pb for s in "lr"):
    why.append("no palm_body in [contact] (the arms are not moved by the IK)")
  if why:
    raise SystemExit(f"M2M_TASK=box_lift needs a model with hands: {profile.name} ({profile.title}) "
                     f"has {why[0]}. Run it with M2M_TASK=none (body motion and joint loads without "
                     "the box), or with a model that has arms (myofullbody, ms_human_700, mass, "
                     "rajagopal2016, lai_uhlrich2022)")


class TaskBody:
  """The task's view of the model (M2M_MODEL): MarkerModel mm (its m / d hold the pose),
  hand bodies [2] and palm points in them [2, 3] (l, r), sole height and foot positions."""

  def __init__(self, profile=None, mm=None):
    self.mm = mm if mm is not None else ik.MarkerModel(profile)
    p = self.mm.profile
    check_model(p)
    if len(self.mm.hand_b) != 2:
      raise SystemExit(f"M2M_TASK=box_lift: {p.name} has no palm points (its hand landmarks are "
                       "missing: wrist, mc3_head, styloid_rad, styloid_uln); run it with M2M_TASK=none")
    self.m, self.d = self.mm.m, self.mm.d
    self.hand = list(self.mm.hand_b)
    self.palm_local = [np.asarray(c, float) for c in self.mm.palm_c]
    self.foot_body = [self.m.body(p.segment_body(f"foot_{s}")).id for s in "lr"]
    # MyoFullBody: the lowest point of its foot collision capsules; others: of the sole points
    self.foot_geoms = foot_collision_geoms(self.m) if is_myofullbody(p) else None

  def palms(self, d=None):
    """Palm points [2, 3] (l, r) for the pose in d (default self.d; kinematics done)."""
    d = self.d if d is None else d
    return np.stack([d.xpos[b] + d.xmat[b].reshape(3, 3) @ c for b, c in zip(self.hand, self.palm_local)])

  def sole(self):
    """Lowest sole point (z) for the pose in self.d."""
    if self.foot_geoms is not None:
      return sole_height(self.m, self.d, self.foot_geoms)
    return float(self.mm.foot_points()[:, 2].min())


def tracked_bodies(profile, m):
  """The motion dataset's tracked bodies on this model (MyoFullBody: motion_io.TRACKED)."""
  out = []
  for seg in TRACKED_SEGMENTS:
    if seg.startswith("ankle_"):
      jc = profile.joint_centres.get(seg, "")
      kind, _, name = jc.partition(":")
      b = (m.body(int(m.jnt_bodyid[m.joint(name).id])).name if kind == "jnt" else name) if name else None
    else:
      b = profile.segment_body(seg)
    if b is None:
      raise SystemExit(f"{profile.name}: no body for the tracked segment {seg}")
    out.append(b)
  return tuple(out)
