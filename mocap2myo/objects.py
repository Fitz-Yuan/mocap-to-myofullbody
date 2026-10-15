"""Hand contact spheres and the lift objects (geometry shared by data + sim).

Hands: one collision sphere per hand (radius HAND_R) in the palm; the finger /
metacarpal collision capsules are switched off, the bone meshes still render,
so the hand looks like a hand but touches the world as a ball. Grasp = a weld
between each gripping hand and the object (stage 2; a firm grip has enough
friction that the object does not turn in the hand), so the fingers need no
control.

Objects (2 kg; MMH does not describe its load, so these are our own designs):
  carton    two-handed. Closed cardboard box with a hand hole near the top of
            each end wall; the hands hold the two end walls. Size W (between
            the hands, from the data) x CARTON_DEPTH x H. Grip anchors = sphere
            centres, just outside the end walls, GRIP_DROP below the top.
            (A first open blue tote looked like a bin.)
  toolcase  one-handed (right). Case with a top handle; it hangs under the hand.
            Size TOOLCASE_LEN (along the handle) x TOOLCASE_WIDTH x H, grip
            anchor HANDLE_H above the lid.
Two-handed reference: the forearm collision capsules (radius up to 3 cm,
soft-tissue bulk) overlap the carton ~3 cm near the wrists (audit.json), since
they are thicker than the 2 cm palm sphere that sets the carton width. Stage 2
should exclude object <-> radius/ulna contacts rather than fight it.
Both: body frame at the centre, x = between the hands / along the handle,
z up. Physics uses one solid box (the carton body itself / a hidden one for the
toolcase); the rest is visual only.
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

from mocap2myo.model import MYOFULLBODY_XML

HAND_R = 0.02  # m: palm soft tissue over the metacarpals
SPHERE_PALMAR = 0.0  # sphere centre on the metacarpal (surface = palm skin)
SPHERE_DISTAL = 0.015  # and toward the fingers
HAND_BODIES = ("mc", "proxph", "midph", "distph", "thumb")  # capsules switched off

CARTON_DEPTH = 0.32
GRIP_DROP = 0.035  # hand-hole centre below the top
WALL = 0.008
TOOLCASE_LEN = 0.36
TOOLCASE_WIDTH = 0.17
HANDLE_H = 0.075  # sphere centre above the lid


@lru_cache(maxsize=1)
def hand_sphere_local() -> dict[str, np.ndarray]:
  """Sphere centre in the thirdmc_{l,r} body frame.

  Palm normal from the hand bones' chirality (posture independent):
  across = index - little metacarpal, distal = proximal phalanx 3 - thirdmc;
  palmar = across x distal (right hand), distal x across (left hand).
  """
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  p = lambda n: d.xpos[m.body(n).id]
  out = {}
  for s in ("l", "r"):
    across = p(f"secondmc_{s}") - p(f"fifthmc_{s}")
    distal = p(f"3proxph_{s}") - p(f"thirdmc_{s}")
    distal /= np.linalg.norm(distal)
    n = np.cross(across, distal) if s == "r" else np.cross(distal, across)
    n /= np.linalg.norm(n)
    w = p(f"thirdmc_{s}") + SPHERE_PALMAR * n + SPHERE_DISTAL * distal
    R = d.xmat[m.body(f"thirdmc_{s}").id].reshape(3, 3)
    out[s] = R.T @ (w - p(f"thirdmc_{s}"))
  return out


@lru_cache(maxsize=1)
def hand_palmar_local() -> dict[str, np.ndarray]:
  """Unit palm normal (pointing out of the palm) in the thirdmc frame."""
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  p = lambda n: d.xpos[m.body(n).id]
  out = {}
  for s in ("l", "r"):
    across = p(f"secondmc_{s}") - p(f"fifthmc_{s}")
    distal = p(f"3proxph_{s}") - p(f"thirdmc_{s}")
    n = np.cross(across, distal) if s == "r" else np.cross(distal, across)
    R = d.xmat[m.body(f"thirdmc_{s}").id].reshape(3, 3)
    out[s] = R.T @ (n / np.linalg.norm(n))
  return out


def hand_across_local() -> dict[str, np.ndarray]:
  """Index -> little direction (unit) in the thirdmc frame (handle axis)."""
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  p = lambda n: d.xpos[m.body(n).id]
  out = {}
  for s in ("l", "r"):
    a = p(f"fifthmc_{s}") - p(f"secondmc_{s}")
    R = d.xmat[m.body(f"thirdmc_{s}").id].reshape(3, 3)
    out[s] = R.T @ (a / np.linalg.norm(a))
  return out


def add_hand_spheres(spec: mujoco.MjSpec, visible: bool = False) -> None:
  """Hand capsules off, one palm sphere per hand (same collision bits as the
  other body geoms: contype 1, conaffinity 0)."""
  for body in spec.bodies:
    b = body.name
    if not (any(k in b for k in HAND_BODIES) and b[-2:] in ("_l", "_r")):
      continue
    for g in body.geoms:
      g.contype = 0
      g.conaffinity = 0
  for s, c in hand_sphere_local().items():
    spec.body(f"thirdmc_{s}").add_geom(
      name=f"hand_sphere_{s}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[HAND_R, 0, 0],
      pos=c.tolist(), contype=1, conaffinity=0, group=1 if visible else 3,
      rgba=[0.9, 0.3, 0.3, 0.35])


# --- objects ----------------------------------------------------------------

def carton_anchors(size) -> np.ndarray:
  """Grip anchors (left hand, right hand) in the carton frame. Left = +x."""
  W, _, H = size
  z = H / 2 - GRIP_DROP
  return np.array([[W / 2 + HAND_R, 0, z], [-W / 2 - HAND_R, 0, z]])


def toolcase_anchor(size) -> np.ndarray:
  return np.array([0.0, 0.0, size[2] / 2 + HANDLE_H])


def _box(body, size, pos, rgba, name=None, group=0, collide=False):
  kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(np.asarray(size) / 2),
            pos=list(pos), rgba=rgba, group=group,
            contype=1 if collide else 0, conaffinity=1 if collide else 0)
  if name:
    kw["name"] = name
  return body.add_geom(**kw)


def add_carton(body, size, collision_name="obj_col") -> None:
  """Closed cardboard box: taped top seam, a hand-hole slot near the top of
  each end wall (where the hands hold it), printed label."""
  W, D, H = size
  kraft, tape, hole = [0.66, 0.48, 0.3, 1], [0.78, 0.66, 0.45, 1], [0.12, 0.08, 0.05, 1]
  label = [0.92, 0.92, 0.88, 1]
  _box(body, size, (0, 0, 0), kraft, collision_name, group=0, collide=True)
  _box(body, (0.05, D + 0.002, 0.003), (0, 0, H / 2 + 0.0005), tape)  # top tape
  for sy in (-1, 1):  # tape running down the long faces
    _box(body, (0.05, 0.003, 0.06), (0, sy * (D / 2 + 0.0005), H / 2 - 0.03), tape)
  _box(body, (0.003, D, 0.002), (0, 0, H / 2 + 0.002), [0.5, 0.36, 0.22, 1])  # flap seam
  zc = H / 2 - GRIP_DROP
  for sx in (-1, 1):  # hand holes
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.016, 0.035, 0],
                  pos=[sx * (W / 2 + 0.0005), 0, zc], quat=[0.7071068, 0.7071068, 0, 0],
                  rgba=hole, contype=0, conaffinity=0)
  _box(body, (min(0.1, W - 0.04), 0.002, 0.07), (0, -(D / 2 + 0.001), -H * 0.1), label)


def add_toolcase(body, size, collision_name="obj_col") -> None:
  """Red case with a lid seam, latches and a top handle on two posts."""
  L, Wd, H = size
  red, black, metal = [0.75, 0.12, 0.1, 1], [0.08, 0.08, 0.08, 1], [0.7, 0.7, 0.72, 1]
  _box(body, size, (0, 0, 0), [1, 1, 1, 0], collision_name, group=3, collide=True)
  _box(body, (L, Wd, H * 0.72), (0, 0, -H * 0.14), red)  # body
  _box(body, (L + 0.006, Wd + 0.006, H * 0.06), (0, 0, H * 0.25), black)  # seam
  _box(body, (L, Wd, H * 0.22), (0, 0, H * 0.39), red)  # lid
  for sx in (-1, 1):  # latches + handle posts
    _box(body, (0.03, 0.006, 0.05), (sx * L * 0.3, Wd / 2 + 0.003, H * 0.25), metal)
    _box(body, (0.02, 0.03, HANDLE_H + 0.01),
         (sx * 0.065, 0, H / 2 + (HANDLE_H + 0.01) / 2), black)
  body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.013, 0.07, 0],
                pos=[0, 0, H / 2 + HANDLE_H], quat=[0.7071068, 0, 0.7071068, 0],
                rgba=black, contype=0, conaffinity=0)


def add_storage_box(body, size, collision_name="obj_col") -> None:
  """The case study's box (MMH Elevator data): clear storage box (translucent
  walls, lid, a dark weight inside, blue latches on the two ends = where the
  hands hold it). Size L (between the hands) x W x H."""
  L, W, H = size
  clear, lid = [0.82, 0.90, 0.98, 0.30], [0.55, 0.72, 0.92, 0.65]
  latch_rgba, weight = [0.12, 0.33, 0.78, 1], [0.22, 0.23, 0.25, 1]
  t, la = 0.008, 0.018  # wall thickness, latch depth
  Lw = L - 2 * la  # walls inside the latches
  _box(body, size, (0, 0, 0), [1, 1, 1, 0], collision_name, group=3, collide=True)
  for s in (-1, 1):
    _box(body, (Lw, t, H - 0.02), (0, s * (W / 2 - t / 2), -0.01), clear)
    _box(body, (t, W, H - 0.02), (s * (Lw / 2 - t / 2), 0, -0.01), clear)
    _box(body, (la, W * 0.64, 0.044), (s * (Lw / 2 + la / 2), 0, H / 2 - 0.03), latch_rgba)
  _box(body, (Lw, W, t), (0, 0, -H / 2 + t / 2), clear)  # floor
  _box(body, (Lw + 0.012, W + 0.012, 0.02), (0, 0, H / 2 - 0.01), lid)
  _box(body, (Lw * 0.45, W * 0.6, H * 0.36), (0, 0, -H / 2 + t + H * 0.18), weight)


# OMOMO objects (lift_omomo data): drawn as their oriented bounding boxes
PLAIN_RGBA = {"largebox": [0.62, 0.45, 0.28, 1], "smallbox": [0.72, 0.56, 0.36, 1],
              "plasticbox": [0.35, 0.45, 0.6, 1], "suitcase": [0.2, 0.2, 0.22, 1]}


def add_object(body, kind: str, size, collision_name="obj_col") -> None:
  if kind == "carton":
    add_carton(body, size, collision_name)
  elif kind == "toolcase":
    add_toolcase(body, size, collision_name)
  elif kind == "storage_box":
    add_storage_box(body, size, collision_name)
  else:
    _box(body, size, (0, 0, 0), PLAIN_RGBA[kind], collision_name, group=0, collide=True)
