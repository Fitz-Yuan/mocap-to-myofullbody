"""MASS (Lee, Park, Lee, Lee 2019; github.com/lsw9021/MASS, Apache-2.0) -> MuJoCo MJCF.

  python models/mass_to_mjcf.py HUMAN_XML MUSCLE_XML OBJ_DIR OUT_XML [--check]

MASS describes its character for DART: data/human.xml gives every body as a box (full
side lengths, mass, rest pose in world coordinates, y up, facing +z, left = +x; T-pose)
with the joint to its parent (Free / Ball / Revolute, joint frame in world coordinates)
and a bone mesh (data/OBJ, centimetres, already in world coordinates at the rest pose);
data/muscle284.xml gives 284 muscles as polylines of waypoints (world coordinates at the
rest pose, each on a body) with f0, lm, lt.

The conversion:
  bodies    one MuJoCo body per node, its frame at the node's joint centre and aligned
            with the world at the rest pose (so the rest pose is qpos = 0 with the root at
            its rest position); the box as a geom (group 1, colliding where MASS sets
            contact="On") and the bone mesh as a visual geom (group 0, scale 0.01);
            inertia of the box (DART: solid box of the given mass) set explicitly. Box
            rotations are Gram-Schmidt orthonormalised exactly as MASS does (one of them,
            FemurR, has a 0.21 that is likely a typo for 0.0021: MASS tilts that box by
            ~12 deg and so does this file; the mesh is not affected).
  joints    Free -> MuJoCo free joint "root"; Revolute -> hinge (axis, range); Ball -> three
            hinges <node>_x / _y / _z about the world axes at rest, ranges from the ball
            joint's lower / upper (DART limits the rotation-vector components; the same
            numbers as hinge limits are an approximation). Hinge order x, z, y (flexion,
            ab/adduction, axial rotation); for the upper arms z, x, y, so that the arms
            hanging down (z = -+90 deg from the T-pose) is far from gimbal lock.
  muscles   a spatial tendon through one site per waypoint, on the waypoint's body (MASS
            attaches intermediate waypoints by linear blend skinning to the nearest body
            and its parent when within 8 cm of a joint; here they are rigid on the given
            body), and a MuJoCo muscle on it: peak force f0; MASS's rigid-tendon model has
            optimal fibre length lm x L and tendon slack length lt x L with L the path
            length at the rest pose, so lengthrange = lt L + (0.75, 1.05) lm L with
            range = (0.75, 1.05) gives MuJoCo's L0 = lm L and LT = lt L. The force-length /
            velocity curves are MuJoCo's, not MASS's; pen_angle (0 throughout) and lmax are
            not used.

--check prints the validation: rest pose (box centres and orientations, joint centres vs
the file), mesh vs box placement, tendon lengths at rest vs the waypoint polylines, and
the normalised fibre lengths in the T-pose and with the arms down.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def vec(s, n=3):
  return np.array([float(x) for x in s.split()[:n]])


def gram_schmidt(R):
  """MASS::Orthonormalize: Gram-Schmidt on the columns."""
  v0, v1, v2 = R[:, 0], R[:, 1], R[:, 2]
  proj = lambda u, v: u.dot(v) / u.dot(u) * u
  u0 = v0
  u1 = v1 - proj(u0, v1)
  u2 = v2 - proj(u0, v2) - proj(u1, v2)
  U = np.stack([u / np.linalg.norm(u) for u in (u0, u1, u2)], 1)
  if np.linalg.det(U) < 0:
    raise ValueError("left-handed box frame")
  return U


def mat2quat(R):
  w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
  x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
  y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
  z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
  x = np.copysign(x, R[2, 1] - R[1, 2])
  y = np.copysign(y, R[0, 2] - R[2, 0])
  z = np.copysign(z, R[1, 0] - R[0, 1])
  q = np.array([w, x, y, z])
  return q / np.linalg.norm(q)


def fmt(v, nd=6):
  return " ".join(f"{x:.{nd}g}" if abs(x) > 1e-12 else "0" for x in np.atleast_1d(v))


ARM_NODES = ("ArmL", "ArmR")
RANGE_R = (0.75, 1.05)


def read_skeleton(path):
  nodes = []
  for n in ET.parse(path).getroot().findall("Node"):
    b, j = n.find("Body"), n.find("Joint")
    tb, tj = b.find("Transformation"), j.find("Transformation")
    Rb = gram_schmidt(vec(tb.get("linear"), 9).reshape(3, 3))
    nodes.append(dict(
      name=n.get("name"), parent=n.get("parent"), btype=b.get("type"), mass=float(b.get("mass")),
      size=vec(b.get("size")), contact=b.get("contact") == "On", obj=b.get("obj"),
      R=Rb, c=vec(tb.get("translation")), jtype=j.get("type"), J=vec(tj.get("translation")),
      jR=vec(tj.get("linear"), 9).reshape(3, 3),
      axis=vec(j.get("axis")) if j.get("axis") else None,
      lower=vec(j.get("lower"), 3 if j.get("type") == "Ball" else 1) if j.get("lower") else None,
      upper=vec(j.get("upper"), 3 if j.get("type") == "Ball" else 1) if j.get("upper") else None))
  return nodes


def read_muscles(path):
  out = []
  for u in ET.parse(path).getroot().findall("Unit"):
    wps = [(w.get("body"), vec(w.get("p"))) for w in u.findall("Waypoint")]
    out.append(dict(name=u.get("name"), f0=float(u.get("f0")), lm=float(u.get("lm")),
                    lt=float(u.get("lt")), pen=float(u.get("pen_angle")), wps=wps))
  return out


def convert(human, muscle, obj_dir, out):
  nodes = read_skeleton(human)
  muscles = read_muscles(muscle)
  byname = {n["name"]: n for n in nodes}
  for n in nodes:
    if n["btype"] != "Box":
      raise SystemExit(f"{n['name']}: body type {n['btype']} not supported")
    if not np.allclose(n["jR"], np.eye(3), atol=1e-6):
      raise SystemExit(f"{n['name']}: rotated joint frame not supported")
  root = ET.Element("mujoco", model="MASS")
  ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", autolimits="true",
                meshdir=os.path.relpath(obj_dir, out.parent))
  ET.SubElement(root, "option", timestep="0.002")
  vis = ET.SubElement(root, "visual")
  ET.SubElement(vis, "global", offwidth="1600", offheight="1600")
  df = ET.SubElement(root, "default")
  ET.SubElement(df, "joint", damping="0.5", armature="0.01")
  ET.SubElement(df, "geom", contype="0", conaffinity="0")
  ET.SubElement(df, "site", size="0.004", group="3")
  ET.SubElement(df, "tendon", width="0.003", rgba="0.8 0.2 0.2 1")
  ET.SubElement(df, "muscle", ctrllimited="true", ctrlrange="0 1")
  asset = ET.SubElement(root, "asset")
  for n in nodes:
    if n["obj"]:
      ET.SubElement(asset, "mesh", name=f"{n['name']}_mesh", file=n["obj"], scale="0.01 0.01 0.01")
  wb = ET.SubElement(root, "worldbody")
  ET.SubElement(wb, "light", pos="0 3 3", dir="0 -1 -1", directional="true")
  elem = {}
  for n in nodes:
    parent = wb if n["parent"] == "None" else elem[n["parent"]]
    ppos = np.zeros(3) if n["parent"] == "None" else byname[n["parent"]]["J"]
    b = ET.SubElement(parent, "body", name=n["name"], pos=fmt(n["J"] - ppos))
    elem[n["name"]] = b
    a, bb, c = n["size"]
    I = n["mass"] / 12 * np.array([bb * bb + c * c, a * a + c * c, a * a + bb * bb])
    ET.SubElement(b, "inertial", pos=fmt(n["c"] - n["J"]), quat=fmt(mat2quat(n["R"])),
                  mass=fmt(n["mass"]), diaginertia=fmt(I))
    nm = n["name"]
    if n["jtype"] == "Free":
      ET.SubElement(b, "freejoint", name="root")
    elif n["jtype"] == "Revolute":
      ET.SubElement(b, "joint", name=nm, type="hinge", axis=fmt(n["axis"]),
                    range=fmt([n["lower"][0], n["upper"][0]]))
    elif n["jtype"] == "Ball":
      order = "zxy" if nm in ARM_NODES else "xzy"
      for ax in order:
        i = "xyz".index(ax)
        e = np.zeros(3)
        e[i] = 1
        ET.SubElement(b, "joint", name=f"{nm}_{ax}", type="hinge", axis=fmt(e),
                      range=fmt([n["lower"][i], n["upper"][i]]))
    elif n["jtype"] != "Weld":
      raise SystemExit(f"{nm}: joint type {n['jtype']} not supported")
    ET.SubElement(b, "geom", name=f"{nm}_box", type="box", size=fmt(n["size"] / 2),
                  pos=fmt(n["c"] - n["J"]), quat=fmt(mat2quat(n["R"])), group="1",
                  rgba="0.5 0.55 0.9 0.35", contype="1" if n["contact"] else "0",
                  conaffinity="1" if n["contact"] else "0")
    if n["obj"]:
      ET.SubElement(b, "geom", name=f"{nm}_bone", type="mesh", mesh=f"{nm}_mesh",
                    pos=fmt(-n["J"]), group="0", rgba="0.9 0.85 0.75 1")
  # muscles
  tend = ET.SubElement(root, "tendon")
  act = ET.SubElement(root, "actuator")
  rest = {}
  for mu in muscles:
    t = ET.SubElement(tend, "spatial", name=mu["name"])
    for k, (bn, p) in enumerate(mu["wps"]):
      s = f"{mu['name']}-P{k + 1}"
      ET.SubElement(elem[bn], "site", name=s, pos=fmt(p - byname[bn]["J"]))
      ET.SubElement(t, "site", site=s)
    P = np.array([p for _, p in mu["wps"]])
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    rest[mu["name"]] = L
    L0, LT = mu["lm"] * L, mu["lt"] * L
    ET.SubElement(act, "muscle", name=mu["name"], tendon=mu["name"], force=fmt(mu["f0"]),
                  range=fmt(RANGE_R), lengthrange=fmt([LT + RANGE_R[0] * L0, LT + RANGE_R[1] * L0]))
  ET.indent(root, space="  ")
  out.parent.mkdir(parents=True, exist_ok=True)
  head = ("<!-- MASS (github.com/lsw9021/MASS, Apache-2.0) converted to MJCF by "
          "mocap-to-myofullbody models/mass_to_mjcf.py from data/human.xml and data/muscle284.xml -->\n")
  out.write_text(head + ET.tostring(root, encoding="unicode") + "\n")
  return nodes, muscles, rest


def check(out, nodes, muscles, rest):
  import mujoco
  m = mujoco.MjModel.from_xml_path(str(out))
  d = mujoco.MjData(m)
  print(f"compiled: bodies {m.nbody - 1}, joints {m.njnt}, nq {m.nq}, muscles {m.nu}, tendons {m.ntendon}, "
        f"mass {m.body_subtreemass[0]:.1f} kg")
  mujoco.mj_forward(m, d)  # qpos0: the rest pose
  ec, ea, ej, em = 0.0, 0.0, 0.0, []
  for n in nodes:
    g = m.geom(f"{n['name']}_box").id
    ec = max(ec, np.linalg.norm(d.geom_xpos[g] - n["c"]))
    R = d.geom_xmat[g].reshape(3, 3)
    ea = max(ea, np.degrees(np.arccos(np.clip((np.trace(R.T @ n["R"]) - 1) / 2, -1, 1))))
    ej = max(ej, np.linalg.norm(d.xpos[m.body(n["name"]).id] - n["J"]))
    gm = m.geom(f"{n['name']}_bone").id
    mid = m.geom_dataid[gm]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    w = d.geom_xpos[gm] + v @ d.geom_xmat[gm].reshape(3, 3).T
    lo, hi = w.min(0), w.max(0)
    em.append((n["name"], np.linalg.norm((lo + hi) / 2 - n["c"])))
  print(f"rest pose vs human.xml: box centres {1e3 * ec:.3g} mm, box orientations {ea:.3g} deg, "
        f"joint centres {1e3 * ej:.3g} mm (largest)")
  em.sort(key=lambda x: -x[1])
  print("bone mesh bounding-box centre vs box centre (largest 5): "
        + ", ".join(f"{n} {1e3 * e:.0f} mm" for n, e in em[:5]))
  el = max(abs(d.ten_length[m.tendon(k).id] - L) for k, L in rest.items())
  print(f"tendon lengths at rest vs waypoint polylines: largest difference {el:.2e} m; "
        f"rest lengths {min(rest.values()):.3f}-{max(rest.values()):.3f} m")

  def fibre():
    L0 = np.array([mu["lm"] * rest[mu["name"]] for mu in muscles])
    LT = np.array([mu["lt"] * rest[mu["name"]] for mu in muscles])
    return (d.actuator_length - LT) / L0
  f = fibre()
  print(f"normalised fibre length (l - LT) / L0, T-pose: {f.min():.2f}-{f.max():.2f} (MASS: (1 - lt) / lm)")
  q = m.qpos0.copy()
  for nm, v in (("ArmL_z", -1.5), ("ArmR_z", 1.5)):
    q[m.jnt_qposadr[m.joint(nm).id]] = v
  d.qpos[:] = q
  mujoco.mj_forward(m, d)
  f2 = fibre()
  i = np.argsort(f2)
  print(f"  arms down (ArmL_z = -1.5, ArmR_z = +1.5): {f2.min():.2f}-{f2.max():.2f}; shortest "
        + ", ".join(f"{muscles[k]['name']} {f2[k]:.2f}" for k in i[:3]) + "; longest "
        + ", ".join(f"{muscles[k]['name']} {f2[k]:.2f}" for k in i[-3:]))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("human")
  ap.add_argument("muscle")
  ap.add_argument("obj_dir")
  ap.add_argument("out")
  ap.add_argument("--check", action="store_true")
  a = ap.parse_args()
  out = Path(a.out).resolve()
  nodes, muscles, rest = convert(a.human, a.muscle, Path(a.obj_dir).resolve(), out)
  print(f"wrote {out}: {len(nodes)} bodies, {len(muscles)} muscles")
  if a.check:
    check(out, nodes, muscles, rest)


if __name__ == "__main__":
  sys.exit(main())
