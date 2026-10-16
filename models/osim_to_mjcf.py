"""OpenSim model -> MuJoCo MJCF with MyoConverter (run by models/fetch.py).

  PY_MYOCONVERTER models/osim_to_mjcf.py --myoconverter SRC [--steps 1[,2,3]] OSIM GEOMETRY OUT_XML

Runs MyoConverter's O2MPipeline (github.com/MyoHub/myoconverter, Apache-2.0) on OSIM with
the meshes in GEOMETRY, into the folder of OUT_XML (MyoConverter names its output
<osim name>_cvt<step>.xml; OUT_XML is that file, or <osim name>_cvt<step>_m2m.xml for a
copy with the joint-frame fix below). Step 1 converts the model (bodies,
joints with their coupled coordinates as joint equalities, moving / conditional path
points as small bodies driven by equalities, muscles with their OpenSim parameters) and
checks the forward kinematics of the model's markers against OpenSim (validation, written
to Step1_xmlConvert/end_points); steps 2 and 3 optimise the muscle paths (moment arms) and
force parameters and take much longer (minutes to hours per model).

Needs a Python with MyoConverter's dependencies (its conda_env.yml: opensim 4.4.1,
mujoco 2.3.7, numpy 1.21, ...); SRC is the fetched myoconverter checkout. The output is
MuJoCo 2.3 XML: the profiles patch the one attribute MuJoCo 3 rejects (option collision).
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def polynomials_to_splines(osim: Path, out: Path, n=41) -> int:
  """Copy of the model with every PolynomialFunction of a joint transform replaced by a
  SimmSpline through n samples over its coordinate's range. Returns how many."""
  tree = ET.parse(osim)
  coords = {c.get("name"): [float(v) for v in c.findtext("range").split()]
            for c in tree.iter("Coordinate") if c.find("range") is not None}
  try:
    import opensim
  except ImportError:
    opensim = None
  k = 0
  for ta in tree.iter("TransformAxis"):
    for f in list(ta):
      if f.tag != "PolynomialFunction":
        continue
      c = np.array([float(v) for v in f.findtext("coefficients").split()])
      lo, hi = coords[ta.findtext("coordinates").strip()]
      x = np.linspace(lo, hi, n)
      y = np.polyval(c, x)  # OpenSim / SimTK: highest power first
      if opensim is not None:
        pf = opensim.PolynomialFunction(opensim.Vector(list(c)))
        yo = np.array([pf.calcValue(opensim.Vector(1, float(v))) for v in x])
        if not np.allclose(y, yo, atol=1e-12):
          raise SystemExit(f"polynomial evaluation differs from OpenSim's: {np.abs(y - yo).max()}")
      sp = ET.Element("SimmSpline", name=f.get("name", "function"))
      ET.SubElement(sp, "x").text = " ".join(f"{v:.12g}" for v in x)
      ET.SubElement(sp, "y").text = " ".join(f"{v:.12g}" for v in y)
      i = list(ta).index(f)
      ta.remove(f)
      ta.insert(i, sp)
      k += 1
  tree.write(out, encoding="UTF-8", xml_declaration=True)
  return k


def fix_child_offsets(osim: Path, mjcf: Path, out: Path) -> list[str]:
  """MyoConverter puts a joint's rotation axes through the child body's origin; OpenSim
  rotates about the child joint frame, which some joints place away from that origin
  (Rajagopal's walker knee: 8.9 mm). Write a copy of the MJCF with pos = that offset on
  every joint of such a child body (in its frame; MyoConverter keeps OpenSim's body
  frames). Returns the bodies changed."""
  tree = ET.parse(osim)
  offs = {}
  for js in tree.iter("JointSet"):
    for j in js.find("objects"):
      frames = {f.get("name"): f for f in j.iter("PhysicalOffsetFrame")}
      cf = frames.get(j.findtext("socket_child_frame"))
      if cf is None:
        continue
      t = np.array(cf.findtext("translation").split(), float)
      if np.linalg.norm(t) > 1e-9:
        offs[cf.findtext("socket_parent").split("/")[-1]] = t
  mt = ET.parse(mjcf, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))  # keep the credits
  done = []
  for b in mt.iter("body"):
    if b.get("name") in offs:
      for jn in b.findall("joint"):
        if np.linalg.norm(np.array((jn.get("pos") or "0 0 0").split(), float)) > 0:
          raise SystemExit(f"{jn.get('name')}: already has a pos")
        jn.set("pos", " ".join(f"{v:.9g}" for v in offs[b.get("name")]))
      done.append(b.get("name"))
  root = mt.getroot()
  root.insert(0, ET.Comment(f" mocap-to-myofullbody models/osim_to_mjcf.py: joints of {done} moved to the "
                            "child joint frame origin (MyoConverter put them at the body origin) "))
  mt.write(out, encoding="unicode")
  return done


def validate(osim: Path, mjcf: Path, n=60, seed=0) -> dict:
  """Body origins of the converted model vs OpenSim over random poses of the independent
  coordinates (within their ranges; OpenSim assembles its constraints, MuJoCo applies its
  joint equalities). The two worlds are aligned once (rigid fit at the default pose)."""
  import mujoco
  import opensim
  rng = np.random.default_rng(seed)
  om = opensim.Model(str(osim))
  st = om.initSystem()
  m = mujoco.MjModel.from_xml_path(str(mjcf))  # MuJoCo 2.3 reads MyoConverter's output as it is
  d = mujoco.MjData(m)
  cs = om.getCoordinateSet()
  free = []
  for i in range(cs.getSize()):
    c = cs.get(i)
    if c.get_locked() or c.isDependent(st) or c.getName().startswith("pelvis_"):
      continue
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, c.getName()) < 0:
      continue
    free.append(c)
  names = [om.getBodySet().get(i).getName() for i in range(om.getBodySet().getSize())]
  names = [b for b in names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0]

  def pose(vals):
    for c, v in zip(free, vals):
      c.setValue(st, float(v), False)
    om.assemble(st)
    om.realizePosition(st)
    P_os = np.array([[om.getBodySet().get(b).getPositionInGround(st).get(k) for k in range(3)] for b in names])
    q = m.qpos0.copy()
    if m.nkey:
      q[:] = m.key_qpos[0]
    for c in free:
      j = m.joint(c.getName()).id
      q[m.jnt_qposadr[j]] = c.getValue(st)
    for e in range(m.neq):  # joint equalities, as mocap2myo.kinematics.apply_joint_equalities
      if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
        continue
      a1 = m.jnt_qposadr[m.eq_obj1id[e]]
      co = m.eq_data[e][:5]
      x = q[m.jnt_qposadr[m.eq_obj2id[e]]] - m.qpos0[m.jnt_qposadr[m.eq_obj2id[e]]] if m.eq_obj2id[e] >= 0 else 0.0
      q[a1] = m.qpos0[a1] + co[0] + co[1] * x + co[2] * x ** 2 + co[3] * x ** 3 + co[4] * x ** 4
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    P_mj = np.array([d.xpos[m.body(b).id] for b in names])
    return P_os, P_mj

  A, B = pose([c.getDefaultValue() for c in free])
  ca, cb = A.mean(0), B.mean(0)
  U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
  D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
  R = Vt.T @ D @ U.T
  err = []
  for _ in range(n):
    vals = [c.getRangeMin() + rng.random() * (c.getRangeMax() - c.getRangeMin()) for c in free]
    A, B = pose(vals)
    err.append(np.linalg.norm((A - ca) @ R.T + cb - B, axis=1))
  err = np.array(err)
  worst = np.argsort(-err.max(0))[:4]
  return dict(coordinates=len(free), bodies=len(names), poses=n, mean_mm=1e3 * err.mean(),
              p95_mm=1e3 * np.percentile(err, 95), max_mm=1e3 * err.max(),
              worst={names[i]: round(1e3 * err[:, i].max(), 1) for i in worst})


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("osim")
  ap.add_argument("geometry")
  ap.add_argument("out_xml")
  ap.add_argument("--myoconverter", required=True, help="the fetched myoconverter repository")
  ap.add_argument("--steps", default="1")
  ap.add_argument("--no-validation", action="store_true")
  ap.add_argument("--validate-only", action="store_true", help="only compare OUT_XML with OSIM")
  a = ap.parse_args()
  sys.path.insert(0, str(Path(a.myoconverter).resolve()))
  os.environ.setdefault("MUJOCO_GL", "egl")
  from myoconverter.O2MPipeline import O2MPipeline

  out = Path(a.out_xml).resolve()
  steps = [int(s) for s in a.steps.split(",")]
  expect = f"{Path(a.osim).name[:-5]}_cvt{max(steps)}"
  if out.name not in (expect + ".xml", expect + "_m2m.xml"):
    raise SystemExit(f"MyoConverter writes {expect}.xml; give that (or {expect}_m2m.xml, the copy with "
                     f"the joint-frame fix) as OUT_XML (got {out.name})")
  raw = out.parent / (expect + ".xml")
  out.parent.mkdir(parents=True, exist_ok=True)
  osim = Path(a.osim).resolve()
  if a.validate_only:
    print(validate(osim, out))
    return
  if "PolynomialFunction" in osim.read_text():
    pre = out.parent / osim.name
    k = polynomials_to_splines(osim, pre)
    print(f"{k} PolynomialFunction transforms -> SimmSpline: {pre}")
    osim = pre
  t = time.time()
  O2MPipeline(str(osim), str(Path(a.geometry).resolve()), str(out.parent),
              convert_steps=steps, muscle_list=None, osim_data_overwrite=True, conversion=True,
              validation=not a.no_validation, speedy=False, generate_pdf=False, add_ground_geom=False,
              treat_as_normal_path_point=False)
  print(f"converted in {time.time() - t:.0f} s: {raw} {'ok' if raw.exists() else 'MISSING'}")
  files = [raw]
  if out != raw:
    print(f"joint frame fix: {fix_child_offsets(osim, raw, out)} -> {out}")
    files.append(out)
  if not a.no_validation:
    import json
    for f in files:
      v = validate(osim, f)
      print(f"{f.name}: body origins vs OpenSim over {v['poses']} random poses of {v['coordinates']} coordinates: "
            f"mean {v['mean_mm']:.2f} mm, 95th pct {v['p95_mm']:.2f} mm, max {v['max_mm']:.2f} mm {v['worst']}")
      (out.parent / f"{f.stem}_validation.json").write_text(json.dumps(v, indent=1) + "\n")


if __name__ == "__main__":
  main()
