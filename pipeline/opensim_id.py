"""OpenSim side of the comparison (optional; run_all.sh --opensim): for trials solved by
the OpenSim IK baseline (validate/opensim_ik.py: Scale + Marker Placer + IK with the
Rajagopal 2016 model), the joint centres per frame (body origins, drawn by the viewer as
a stick figure) and OpenSim Inverse Dynamics with the same external loads as our
kinetics (pipeline/kinetics.py): each hand's share of the carried load at our palm
points, and the ground reaction under each foot at our estimated centres of pressure,
plus the same worn equipment mass on the torso.

Outputs: <OSIM_ID_DIR>/<trial>.npz (joints [T, nb, 3] Z up, lab frame, m; bodies; links;
angle_<coordinate> in deg; moment_<our name> in N m with our sign conventions) and the
ID's .sto / load files.
  python pipeline/opensim_id.py [trial ...]          (needs opensim, from conda)
"""

import os
import sys

import numpy as np
import opensim as osim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import config  # noqa: E402

osim.Logger.setLevelString("error")

BODIES = ["pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r", "femur_l", "tibia_l",
          "talus_l", "calcn_l", "toes_l", "torso", "humerus_r", "ulna_r", "radius_r", "hand_r",
          "humerus_l", "ulna_l", "radius_l", "hand_l"]
# angle / moment columns -> (our name, sign): our conventions (hip/knee extension and ankle
# plantarflexion moments positive; angles as the coordinates)
ANGLES = {"hip_flexion_r": "hip_flexion_r", "hip_flexion_l": "hip_flexion_l",
          "hip_adduction_r": "hip_adduction_r", "hip_adduction_l": "hip_adduction_l",
          "knee_angle_r": "knee_angle_r", "knee_angle_l": "knee_angle_l",
          "ankle_angle_r": "ankle_angle_r", "ankle_angle_l": "ankle_angle_l",
          "elbow_flex_r": "elbow_flex_r", "elbow_flex_l": "elbow_flex_l",
          "lumbar_extension": "lumbar_extension"}
MOMENTS = {"lumbar_extension_moment": ("L5S1_ext", 1),
           "hip_flexion_r_moment": ("hip_r_ext", -1), "hip_flexion_l_moment": ("hip_l_ext", -1),
           "hip_adduction_r_moment": ("hip_r_abd", -1), "hip_adduction_l_moment": ("hip_l_abd", -1),
           "knee_angle_r_moment": ("knee_r_ext", -1), "knee_angle_l_moment": ("knee_l_ext", -1),
           "ankle_angle_r_moment": ("ankle_r_plantar", -1), "ankle_angle_l_moment": ("ankle_l_plantar", -1),
           "elbow_flex_r_moment": ("elbow_r_flex", 1), "elbow_flex_l_moment": ("elbow_l_flex", 1),
           "arm_flex_r_moment": ("shoulder_r_flex", 1), "arm_flex_l_moment": ("shoulder_l_flex", 1)}


def to_yup(P):
  return np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1)


def to_zup(P):
  return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def read_table(path):
  L = open(path).read().splitlines()
  i = next(k for k, l in enumerate(L) if l.startswith("time"))
  cols = L[i].split("\t")
  A = np.array([[float(x) for x in l.split("\t")] for l in L[i + 1:] if l.strip()])
  return cols, A


def write_sto(path, t, cols, A):
  with open(path, "w") as f:
    f.write(f"{os.path.basename(path)}\nversion=1\nnRows={len(t)}\nnColumns={len(cols) + 1}\n"
            "inDegrees=no\nendheader\n")
    f.write("time\t" + "\t".join(cols) + "\n")
    for i in range(len(t)):
      f.write(f"{t[i]:.5f}\t" + "\t".join(f"{x:.6f}" for x in A[i]) + "\n")


def joint_centres(model, cols, A):
  """Body origins in ground for every row of a coordinates table (degrees)."""
  state = model.initSystem()
  cs = model.getCoordinateSet()
  coords = []
  for k in range(cs.getSize()):
    c = cs.get(k)
    if c.getName() in cols:
      rot = c.getMotionType() == 1  # Rotational
      coords.append((c, cols.index(c.getName()), rot))
  bodies = [model.getBodySet().get(b) for b in BODIES]
  out = np.zeros((len(A), len(BODIES), 3))
  for t in range(len(A)):
    for c, j, rot in coords:
      c.setValue(state, np.radians(A[t, j]) if rot else A[t, j], False)
    model.realizePosition(state)
    for i, b in enumerate(bodies):
      p = b.getPositionInGround(state)
      out[t, i] = (p.get(0), p.get(1), p.get(2))
  return to_zup(out)


def links(model):
  idx = {b: i for i, b in enumerate(BODIES)}
  out = []
  for k in range(model.getJointSet().getSize()):
    j = model.getJointSet().get(k)
    p = j.getParentFrame().findBaseFrame().getName()
    c = j.getChildFrame().findBaseFrame().getName()
    if p in idx and c in idx:
      out.append((idx[p], idx[c]))
  return out


def inverse_dynamics(model_path, mot, kin, out, trial, t):
  F = kin["hand_force"]  # [T, 2, 3] (l, r) box on each hand
  P = kin["palm"]
  Th = kin["hand_couple"] if "hand_couple" in kin else np.zeros_like(F)
  g, share, fcop, free = kin["grf"], kin["foot_share"], kin["foot_cop"], kin["grf_free_moment"]
  cols, data, forces = [], [], []
  for name, body, f, p, tq in (
      ("hand_l", "hand_l", F[:, 0], P[:, 0], Th[:, 0]), ("hand_r", "hand_r", F[:, 1], P[:, 1], Th[:, 1]),
      ("foot_l", "calcn_l", share[:, 0, None] * g, fcop[:, 0], share[:, 0, None] * free[:, None] * [0, 0, 1]),
      ("foot_r", "calcn_r", share[:, 1, None] * g, fcop[:, 1], share[:, 1, None] * free[:, None] * [0, 0, 1])):
    cols += [f"{name}_v{a}" for a in "xyz"] + [f"{name}_p{a}" for a in "xyz"] + [f"{name}_t{a}" for a in "xyz"]
    data += [to_yup(f), to_yup(np.nan_to_num(p)), to_yup(tq)]
    forces.append((name, body))
  ext = os.path.join(out, f"{trial}_loads.sto")
  write_sto(ext, t, cols, np.concatenate(data, 1))
  loads = osim.ExternalLoads()
  for name, body in forces:
    ef = osim.ExternalForce()
    ef.setName(name)
    ef.set_applied_to_body(body)
    ef.set_force_expressed_in_body("ground")
    ef.set_point_expressed_in_body("ground")
    ef.set_force_identifier(f"{name}_v")
    ef.set_point_identifier(f"{name}_p")
    ef.set_torque_identifier(f"{name}_t")
    ef.set_data_source_name(ext)
    loads.cloneAndAppend(ef)
  loads.setDataFileName(ext)
  lx = os.path.join(out, f"{trial}_loads.xml")
  loads.printToXML(lx)
  model = osim.Model(model_path)  # keep a reference: the tool does not own it
  extra = float(kin["extra_kg"]) if "extra_kg" in kin else 0.0
  if extra:  # carried mass (e.g. a backpack), lumped at the torso centre of mass as in ours
    torso = model.updBodySet().get("torso")
    torso.setMass(torso.getMass() + extra)
  tool = osim.InverseDynamicsTool()
  tool.setModel(model)
  tool.setCoordinatesFileName(mot)
  tool.setLowpassCutoffFrequency(6.0)
  tool.setExternalLoadsFileName(lx)
  tool.setStartTime(float(t[0]))
  tool.setEndTime(float(t[-1]))
  tool.setResultsDir(out)
  tool.setOutputGenForceFileName(f"{trial}_id.sto")
  ex = osim.ArrayStr()
  ex.append("Muscles")
  tool.setExcludedForces(ex)
  if not tool.run():
    raise RuntimeError(f"{trial}: OpenSim ID failed")
  return read_table(os.path.join(out, f"{trial}_id.sto"))


def main():
  config.usage(__doc__)
  out = config.OSIM_ID_DIR
  os.makedirs(out, exist_ok=True)
  trials = sys.argv[1:] or [t for t in config.trials(config.KIN_DIR)
                            if os.path.exists(os.path.join(config.OSIM_DIR, config.subject_of(t), f"{t}.mot"))]
  for trial in trials:
    d = os.path.join(config.OSIM_DIR, config.subject_of(trial))
    model_path = os.path.join(d, "scaled_placed.osim")
    mot = os.path.join(d, f"{trial}.mot")
    kin = np.load(os.path.join(config.KIN_DIR, f"{trial}.npz"))
    cols, A = read_table(mot)
    model = osim.Model(model_path)
    J = joint_centres(model, cols, A)
    rate = float(kin["rate"])
    n = min(len(A), len(kin["grf"]))
    t = np.arange(n) / rate
    idc, M = inverse_dynamics(model_path, mot, _Slice(kin, n), out, trial, t)
    res = dict(joints=J.astype(np.float32), bodies=np.array(BODIES), links=np.array(links(model)), rate=rate)
    for c, nm in ANGLES.items():
      if c in cols:
        res[f"angle_{nm}"] = A[:, cols.index(c)].astype(np.float32)
    for c, (nm, sgn) in MOMENTS.items():
      if c in idc:
        res[f"moment_{nm}"] = (sgn * M[:, idc.index(c)]).astype(np.float32)
    np.savez_compressed(os.path.join(out, f"{trial}.npz"), **res)
    print(trial, "joints", J.shape, "moments", sum(k.startswith("moment_") for k in res), flush=True)


class _Slice:
  """npz view cut to the first n frames (the OpenSim IK may stop a frame early)."""

  def __init__(self, z, n):
    self.z, self.n, self.files = z, n, z.files

  def __getitem__(self, k):
    v = self.z[k]
    return v[:self.n] if v.ndim and len(v) >= self.n else v

  def __contains__(self, k):
    return k in self.files


if __name__ == "__main__":
  main()
