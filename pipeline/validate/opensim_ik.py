"""OpenSim baseline for our IK: the standard Scale -> Marker Placer -> IK workflow on
exactly the markers our pipeline uses (<M2M_NPZ_DIR>: raw takes, label fixes,
mislabel rejection, pelvis refill), so the two methods can be scored side by side
(opensim_compare.py).

Inputs in <M2M_OSIM_SETUP_DIR> (default <M2M_ROOT>/opensim_setup/): Model.osim (the
Rajagopal 2016 full-body model), markers.xml (an OpenSim MarkerSet with the marker set's
labels, mocap2myo/markerset.py) and a ScaleTool setup (M2M_OSIM_SCALE_SETUP, a file
name in that folder or a path). One configuration for every subject: the setup's
scaling measurements (case study: pelvis width, ASIS-knee, knee-ankle, shoulder-elbow,
elbow-wrist, wrist-finger, shoulder width; feet not scaled) and the marker weights of
its Marker Placer, with the markers it switches off (case study: trunk, wands, back of
head) switched on at low weight (EXTRA_W). The script sets the model and marker files,
the subject's mass and height (anthropometrics), the marker file and time range: the
static standing trial, or 1 s of quiet standing from the subject's first trial when
it has none.

Outputs per trial in opensim_ik/<subject>/: <trial>.mot (coordinates, deg), <trial>_markers.npz
(model marker positions [T, 39, 3] in the IK frame, Z up, m), <trial>_bodies.npz (body
origins [T, nb, 3] of pelvis / femur / tibia / talus / torso, Z up, m) and timing.

  python pipeline/validate/opensim_ik.py [subject ...]     (needs opensim, from conda)
"""

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config as _cfg  # noqa: E402

ms = _cfg.marker_set()  # the dataset's marker set (its labels are the OpenSim markers')
from pipeline import config  # noqa: E402

NPZ = config.NPZ_DIR
SRC = config.OSIM_SETUP_DIR
OUT = config.OSIM_DIR
TEMPLATE = config.OSIM_SCALE_SETUP
# switched off in the scale setup -> on at low weight here
EXTRA_W = {"C7": 5, "CLAV": 5, "T10": 5, "STRN": 5, "RBAK": 1, "LBHD": 1, "RBHD": 1,
           "LUPA": 1, "RUPA": 1, "LFRM": 1, "RFRM": 1, "LTHI": 1, "RTHI": 1, "LTIB": 1, "RTIB": 1}
BODIES = ["pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l", "talus_r", "talus_l", "torso"]


def to_yup(P):
  """Z-up (x, y, z) -> OpenSim Y-up (x, z, -y)."""
  return np.stack([P[..., 0], P[..., 2], -P[..., 1]], -1)


def to_zup(P):
  return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def write_trc(path, P, labels, rate):
  T = len(P)
  with open(path, "w") as f:
    f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}\n")
    f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
    f.write(f"{rate:g}\t{rate:g}\t{T}\t{len(labels)}\tm\t{rate:g}\t1\t{T}\n")
    f.write("Frame#\tTime\t" + "\t\t\t".join(labels) + "\t\t\n")
    f.write("\t\t" + "\t".join(f"X{k + 1}\tY{k + 1}\tZ{k + 1}" for k in range(len(labels))) + "\n\n")
    for t in range(T):
      # NaN, not an empty field: OpenSim drops trailing empty fields and then
      # rejects the row when the last markers are missing
      v = ["NaN" if np.isnan(x) else f"{x:.6f}" for x in P[t].ravel()]
      f.write(f"{t + 1}\t{t / rate:.5f}\t" + "\t".join(v) + "\n")


def quiet_window(P, rate, labels, secs=1.0):
  """First quiet upright standing window (pelvis and ankles < 5 cm/s, pelvis within
  3 cm of its upright height) of secs seconds -> frame slice (no static trial)."""
  pel = np.nanmean(P[:, [labels.index(l) for l in ms.PELVIS_LABELS]], 1)
  ank = P[:, [labels.index(l) for l in ms.ANKLES]]
  sp = lambda x: np.r_[np.inf, np.linalg.norm(np.diff(x, axis=0), axis=-1).reshape(len(x) - 1, -1).max(1) * rate]
  ok = (sp(pel) < 0.05) & (sp(ank) < 0.05) & (pel[:, 2] > np.nanpercentile(pel[:, 2], 95) - 0.03)
  w = int(secs * rate)
  run = np.convolve(ok.astype(float), np.ones(w), "valid") >= w
  i = int(np.argmax(run))
  return slice(i, i + w)


def scale_subject(subj, labels, mass, height_m):
  import opensim as osim
  d = os.path.join(OUT, subj)
  os.makedirs(d, exist_ok=True)
  st = os.path.join(NPZ, "static", f"{subj}.npz")
  if os.path.exists(st):
    D = np.load(st)
    P, rate, src = D["markers"].astype(float), float(D["marker_rate"]), "static trial"
  else:  # quiet standing in the subject's first trial
    first = [t for t in config.trials(NPZ) if config.subject_of(t) == subj][0]
    D = np.load(os.path.join(NPZ, f"{first}.npz"))
    P, rate = D["markers"].astype(float), float(D["marker_rate"])
    P, src = P[quiet_window(P, rate, labels)], f"quiet standing in {first}"
  trc = os.path.join(d, "static.trc")
  write_trc(trc, to_yup(P), labels, rate)
  t1 = (len(P) - 1) / rate
  tool = osim.ScaleTool(TEMPLATE)
  tool.setPathToSubject("")  # otherwise every file name is joined onto the setup's folder
  tool.setName(subj)
  tool.setSubjectMass(mass)
  tool.setSubjectHeight(1000 * height_m)
  gm = tool.getGenericModelMaker()
  gm.setModelFileName(os.path.join(SRC, "Model.osim"))
  gm.setMarkerSetFileName(os.path.join(SRC, "markers.xml"))
  rng = osim.ArrayDouble()
  rng.append(0.0)
  rng.append(t1)
  ms = tool.getModelScaler()
  ms.setApply(True)
  ms.setMarkerFileName(trc)
  ms.setTimeRange(rng)
  ms.setPreserveMassDist(True)
  ms.setOutputModelFileName(os.path.join(d, "scaled.osim"))
  ms.setOutputScaleFileName(os.path.join(d, "scale_factors.xml"))
  mpl = tool.getMarkerPlacer()
  mpl.setApply(True)
  mpl.setMarkerFileName(trc)
  mpl.setTimeRange(rng)
  mpl.setOutputModelFileName(os.path.join(d, "scaled_placed.osim"))
  mpl.setOutputMotionFileName(os.path.join(d, "static_pose.mot"))
  ts = mpl.getIKTaskSet()
  for i in range(ts.getSize()):
    t = ts.get(i)
    name = t.getName()
    if not t.getApply():
      t.setApply(True)
      t.setWeight(EXTRA_W.get(name, 1))
  ok = tool.run()
  return dict(subject=subj, ok=bool(ok), static=src, mass=mass)


def task_weights():
  import opensim as osim
  tool = osim.ScaleTool(TEMPLATE)
  ts = tool.getMarkerPlacer().getIKTaskSet()
  w = {}
  for i in range(ts.getSize()):
    t = ts.get(i)
    w[t.getName()] = t.getWeight() if t.getApply() else EXTRA_W.get(t.getName(), 1)
  return w


def ik_trial(trial):
  import opensim as osim
  subj = config.subject_of(trial)
  d = os.path.join(OUT, subj)
  D = np.load(os.path.join(NPZ, f"{trial}.npz"))
  labels = [str(x) for x in D["marker_labels"]]
  P, rate = D["markers"].astype(float), float(D["marker_rate"])
  trc = os.path.join(d, f"{trial}.trc")
  write_trc(trc, to_yup(P), labels, rate)
  model = osim.Model(os.path.join(d, "scaled_placed.osim"))
  model.initSystem()
  ik = osim.InverseKinematicsTool()
  ik.setName(trial)
  ik.setModel(model)
  ts = ik.getIKTaskSet()
  for name, w in task_weights().items():
    t = osim.IKMarkerTask()
    t.setName(name)
    t.setApply(True)
    t.setWeight(float(w))
    ts.cloneAndAppend(t)
  ik.setMarkerDataFileName(trc)
  ik.setStartTime(0.0)
  ik.setEndTime((len(P) - 1) / rate)
  ik.setOutputMotionFileName(os.path.join(d, f"{trial}.mot"))
  ik.setResultsDir(d)
  ik.set_report_errors(True)
  t0 = time.time()
  ok = ik.run()
  dt = time.time() - t0
  # model marker positions and body origins per frame, from the solved coordinates
  sto = osim.TimeSeriesTable(os.path.join(d, f"{trial}.mot"))
  cols = list(sto.getColumnLabels())
  M = sto.getMatrix().to_numpy()
  state = model.initSystem()
  cs = model.getCoordinateSet()
  coords = [cs.get(c) for c in cols]
  rot = [c.getMotionType() == 1 for c in coords]  # 1 = rotational: .mot in degrees
  mset = model.getMarkerSet()
  mk = [mset.get(l) for l in labels]
  bodies = [model.getBodySet().get(b) for b in BODIES]
  T = len(M)
  MK = np.full((T, len(labels), 3), np.nan)
  BO = np.full((T, len(bodies), 3), np.nan)
  for i in range(T):
    for c, v, r in zip(coords, M[i], rot):
      c.setValue(state, np.radians(v) if r else v, False)
    model.realizePosition(state)
    for k, m_ in enumerate(mk):
      p = m_.getLocationInGround(state)
      MK[i, k] = [p.get(0), p.get(1), p.get(2)]
    for k, b in enumerate(bodies):
      p = b.getPositionInGround(state)
      BO[i, k] = [p.get(0), p.get(1), p.get(2)]
  np.savez_compressed(os.path.join(d, f"{trial}_markers.npz"), markers=to_zup(MK), labels=np.array(labels))
  np.savez_compressed(os.path.join(d, f"{trial}_bodies.npz"), origins=to_zup(BO), bodies=np.array(BODIES))
  return trial, bool(ok), dt, T


def main():
  config.usage(__doc__)
  need = [os.path.join(SRC, "Model.osim"), os.path.join(SRC, "markers.xml"), TEMPLATE]
  missing = [p for p in need if not os.path.exists(p)]
  if missing:
    raise SystemExit(f"OpenSim setup missing: {missing} (see the README, OpenSim comparison)")
  all_trials = config.trials(NPZ)
  subjects = sys.argv[1:] or sorted({config.subject_of(t) for t in all_trials})
  anthro = {s: (a["mass_kg"], a["height_cm"] / 100) for s, a in config.anthropometrics().items()}
  labels = [str(x) for x in np.load(os.path.join(NPZ, f"{all_trials[0]}.npz"))["marker_labels"]]
  os.makedirs(OUT, exist_ok=True)
  log = {}
  t0 = time.time()
  ctx = mp.get_context("spawn")
  with ctx.Pool(min(len(subjects), config.workers())) as pool:
    res = pool.starmap(scale_subject, [(s, labels, *anthro[s]) for s in subjects])
  for r in res:
    print("scale", r, flush=True)
    log[r["subject"]] = r
  t_scale = time.time() - t0
  trials = [t for t in all_trials if config.subject_of(t) in subjects]
  with ctx.Pool(min(len(trials), config.workers())) as pool:
    for trial, ok, dt, T in pool.imap_unordered(ik_trial, trials):
      print(f"ik {trial}: ok={ok} {T} frames in {dt:.0f} s ({T / max(dt, 1e-9):.0f} frames/s)", flush=True)
      log[trial] = dict(ok=ok, seconds=dt, frames=T)
  log["_wall_s"] = dict(scale=t_scale, total=time.time() - t0)
  json.dump(log, open(os.path.join(OUT, "run.json"), "w"), indent=1)


if __name__ == "__main__":
  main()
