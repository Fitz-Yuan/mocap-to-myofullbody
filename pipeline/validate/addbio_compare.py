"""Our joint angles and moments against AddBiomechanics' own, from the same .b3d file.

An AddBiomechanics .b3d (Werling et al. 2023, PLOS ONE; dataset: Werling et al. 2024) holds
the markers and force plates of each trial and what AddBiomechanics made of them with its
scaled OpenSim model (Rajagopal 2015 with arms): the joint angles after its low-pass pass
and the joint torques of its dynamics pass. read_raw.py reads the same markers and plates
(M2M_INPUT=b3d), so the two pipelines start from identical data. With M2M_MODEL=
rajagopal2016 the coordinates are the same model's (hip_flexion_r, knee_angle_r, ...); with
another model the coordinates of the same name are compared.

Per trial and coordinate: correlation, mean difference (offset) and RMS difference after
removing it; angles over every frame, moments over the frames AddBiomechanics marks as
having ground reaction forces (notMissingGRF) where our loads are not uncertain. Moments:
our anatomical moments (hip / knee extension, ankle plantarflexion, hip abduction, L5/S1
extension, elbow flexion) against AddBiomechanics' torque on the coordinate. The sign that
relates each pair is a convention; it is taken once per quantity, from the median
correlation over the trials, and reported.

  M2M_DATASET=datasets/addbiomechanics_falisse2017.toml python pipeline/validate/addbio_compare.py IK_DIR KIN_DIR
Needs nimblephysics. Writes <KIN_DIR>/addbio_compare.json and .md.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402

ANGLES = [("hip flexion", "hip_flexion_{s}"), ("hip adduction", "hip_adduction_{s}"),
          ("hip rotation", "hip_rotation_{s}"), ("knee flexion", "knee_angle_{s}"),
          ("ankle dorsiflexion", "ankle_angle_{s}"), ("subtalar", "subtalar_angle_{s}"),
          ("elbow flexion", "elbow_flex_{s}"), ("lumbar extension", "lumbar_extension")]
MOMENTS = [("hip extension", "hip_flexion_{s}", "hip_{s}_ext"),
           ("hip abduction", "hip_adduction_{s}", "hip_{s}_abd"),
           ("knee extension", "knee_angle_{s}", "knee_{s}_ext"),
           ("ankle plantarflexion", "ankle_angle_{s}", "ankle_{s}_plantar"),
           ("elbow flexion", "elbow_flex_{s}", "elbow_{s}_flex"),
           ("L5/S1 extension", "lumbar_extension", "L5S1_ext")]
LP_PASS, DYN_PASS = "LOW_PASS_FILTER", "DYNAMICS"


def stats(a, b):
  ok = np.isfinite(a) & np.isfinite(b)
  if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
    return None
  a, b = a[ok], b[ok]
  off = float(np.mean(a - b))
  return dict(r=float(np.corrcoef(a, b)[0, 1]), offset=off,
              rms=float(np.sqrt(np.mean((a - b - off) ** 2))), n=int(ok.sum()))


def addbio(path):
  """AddBiomechanics' angles (low-pass pass) and torques (dynamics pass) of one trial
  ("<file>.b3d#<trial>"): {coordinate: [T]}, {coordinate: [T]}, frames with GRF [T]."""
  import nimblephysics as nimble
  file, _, trial = path.partition("#")
  sub = nimble.biomechanics.SubjectOnDisk(file)
  t = [sub.getTrialName(i) for i in range(sub.getNumTrials())].index(trial)
  kinds = [str(sub.getProcessingPassType(i)).split(".")[-1] for i in range(sub.getNumProcessingPasses())]
  lp, dyn = kinds.index(LP_PASS), kinds.index(DYN_PASS)
  skel = sub.readSkel(dyn, ignoreGeometry=True)
  dofs = [skel.getDofByIndex(i).getName() for i in range(skel.getNumDofs())]
  T = sub.getTrialLength(t)
  fr = sub.readFrames(t, 0, T, includeSensorData=False, includeProcessingPasses=True)
  pos = np.array([np.ravel(f.processingPasses[lp].pos) for f in fr])
  tau = np.array([np.ravel(f.processingPasses[dyn].tau) for f in fr])
  grf = np.array([str(x).endswith("notMissingGRF") for x in sub.getMissingGRF(t)])
  return dict(zip(dofs, pos.T)), dict(zip(dofs, tau.T)), grf


def series(name, path, ik_dir, kin_dir):
  """{"quantity | coordinate": (ours [T], AddBiomechanics' [T])}, NaN outside the compared frames."""
  K = np.load(os.path.join(ik_dir, f"{name}.npz"))
  kn = np.load(os.path.join(kin_dir, f"{name}.npz"))
  jn = list(K["joint_names"])
  ours = {j: K["qpos"][:, 7 + i - 1] for i, j in enumerate(jn) if i > 0}  # root (free joint) first
  pos, tau, grf = addbio(path)
  if len(K["qpos"]) != len(grf):
    raise SystemExit(f"{name}: {len(K['qpos'])} IK frames but {len(grf)} in the .b3d")
  ok = grf & ~kn["uncertain"].astype(bool)
  out = {}
  for side in ("r", "l"):
    for q, coord in ANGLES:
      c = coord.format(s=side)
      if c in ours and c in pos:
        out[f"{q} | {c}"] = (np.degrees(ours[c]), np.degrees(pos[c]))
    for q, coord, key in MOMENTS:
      c, k = coord.format(s=side), key.format(s=side)
      if k in kn.files and c in tau:
        out[f"{q} moment | {c}"] = (np.where(ok, kn[k], np.nan), np.where(ok, tau[c], np.nan))
  return out


def main():
  config.usage(__doc__)
  ik_dir, kin_dir = sys.argv[1], sys.argv[2]
  trials, _ = config.raw_trials()
  S = {n: series(n, p, ik_dir, kin_dir) for n, p in trials.items()
       if os.path.exists(os.path.join(ik_dir, f"{n}.npz")) and os.path.exists(os.path.join(kin_dir, f"{n}.npz"))}
  keys = sorted({k for v in S.values() for k in v})
  sign = {}
  for k in keys:
    r = [st["r"] for v in S.values() if k in v and (st := stats(*v[k])) is not None]
    sign[k] = 1.0 if not r or np.median(r) >= 0 else -1.0
  res = {n: {k: st for k, (a, b) in v.items() if (st := stats(a, sign[k] * b)) is not None}
         for n, v in S.items()}
  groups = {}
  for n in res:
    groups.setdefault(n.split("-", 1)[1].split("_")[0], []).append(n)  # task: Gait, STS, ...
  summ = {}
  for task, names in sorted(groups.items()):
    for k in keys:
      v = [res[n][k] for n in names if k in res[n]]
      if v:
        summ[f"{task} | {k}"] = dict(trials=len(v), sign=sign[k],
                                     **{m: round(float(np.median([x[m] for x in v])), 3) for m in ("r", "offset", "rms")})
  json.dump(dict(per_trial=res, median=summ), open(os.path.join(kin_dir, "addbio_compare.json"), "w"), indent=1)
  lines = ["Ours vs AddBiomechanics (same .b3d), median over trials", "",
           "| task | quantity | coordinate | trials | r | offset | RMS after offset | sign |",
           "|---|---|---|---|---|---|---|---|"]
  for k, v in summ.items():
    task, q, c = (x.strip() for x in k.split("|"))
    unit = "N m" if q.endswith("moment") else "deg"
    lines.append(f"| {task} | {q} | {c} | {v['trials']} | {v['r']:.3f} | {v['offset']:+.1f} {unit} | "
                 f"{v['rms']:.1f} {unit} | {'+' if v['sign'] > 0 else '-'} |")
  open(os.path.join(kin_dir, "addbio_compare.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
