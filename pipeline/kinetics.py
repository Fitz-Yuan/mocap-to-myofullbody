"""Inverse dynamics: net joint forces and moments from the IK (after the reference).

For every trial in <M2M_IK_DIR>: MyoFullBody at the subject's size and mass (the
anthropometrics; equipment worn: M2M_CARRIED_COLUMN), the task's external loads on the
body (box_lift: the box on the hands while it is carried, tasks/box_lift/loads.py;
M2M_TASK=none: no external load but the ground), and mocap2myo/kinetics.py: L5/S1,
shoulders and elbows top-down (no ground reaction needed), the total ground reaction
with its centre of pressure, and hips / knees / ankles with that reaction shared
between the feet (estimated in double support). Positions are low-passed at
M2M_KIN_LOWPASS_HZ (default 6 Hz) before they are differentiated.

Frames the IK did not solve (unsolved_frames in the IK output; for an older IK file,
the frames whose marker errors are all NaN) are interpolated between the solved
neighbours (held at the ends) and flagged.

Outputs: <M2M_KIN_DIR>/<trial>.npz (keys: see mocap2myo/kinetics.py; "unsolved" marks
the frames the IK did not solve, "uncertain" those and the frames whose top-down
loads are not determined) and summary.json.
  python pipeline/kinetics.py [trial ...]
"""

import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mocap2myo import ik, kinetics  # noqa: E402
from pipeline import config  # noqa: E402

KIN = None


def unsolved_frames(K):
  """[T] bool: the frames the IK did not solve (see solve_ik.py)."""
  if "unsolved_frames" in K.files:
    return K["unsolved_frames"].astype(bool)
  return np.isnan(K["marker_err"]).all(1)  # IK files written before unsolved_frames


def fill_unsolved(Q, bad):
  """Linear interpolation of the qpos over the frames `bad` (the IK's unsolved frames,
  or NaN rows), held at the ends; root quaternion renormalised."""
  bad = bad | np.isnan(Q).any(1)
  if bad.any() and not bad.all():
    t = np.arange(len(Q))
    for j in range(Q.shape[1]):
      Q[bad, j] = np.interp(t[bad], t[~bad], Q[~bad, j])
    Q[bad, 3:7] /= np.linalg.norm(Q[bad, 3:7], axis=1, keepdims=True)
  return Q, bad


def force_plates(trial, n):
  """Calibrated force plates stored with the conditioned markers (C3D inputs), or None."""
  f = os.path.join(config.NPZ_DIR, f"{trial}.npz")
  if not os.path.exists(f):
    return None
  D = np.load(f)
  if "fp_force" not in D.files:
    return None
  a = int(config.IK_FRAMES.split(":")[0]) if config.IK_FRAMES else 0
  cut = lambda x: x[a:a + n].astype(np.float64)
  return dict(force=cut(D["fp_force"]), cop=cut(D["fp_cop"]), free=cut(D["fp_free"]))


def init():
  global KIN
  KIN = kinetics.Kinetics(ik.MarkerModel())


def run(trial):
  K = np.load(os.path.join(config.IK_DIR, f"{trial}.npz"))
  scale, rate = float(K["scale"]), float(K["rate"])
  Q, bad = fill_unsolved(K["qpos"].astype(np.float64), unsolved_frames(K))
  subj = config.subject_of(trial)
  body = config.anthropometrics()[subj]["mass_kg"]
  extra = config.carried_mass().get(subj, 0.0)
  kin = KIN.kinematics(Q)
  task = config.task()
  loads, more = [], {}
  if task is not None and hasattr(task, "kinetics_loads"):
    loads, more = task.kinetics_loads(trial, KIN, kin, scale, rate)
  out = KIN.solve(kin, rate, scale, body, {config.CARRIED_BODY: extra} if extra > 0 else None,
                  loads, lp_hz=config.KIN_LOWPASS_HZ, plates=force_plates(trial, len(Q)))
  out.update(more)
  out["unsolved"] = bad
  out["uncertain"] = out.get("uncertain", np.zeros(len(Q), bool)) | bad
  out["scale"] = scale
  np.savez_compressed(os.path.join(config.KIN_DIR, f"{trial}.npz"), **out)
  ok = ~out["uncertain"]
  W = (body + extra) * 9.81
  nb = ~out.get("held", np.zeros(len(Q), bool))
  pct = lambda k, q=95: round(float(np.percentile(out[k][ok], q)), 1)
  summ = dict(body_kg=body, carried_kg=round(extra, 2), unsolved=int(bad.sum()),
              uncertain_frac=round(float(1 - ok.mean()), 4),
              grf_over_weight_no_load=round(float(np.median(out["grf"][nb, 2]) / W), 4),
              L5S1_ext_p95=pct("L5S1_ext"), L5S1_ext_max=round(float(out["L5S1_ext"][ok].max()), 1),
              L5S1_compression_p95=pct("L5S1_compression"),
              L5S1_compression_max=round(float(out["L5S1_compression"][ok].max()), 1))
  for j in ("shoulder_r_flex", "shoulder_l_flex", "elbow_r_flex", "elbow_l_flex", "hip_r_ext",
            "hip_l_ext", "knee_r_ext", "knee_l_ext", "ankle_r_plantar", "ankle_l_plantar"):
    summ[f"{j}_p95"] = pct(j)
  print(trial, {k: summ[k] for k in ("grf_over_weight_no_load", "L5S1_ext_p95", "L5S1_ext_max",
                                     "L5S1_compression_p95", "unsolved")}, flush=True)
  return trial, summ


def main():
  config.usage(__doc__)
  trials = sys.argv[1:] or config.trials(config.IK_DIR)
  os.makedirs(config.KIN_DIR, exist_ok=True)
  ctx = mp.get_context("spawn")
  with ctx.Pool(min(len(trials), config.workers()), initializer=init) as pool:
    res = dict(pool.map(run, trials))
  path = os.path.join(config.KIN_DIR, "summary.json")
  old = json.load(open(path)) if os.path.exists(path) and sys.argv[1:] else {}
  json.dump(old | res, open(path, "w"), indent=1)


if __name__ == "__main__":
  main()
