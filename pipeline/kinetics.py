"""Inverse dynamics: net joint forces and moments from the IK (after the reference).

For every trial in <M2M_IK_DIR>: the model (M2M_MODEL, default MyoFullBody; its profile's
[kinetics] joints) at the subject's size and mass (the
anthropometrics; equipment worn: M2M_CARRIED_COLUMN, lumped at M2M_CARRIED_BODY: a body
of the model, or a canonical segment such as torso), the task's external loads on the
body (box_lift: the box on the hands while it is carried, tasks/box_lift/loads.py;
M2M_TASK=none: no external load but the ground), and mocap2myo/kinetics.py: L5/S1,
shoulders and elbows top-down (no ground reaction needed), the total ground reaction
with its centre of pressure, and hips / knees / ankles with that reaction shared
between the feet (estimated in double support). Positions are low-passed at
M2M_KIN_LOWPASS_HZ (default 6 Hz) before they are differentiated, and force plates (C3D
inputs, where a foot is on one) with the same filter (M2M_KIN_FILTER_PLATES=0: unfiltered).

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
from pipeline import config  # noqa: E402  (first: it selects the model and marker set)
from mocap2myo import kinetics  # noqa: E402

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


def static_plate_mass(subject):
  """Mass (kg) the force plates read during the subject's static standing trial, or None."""
  f = os.path.join(config.NPZ_DIR, "static", f"{subject}.npz")
  if not os.path.exists(f):
    return None
  D = np.load(f)
  if "fp_force" not in D.files:
    return None
  fz = np.nansum(np.where(D["fp_force"][..., 2] > 20.0, D["fp_force"][..., 2], 0.0), axis=-1)
  fz = fz[fz > 0]
  return float(np.median(fz)) / 9.81 if len(fz) else None


def force_plates(trial, n, rate):
  """Calibrated force plates stored with the conditioned markers (C3D inputs), or None;
  low-passed like the body motion (M2M_KIN_FILTER_PLATES, kinetics.lowpass_plates)."""
  f = os.path.join(config.NPZ_DIR, f"{trial}.npz")
  if not os.path.exists(f):
    return None
  D = np.load(f)
  if "fp_force" not in D.files:
    return None
  a = int(config.IK_FRAMES.split(":")[0]) if config.IK_FRAMES else 0
  cut = lambda x: x[a:a + n].astype(np.float64)
  pl = dict(force=cut(D["fp_force"]), cop=cut(D["fp_cop"]), free=cut(D["fp_free"]))
  if not config.KIN_FILTER_PLATES:
    return pl
  if "fp_corners" not in D.files:
    raise SystemExit(f"{f}: no fp_corners (plate geometry) to filter the plates with; run "
                     "prepare_markers.py again, or M2M_KIN_FILTER_PLATES=0")
  return kinetics.lowpass_plates(dict(pl, corners=D["fp_corners"]), rate, config.KIN_LOWPASS_HZ,
                                 config.FP_MIN_FZ)


def init():
  global KIN
  KIN = kinetics.Kinetics(config.marker_model())


def carried_body(m):
  """M2M_CARRIED_BODY as a body of the model: the body of that name, else the body of
  that canonical segment (MASS has no "torso" body: its torso segment's)."""
  import mujoco
  b = config.CARRIED_BODY
  if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) >= 0:
    return b
  seg = KIN.mm.profile.segment_body(b)
  if seg is None:
    raise SystemExit(f"M2M_CARRIED_BODY={b!r}: no such body or segment in {KIN.mm.profile.name}")
  return seg


def run(trial):
  K = np.load(os.path.join(config.IK_DIR, f"{trial}.npz"))
  scale, rate = float(K["scale"]), float(K["rate"])
  Q, bad = fill_unsolved(K["qpos"].astype(np.float64), unsolved_frames(K))
  subj = config.subject_of(trial)
  body = config.anthropometrics()[subj]["mass_kg"]
  extra = config.carried_mass().get(subj, 0.0)
  weighed = static_plate_mass(subj)
  if weighed is not None and 0.8 < weighed / (body + extra) < 1.25:
    body = weighed - extra  # the plates weigh the subject as recorded (shoes, clothes, markers)
  kin = KIN.kinematics(Q)
  task = config.task()
  loads, more = [], {}
  if task is not None and hasattr(task, "kinetics_loads"):
    loads, more = task.kinetics_loads(trial, KIN, kin, scale, rate)
  cal = os.path.join(config.IK_DIR, f"{trial}.npz")
  if "model_profile" in K.files and str(K["model_profile"]) != KIN.mm.profile.name:
    raise SystemExit(f"{cal}: IK of {K['model_profile']}, not M2M_MODEL={KIN.mm.profile.name}")
  out = KIN.solve(kin, rate, scale, body, {carried_body(KIN.m): extra} if extra > 0 else None,
                  loads, lp_hz=config.KIN_LOWPASS_HZ, plates=force_plates(trial, len(Q), rate))
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
    if j in out:  # a model without arms has no shoulders / elbows
      summ[f"{j}_p95"] = pct(j)
  if "plates_all_measured" in out:  # total vertical force: plates vs whole-body dynamics
    pm = out["plates_all_measured"] & ok
    if pm.any():
      summ["plates_fz_rms_over_weight"] = round(float(np.sqrt(np.mean(
        (out["plates_fz"][pm] - out["plates_pred_fz"][pm]) ** 2)) / W), 4)
      summ["plates_frames"] = int(pm.sum())
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
