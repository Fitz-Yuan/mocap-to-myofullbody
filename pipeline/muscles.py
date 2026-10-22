"""Muscle forces: static optimisation on the joint loads (optional, after kinetics.py).

For every trial with an IK file in <M2M_IK_DIR> and joint loads in <M2M_KIN_DIR>: per
frame, the activations of the model's muscles (M2M_MODEL; MuJoCo muscles) whose moments
match the net joint moments of the inverse dynamics, at the least sum of squared
activations, with a penalised reserve torque on every matched component so the frames the
muscles cannot carry show (mocap2myo/muscles.py). Which moments (M2M_MUSCLE_JOINTS):
"auto" = the top-down joints (L5/S1, shoulders, elbows: every component the kinetics
reports for them) and, when the trial's kinetics used force plates, hips, knees and ankles;
"top_down"; "all"; or kinetics joint names, comma separated. Muscle model: MuJoCo's
force-length-velocity curves at the frame's fibre length and velocity (rigid tendon;
M2M_MUSCLE_FV, M2M_MUSCLE_PASSIVE), the model's maximum forces (times
M2M_MUSCLE_STRENGTH). Reserve weight M2M_MUSCLE_RESERVE (N m), cost weighting
M2M_MUSCLE_WEIGHT (none / volume).

Frames the IK did not solve are interpolated as in kinetics.py. A matched component is
"uncertain" where the kinetics marks the frame (the carried load half on a surface,
unsolved frames) and, for a leg, where that foot is on the ground with an estimated share
of the ground reaction (double support off the force plates).

Outputs: <M2M_MUSCLE_DIR>/<trial>.npz (default <M2M_KIN_DIR>/muscles/):
  muscles [n], used [n]       every muscle of the model; used = crosses a matched joint
  crosses [n, m]              the muscle crosses the joint of that matched component
  act, force [T, n]           activation (0..1) and tension (N, active + passive); NaN
                              for muscles not used
  fibre_length [T, n]         normalised fibre length (float16; 1 = optimal)
  comps [m]                   matched components (kinetics keys, e.g. L5S1_ext)
  target, reserve, passive_moment [T, m]   net moment (N m), the reserve torque, the
                              passive muscle forces' share
  uncertain [T, m]            see above
  L5S1_joint_force [T, 3]     with L5/S1 matched: the force across the L5/S1 joint
                              (sacrum on L5, lumbar5 axes) once the muscle forces are
                              taken out of the net force; L5S1_compression_muscles,
                              L5S1_shear_ap_muscles [T]: its superior / anterior parts
  iters [T], scale, reserve_Nm, weight, fv, passive, strength, F0 [n], cost_weight [n]
and summary.json (per trial: reserves per component, the most active muscles, L5/S1
compression from the muscles vs the single-muscle estimate of kinetics.py).
  python pipeline/muscles.py [trial ...]
"""

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import config  # noqa: E402  (first: it selects the model and marker set)
from pipeline.kinetics import fill_unsolved, unsolved_frames  # noqa: E402
from mocap2myo import kinetics, muscles  # noqa: E402

KIN = None
LEGS = ("hip", "knee", "ankle")


def init():
  global KIN
  KIN = kinetics.Kinetics(config.marker_model())


def matched_joints(kn):
  """The kinetics joints whose moments the muscles must match (M2M_MUSCLE_JOINTS)."""
  have = [j for j in KIN.joints if any(f"{j}_{c[0]}" in kn.files for c in KIN.joints[j][3])]
  top = [j for j in have if j in kinetics.TOP_DOWN]
  sel = config.MUSCLE_JOINTS
  if sel == "auto":
    return top + ([j for j in have if j not in top] if "plates_all_measured" in kn.files else [])
  if sel == "top_down":
    return top
  if sel == "all":
    return have
  out = [s.strip() for s in sel.split(",") if s.strip()]
  bad = [j for j in out if j not in have]
  if bad:
    raise SystemExit(f"M2M_MUSCLE_JOINTS: {bad} not among the kinetics joints {have}")
  return out


def uncertain(kn, joints, comps):
  """[T, m] the frames each matched component is not determined (see the docstring)."""
  base = kn["uncertain"].astype(bool)
  out = np.zeros((len(base), len(comps)), bool)
  for c, (j, _) in enumerate(comps):
    u = base.copy()
    if j.split("_")[0] in LEGS and "foot_source" in kn.files:
      i = "lr".index(j[-1])
      u |= kn["foot_contact"][:, i] & (kn["foot_source"][:, i] == 0)
    out[:, c] = u
  return out


def run(trial):
  t0 = time.time()
  K = np.load(os.path.join(config.IK_DIR, f"{trial}.npz"))
  kn = np.load(os.path.join(config.KIN_DIR, f"{trial}.npz"))
  scale, rate = float(K["scale"]), float(K["rate"])
  if "model_profile" in K.files and str(K["model_profile"]) != KIN.mm.profile.name:
    raise SystemExit(f"{trial}: IK of {K['model_profile']}, not M2M_MODEL={KIN.mm.profile.name}")
  Q, bad = fill_unsolved(K["qpos"].astype(np.float64), unsolved_frames(K))
  if len(Q) != len(kn["grf"]):
    raise SystemExit(f"{trial}: {len(Q)} IK frames but {len(kn['grf'])} kinetics frames")
  joints = matched_joints(kn)
  comps = [(j, c[0]) for j in joints for c in KIN.joints[j][3]]
  so = muscles.Muscles(KIN, comps, reserve=config.MUSCLE_RESERVE, weight=config.MUSCLE_WEIGHT,
                       strength=config.MUSCLE_STRENGTH)
  kq = KIN.kinematics(Q)
  fj = ("L5S1",) if "L5S1" in joints else ()
  length, arms, fdir = so.geometry(Q, kq, scale, force_joints=fj)
  keys = [f"{j}_{c}" for j, c in comps]
  target = np.stack([kn[k] for k in keys], 1).astype(np.float64)
  res = so.solve(length, arms, target, rate, lp_hz=config.KIN_LOWPASS_HZ,
                 fv=config.MUSCLE_FV, passive=config.MUSCLE_PASSIVE)
  T, n = len(Q), len(so.names)

  def spread(x, dt=np.float32):  # [T, used] -> [T, all muscles], NaN for the unused
    out = np.full((len(x), n), np.nan, dt)
    out[:, so.use] = x
    return out
  unc = uncertain(kn, joints, comps)
  crosses = np.stack([so.crosses[j] for j, _ in comps], 1)
  out = dict(muscles=np.array(so.names), used=so.use, crosses=crosses, act=spread(res["act"]),
             force=spread(res["tension"]), fibre_length=spread(res["fibre_length"], np.float16),
             comps=np.array(keys), target=target.astype(np.float32),
             reserve=res["reserve"].astype(np.float32),
             passive_moment=res["passive_moment"].astype(np.float32), uncertain=unc,
             iters=res["iters"], scale=scale, reserve_Nm=so.reserve, weight=so.weight,
             fv=config.MUSCLE_FV, passive=config.MUSCLE_PASSIVE, strength=config.MUSCLE_STRENGTH,
             F0=spread(so.prm.F0[None, :])[0], cost_weight=spread(so.w[None, :])[0])
  if fj:
    # the force across the L5/S1 joint: the net force on the trunk above minus the pull of
    # the muscles crossing it (lumbar5 axes); its superior / anterior parts as kinetics.py's
    Fm = np.einsum("tn,tnk->tk", res["tension"], fdir["L5S1"])
    J = kn["L5S1_force"] - Fm
    sup, ant = KIN.trunk_axes
    fr = KIN.joints["L5S1"][2]
    local = lambda a: (np.eye(3)["xyz".index(a)] if not a.startswith("anat:")
                       else KIN.anat[(fr, a[5:])])
    out.update(L5S1_joint_force=J.astype(np.float32),
               L5S1_compression_muscles=(J @ local(sup)).astype(np.float32),
               L5S1_shear_ap_muscles=(J @ local(ant)).astype(np.float32))
  os.makedirs(config.MUSCLE_DIR, exist_ok=True)
  np.savez_compressed(os.path.join(config.MUSCLE_DIR, f"{trial}.npz"), **out)
  # summary: reserves per component over the frames it is determined
  summ = dict(frames=T, unsolved=int(bad.sum()), muscles_used=int(so.use.sum()),
              seconds=round(time.time() - t0, 1), comps={})
  for c, k in enumerate(keys):
    ok = ~unc[:, c]
    r, tau = res["reserve"][ok, c], target[ok, c]
    summ["comps"][k] = dict(net_p95=round(float(np.percentile(np.abs(tau), 95)), 1),
                            reserve_rms=round(float(np.sqrt(np.mean(r**2))), 2),
                            reserve_max=round(float(np.abs(r).max()), 1),
                            reserve_frames=round(float(np.mean(np.abs(r) > 1.0)), 4))
  ok = ~unc.any(1)
  p95 = np.percentile(res["act"][ok], 95, axis=0)
  top = np.argsort(-p95)[:12]
  summ["act_p95_top"] = {so.names[np.where(so.use)[0][i]]: round(float(p95[i]), 3) for i in top}
  if fj:
    cm, cs = out["L5S1_compression_muscles"][ok], kn["L5S1_compression"][ok]
    summ.update(L5S1_compression_muscles_p95=round(float(np.percentile(cm, 95)), 1),
                L5S1_compression_single_muscle_p95=round(float(np.percentile(cs, 95)), 1))
  worst = max(summ["comps"].items(), key=lambda kv: kv[1]["reserve_frames"])
  print(trial, f"{T} frames, {summ['muscles_used']} muscles, {summ['seconds']} s; reserve > 1 N m in "
        f"{100 * worst[1]['reserve_frames']:.1f} % of frames at worst ({worst[0]})", flush=True)
  return trial, summ


def main():
  config.usage(__doc__)
  trials = sys.argv[1:] or [t for t in config.trials(config.IK_DIR)
                            if os.path.exists(os.path.join(config.KIN_DIR, f"{t}.npz"))]
  os.makedirs(config.MUSCLE_DIR, exist_ok=True)
  ctx = mp.get_context("spawn")
  with ctx.Pool(min(len(trials), config.workers()), initializer=init) as pool:
    res = dict(pool.map(run, trials))
  path = os.path.join(config.MUSCLE_DIR, "summary.json")
  old = json.load(open(path)) if os.path.exists(path) and sys.argv[1:] else {}
  json.dump(old | res, open(path, "w"), indent=1)


if __name__ == "__main__":
  main()
