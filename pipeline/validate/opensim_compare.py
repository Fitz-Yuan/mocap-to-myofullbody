"""Score one or more of our IK runs and the OpenSim baseline with the same checks.

Inputs: the cleaned markers every method was given (<M2M_NPZ_DIR>), our IK runs
(LABEL=IK_DIR arguments; default "ours" = <M2M_IK_DIR>; case study: the
motion-trial and the static-trial calibration) and the OpenSim run (opensim_ik/,
opensim_ik.py). All residuals are measured against the same unfiltered markers in
the subject's own size (our IK works on height-scaled markers and a 6 Hz low-pass;
its model markers are scaled back here).

Per trial and method:
  residual     body markers (head excluded), median / 95th pct over marker-frames, mm
  stand knee / hip   median flexion over quiet upright standing (mocap2myo.ik.standing)
  flips        share of frames with any joint faster than 15 rad/s
and for the Xsens trial (M2M_XSENS_TRIAL, if its c3d exists) the knee against Xsens
(xsens.py): offset and RMS after the offset, from the joint centres (femur, tibia,
talus origins) of each model. Also the frame-by-frame difference between the last
of our runs and OpenSim for hip / knee / ankle flexion.

  python pipeline/validate/opensim_compare.py ["LABEL=IK_DIR" ...]    (needs ezc3d)
Writes <M2M_ROOT>/compare_osim/compare.json and compare.md.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mocap2myo import ik as mk  # noqa: E402
from pipeline import config  # noqa: E402
from pipeline.validate import xsens as XC  # noqa: E402

NPZ = config.NPZ_DIR
OSIM = config.OSIM_DIR
OUT = config.OSIM_COMPARE_DIR
ANGLES = ["hip_flexion_r", "hip_flexion_l", "knee_angle_r", "knee_angle_l", "ankle_angle_r", "ankle_angle_l"]
HEAD = mk.HEAD


def read_mot(path):
  L = open(path).read().splitlines()
  i = next(k for k, l in enumerate(L) if l.startswith("time"))
  cols = L[i].split("\t")
  A = np.array([[float(x) for x in l.split("\t")] for l in L[i + 1:] if l.strip()])
  return cols, A


def resid(model_mk, meas):
  e = np.linalg.norm(model_mk[:, ~HEAD] - meas[:, ~HEAD], axis=-1)
  e = e[~np.isnan(e)]
  return 1e3 * float(np.median(e)), 1e3 * float(np.percentile(e, 95))


def flips(rad, rate):
  sp = np.abs(np.diff(rad, axis=0)) * rate
  return float(np.mean(np.nanmax(sp, axis=1) > 15))


def ours(trial, ikd, mm, P, st, rate):
  K = np.load(os.path.join(ikd, f"{trial}.npz"))
  Q, s = K["qpos"].astype(np.float64), float(K["scale"])
  mm.set_offsets(np.load(os.path.join(ikd, f"{config.subject_of(trial)}_offsets.npz"))["offsets"])
  M = np.full((len(Q), len(mk.LABELS), 3), np.nan)
  for t, q in enumerate(Q):
    if not np.isnan(q).any():
      M[t] = mm.markers(q)
  M /= s
  adr = {j: mm.m.jnt_qposadr[mm.m.joint(j).id] for j in ANGLES}
  ang = {j: np.degrees(Q[:, adr[j]]) for j in ANGLES}
  free = Q[:, mm.adr]
  return dict(resid=resid(M, P), angles=ang,
              stand_knee=max(float(np.nanmedian(ang[f"knee_angle_{x}"][st])) for x in "rl"),
              stand_hip=max(abs(float(np.nanmedian(ang[f"hip_flexion_{x}"][st]))) for x in "rl"),
              flips=flips(free, rate))


def osim(trial, P, st, rate):
  d = os.path.join(OSIM, config.subject_of(trial))
  cols, A = read_mot(os.path.join(d, f"{trial}.mot"))
  M = np.load(os.path.join(d, f"{trial}_markers.npz"))["markers"]
  ang = {j: A[:, cols.index(j)] for j in ANGLES}
  rot = [i for i, c in enumerate(cols) if c != "time" and not c.endswith(("_tx", "_ty", "_tz"))]
  return dict(resid=resid(M, P), angles=ang,
              stand_knee=max(float(np.nanmedian(ang[f"knee_angle_{x}"][st])) for x in "rl"),
              stand_hip=max(abs(float(np.nanmedian(ang[f"hip_flexion_{x}"][st]))) for x in "rl"),
              flips=flips(np.radians(A[:, rot]), rate))


def xsens_knee(knee_by_method):
  """Offset and RMS after the offset of each method's knee against Xsens (XC.TRIAL)."""
  xa, xr = XC.xsens_angles(XC.C3D)
  out = {}
  for name, (kr, kl, rate) in knee_by_method.items():
    xs = {k: XC.resample(xa[k], xr, rate) for k in ("knee_r", "knee_l")}
    lag = int(round(0.05 * rate))  # the lag xsens.py found for every IK run of the case study
    res = {}
    for side, k in (("r", kr), ("l", kl)):
      u, v = k[lag:], xs[f"knee_{side}"]
      n = min(len(u), len(v))
      ok = ~np.isnan(u[:n]) & ~np.isnan(v[:n])
      dlt = u[:n][ok] - v[:n][ok]
      res[side] = dict(offset=float(dlt.mean()), rms_after=float(dlt.std()),
                       r=float(np.corrcoef(u[:n][ok], v[:n][ok])[0, 1]))
    out[name] = res
  return out


def main():
  config.usage(__doc__)
  runs = [a.split("=", 1) for a in sys.argv[1:]] or [("ours", config.IK_DIR)]
  OURS = {label: os.path.expanduser(d) for label, d in runs}
  last = list(OURS)[-1]  # compared frame by frame with OpenSim
  os.makedirs(OUT, exist_ok=True)
  mm = mk.MarkerModel()
  trials = config.trials(NPZ)
  rows, diffs = {}, {j: [] for j in ANGLES}
  xs = {}
  for trial in trials:
    D = np.load(os.path.join(NPZ, f"{trial}.npz"))
    P, rate = D["markers"].astype(np.float64), float(D["marker_rate"])
    st = mk.standing(P, rate)
    r = {name: ours(trial, ikd, mm, P, st, rate) for name, ikd in OURS.items()}
    r["OpenSim"] = osim(trial, P, st, rate)
    a, b = r[last]["angles"], r["OpenSim"]["angles"]
    for j in ANGLES:
      n = min(len(a[j]), len(b[j]))
      d = a[j][:n] - b[j][:n]
      diffs[j].append(d[~np.isnan(d)])
    rows[trial] = {k: {kk: vv for kk, vv in v.items() if kk != "angles"} for k, v in r.items()}
    if trial == XC.TRIAL and os.path.exists(XC.C3D):
      geo = {}
      for name, ikd in OURS.items():
        ia, rate_ik = XC.ik_angles(ikd)
        geo[name] = (ia["knee_r"], ia["knee_l"], rate_ik)
      B = np.load(os.path.join(OSIM, config.subject_of(trial), f"{trial}_bodies.npz"))
      O, names = B["origins"], list(B["bodies"])
      g = lambda b: O[:, names.index(b)]
      geo["OpenSim"] = (XC.flex(g("femur_r"), g("tibia_r"), g("talus_r")),
                        XC.flex(g("femur_l"), g("tibia_l"), g("talus_l")), rate)
      xs = xsens_knee(geo)
    print(trial, {k: (round(v["resid"][0], 1), round(v["stand_knee"], 1), round(v["stand_hip"], 1))
                  for k, v in rows[trial].items()}, flush=True)
  methods = list(rows[trials[0]])
  summ = {}
  for m in methods:
    v = [rows[t][m] for t in trials]
    summ[m] = dict(resid_med=float(np.median([x["resid"][0] for x in v])),
                   resid_p95=float(np.median([x["resid"][1] for x in v])),
                   stand_knee_med=float(np.median([x["stand_knee"] for x in v])),
                   stand_knee_over10=int(sum(x["stand_knee"] >= 10 for x in v)),
                   stand_hip_med=float(np.median([x["stand_hip"] for x in v])),
                   stand_hip_over10=int(sum(x["stand_hip"] > 10 for x in v)),
                   flips_med=float(np.median([x["flips"] for x in v])))
  agree = {j: dict(mean=float(np.mean(np.concatenate(diffs[j]))),
                   rms=float(np.sqrt(np.mean(np.concatenate(diffs[j]) ** 2)))) for j in ANGLES}
  run = json.load(open(os.path.join(OSIM, "run.json")))
  json.dump(dict(per_trial=rows, summary=summ, xsens=xs, xsens_trial=XC.TRIAL if xs else None,
                 ours_minus_osim_run=last, ours_minus_osim_deg=agree,
                 opensim_wall_s=run["_wall_s"]), open(os.path.join(OUT, "compare.json"), "w"), indent=1)
  lines = [f"| Measure ({len(trials)} trials) | " + " | ".join(methods) + " |",
           "|---|" + "---|" * len(methods)]
  fmt = [("Marker residual, median (mm)", "resid_med", "{:.1f}"), ("Marker residual, 95th pct (mm)", "resid_p95", "{:.1f}"),
         ("Quiet-standing knee, median (deg)", "stand_knee_med", "{:.1f}"), ("Trials with standing knee >= 10 deg", "stand_knee_over10", "{}"),
         ("Quiet-standing |hip|, median (deg)", "stand_hip_med", "{:.1f}"), ("Trials with standing |hip| > 10 deg", "stand_hip_over10", "{}"),
         ("Frames with a joint > 15 rad/s, median", "flips_med", "{:.4f}")]
  for label, k, f in fmt:
    lines.append(f"| {label} | " + " | ".join(f.format(summ[m][k]) for m in methods) + " |")
  for side in "rl" if xs else "":
    lines.append(f"| Knee vs Xsens, offset / RMS after ({side}) | " + " | ".join(
      f"{xs[m][side]['offset']:+.1f} / {xs[m][side]['rms_after']:.1f}" for m in methods) + " |")
  lines.append("")
  lines.append(f"{last[:1].upper() + last[1:]} minus OpenSim, all frames: " + ", ".join(
    f"{j} {v['mean']:+.1f} (RMS {v['rms']:.1f})" for j, v in agree.items()))
  open(os.path.join(OUT, "compare.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
