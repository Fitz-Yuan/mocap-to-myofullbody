"""Per-joint confidence of the IK: how firmly the markers pin each joint angle, per frame.

For every trial in <M2M_IK_DIR> (optional stage, after solve_ik.py; better after
validate/lomo.py): a standard deviation in degrees for each joint the IK solves, per
frame. pipeline/export_viewer.py draws it as a band around each joint-angle curve.

Method.
  1. A noise level s_k per marker (per axis, the trial's typical value). In a least-
     squares fit a marker's residual has the variance s^2 (1 - h) and its leave-one-out
     prediction error s^2 / (1 - h), h being how much the fit leans on that marker (its
     leverage), so s = sqrt(residual x leave-one-out error) whatever h is. With
     <M2M_IK_DIR>/lomo.json (validate/lomo.py): the medians of the two over the trial.
     Without it, or for a marker it skips (the head): the same from the fit's own
     leverage, s^2 = median |e|^2 / (3 (1 - h)) on one frame per M2M_LOMO_EVERY_S (a
     median distance of an isotropic error is 1.54 s). The two agree (case study, one
     trial: median ratio 1.10 over the markers, r = 0.94).
  2. Per frame, the markers of a segment (a body of the model) get s_k x max(1, its
     residual / its median residual): frames where the segment fits worse than usual
     widen its band.
  3. The IK's marker fit linearised at the solution: J_k (3 x n) the model marker's
     motion with the IK's own unknowns (root position and rotation, the solved joints;
     finite differences), W_k the IK's marker weight and its robust loss's down-weighting
     at that residual (pelvis: squared loss). The single-frame fit's covariance
     A^-1 B A^-1 with A = sum_k W_k J_k'J_k and B = sum_k W_k^2 s_k^2 J_k'J_k over the
     markers present (missing and rejected markers drop out, so gaps widen the band).
     Its diagonal, as degrees (re-solving frames with markers moved at random by s_k:
     the spread is 0.96 x this, median over joints and frames); a joint no marker holds
     (a 1 mm marker error would turn it more than 90 deg: the toes) is NaN.
  Computed every M2M_CONF_STEP_S seconds (the residual ratio: the largest in each step)
  and interpolated; unsolved frames NaN.

What it is not. It measures how well the markers on the segments pin each angle, given how
consistently they move with the model. Soft tissue that carries all the markers of a
segment together fits as joint motion and leaves no residual, so it is not in the band;
nor are errors of the model's joint axes or of the calibration (marker placement), which
shift a curve rather than scatter it, or the IK's frame-to-frame smoothing, contact terms
and joint limits, which hold the actual estimate tighter than a single-frame fit.

Output: <M2M_CONF_DIR>/<trial>.npz (default <M2M_IK_DIR>/confidence/): joint_sd_deg
[T, n] float32 (n = joints, the IK's solved joints), marker_sd_mm [n_markers] (s_k per
axis; NaN = not used), marker_sd_leverage_mm (the leverage-based level, for comparison),
source ("lomo" or "leverage"), labels, step_s, model_profile.
  python pipeline/confidence.py [trial ...]
"""

import json
import multiprocessing as mp
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from pipeline import config  # noqa: E402

OUT = config.CONF_DIR
MEDIAN_DIST = 1.5382  # median |e| / per-axis SD of an isotropic 3-D Gaussian
SD_MAX = np.radians(90.0)  # a joint looser than this is not held by the markers: NaN


def jacobian(mm, x, q_root, eps=1e-6):
  """Model markers [n, 3] at the IK parameters x and their Jacobian [n, 3, len(x)]."""
  m0 = mm.markers(mm.pose(x, q_root))
  J = np.zeros((*m0.shape, len(x)))
  for j in range(len(x)):
    xx = x.copy()
    xx[j] += eps
    J[..., j] = (mm.markers(mm.pose(xx, q_root)) - m0) / eps
  return m0, J


def weights(mm, e, ok, robust):
  """The IK's weight per marker at residuals e [n, 3] (MarkerModel.solve: the marker set's
  weight, the Cauchy loss's down-weighting 1 / (1 + w |e|^2 / f^2), squared loss on the
  pelvis); 0 where not ok."""
  w = np.where(ok, mm.w, 0.0)
  if robust:
    d2 = np.where(ok, (np.nan_to_num(e) ** 2).sum(1), 0.0)
    w = np.where(mm.ms.PELVIS_MASK, w, w / (1 + w * d2 / robust ** 2))
  return w


def info(J, W, lam=1e-10):
  """A = sum_k W_k J_k'J_k (+ a tiny ridge)."""
  return np.einsum("k,kai,kaj->ij", W, J, J) + lam * np.eye(J.shape[-1])


def trial(name):
  from mocap2myo import ik as mk
  K = np.load(os.path.join(config.IK_DIR, f"{name}.npz"))
  neck = "neck_qpos" in K.files
  mm = config.marker_model(neck=neck)
  lp = float(K["lowpass_hz"]) if "lowpass_hz" in K.files else config.LOWPASS_HZ
  robust = float(K["robust"]) if "robust" in K.files else config.ROBUST
  P, rate, _ = mk.load_trial(os.path.join(config.NPZ_DIR, f"{name}.npz"), lowpass_hz=lp or None, mm=mm,
                             max_gap_s=config.MAX_GAP_S, gap_fill=config.GAP_FILL)
  cal = np.load(os.path.join(config.IK_DIR, f"{config.subject_of(name)}_offsets.npz"))
  mm.set_offsets(cal["offsets"])
  if "active" in K.files:
    mm.active &= K["active"].astype(bool)
    mm.w = np.where(mm.active, mm.w, 0.0)
  Q = K["qpos"].astype(np.float64)
  if neck:
    Q = mm.full_qpos(Q, K["neck_qpos"].astype(np.float64))
  a = int(K["frame0"]) if "frame0" in K.files else 0
  P = P[a:a + len(Q)]
  E = K["marker_err"].astype(np.float64)  # NaN: missing, rejected, unsolved
  T, n = E.shape
  bad = K["unsolved_frames"].astype(bool) if "unsolved_frames" in K.files else np.isnan(E).all(1)
  use = ~np.isnan(E) & mm.active[None]
  labels = list(mm.ms.LABELS)

  def frame(t):
    q = Q[t]
    x = np.r_[q[:3], np.zeros(3), q[mm.adr]]
    M, J = jacobian(mm, x, q[3:7])
    e = M - P[t]
    ok = use[t] & np.isfinite(e).all(1)
    return J, e, ok, weights(mm, e, ok, robust)

  # 1. per-marker level: from the leave-one-marker-out check, else from the leverage
  every = max(1, int(round(config.LOMO_EVERY_S * rate)))
  lev = [[] for _ in range(n)]
  for t in np.flatnonzero(~bad)[::every]:
    J, e, ok, W = frame(t)
    Ai = np.linalg.inv(info(J, W))
    for k in np.flatnonzero(ok):
      h = W[k] * np.einsum("ai,ij,aj->", J[k], Ai, J[k]) / 3
      if h < 0.99:
        lev[k].append((e[k] ** 2).sum() / (3 * (1 - h)))
  s_lev = np.array([np.sqrt(np.median(v)) if v else np.nan for v in lev])
  s = s_lev.copy()
  source = "leverage"
  lomo = os.path.join(config.IK_DIR, "lomo.json")
  if os.path.exists(lomo):
    L = json.load(open(lomo)).get(name)
    if L:
      source = "lomo"
      for k, l in enumerate(labels):
        if l in L and L[l]["resid_med"] > 0:
          s[k] = np.sqrt(L[l]["resid_med"] * L[l]["pred_med"]) * 1e-3 / MEDIAN_DIST
  # 2. the segments' residuals relative to their median, per frame
  bodies = sorted({int(b) for b, on in zip(mm.bid, mm.active) if on})
  seg = np.array([bodies.index(int(b)) if on else -1 for b, on in zip(mm.bid, mm.active)])
  ratio = np.ones((T, len(bodies)))
  for i in range(len(bodies)):
    r = E[:, seg == i]
    cnt = np.isfinite(r).sum(1)
    rms = np.where(cnt > 0, np.sqrt(np.nansum(r ** 2, 1) / np.maximum(cnt, 1)), np.nan)
    med = np.nanmedian(rms) if np.isfinite(rms).any() else np.nan
    if med > 0:
      ratio[:, i] = np.where(np.isfinite(rms), np.maximum(1.0, rms / med), 1.0)
  # 3. covariance of the single-frame fit, every step
  step = max(1, int(round(config.CONF_STEP_S * rate)))
  frames = np.flatnonzero(~bad)[::step]
  nj = len(mm.free)
  sd = np.full((len(frames), nj), np.nan)
  for i, t in enumerate(frames):
    J, e, ok, W = frame(t)
    win = ratio[max(0, t - step // 2):t + step // 2 + 1].max(0)
    sk = np.where(ok, np.nan_to_num(s) * win[np.maximum(seg, 0)], 0.0)
    Ai = np.linalg.inv(info(J, W))
    B = np.einsum("k,kai,kaj->ij", W ** 2 * sk ** 2, J, J)
    v = np.sqrt(np.maximum(np.diag(Ai @ B @ Ai), 0.0))[6:]
    sd[i] = np.where(np.diag(Ai)[6:] ** 0.5 * 1e-3 > SD_MAX, np.nan, v)  # at a 1 mm level
  out = np.full((T, nj), np.nan)
  if len(frames):
    for j in range(nj):
      ok = np.isfinite(sd[:, j])
      if ok.any():
        out[:, j] = np.interp(np.arange(T), frames[ok], sd[ok, j])
    out[:, np.isnan(sd).all(0)] = np.nan
  out[bad] = np.nan
  os.makedirs(OUT, exist_ok=True)
  f = os.path.join(OUT, f"{name}.npz")
  np.savez_compressed(f[:-4] + ".part.npz", joint_sd_deg=np.degrees(out).astype(np.float32),
                      joints=np.array(mm.free), labels=np.array(labels),
                      marker_sd_mm=(1e3 * np.where(mm.active, s, np.nan)).astype(np.float32),
                      marker_sd_leverage_mm=(1e3 * np.where(mm.active, s_lev, np.nan)).astype(np.float32),
                      source=source, step_s=step / rate, model_profile=mm.profile.name)
  os.replace(f[:-4] + ".part.npz", f)
  held = np.isfinite(out).any(0)
  med = np.nanmedian(np.degrees(out[:, held]), 0)
  return name, source, {j: round(float(v), 2) for j, v in zip(np.array(mm.free)[held], med)}


def main():
  config.usage(__doc__)
  names = sys.argv[1:] or config.trials(config.IK_DIR)
  with mp.get_context("spawn").Pool(min(len(names), config.workers())) as pool:
    for name, source, med in pool.imap_unordered(trial, names):
      print(f"{name} ({source}): median SD (deg) {med}", flush=True)


if __name__ == "__main__":
  main()
