"""Marker IK of every trial onto MyoFullBody: subject calibration, then per-frame IK.

Input: <M2M_NPZ_DIR>/<trial>.npz (prepare_markers.py), and with M2M_STATIC=1 the
  static standing trials <M2M_NPZ_DIR>/static/<subject>.npz.
Stage A (per subject): marker-offset calibration (mocap2myo.ik.calibrate) on frames
  drawn from all of the subject's trials -- 40 frames, because one trial mixes
  stooping, walking, turning and overhead reaching.
Stage B (per trial): per-frame IK with the contact terms (floor, stance sole
  lock, joint speed limit), mocap2myo.ik.ik_trial(contact=True).
Output: <M2M_IK_DIR>/<subject>_offsets.npz, <trial>.npz with
  qpos [T, nq] (scaled-subject frame: markers x scale, model not scaled),
  marker_err [T, n] (m; NaN = marker missing or rejected), unsolved_frames [T] bool
  (frames the IK did not solve: before the first frame with all four pelvis markers,
  or no marker at all; their qpos repeats a solved pose and marker_err is all NaN),
  unsolved (their count), scale, rate, labels, joint_names, model and the settings;
  with M2M_NECK=1 also neck_qpos [T, 3] and neck_joints.
Settings (env): see the IK section of config.py.

  python pipeline/solve_ik.py [--calib-only] [trial ...]
M2M_IK_FRAMES=a:b restricts stage B to a frame range (pilot / timing).
"""

import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pipeline import config  # noqa: E402

NPZ = config.NPZ_DIR
OUT = config.IK_DIR
OUTLIER = config.OUTLIER
TRACK_OUTLIER = config.TRACK_OUTLIER
ROOT_INIT = config.ROOT_INIT
MTP_PRIOR = config.MTP_PRIOR
ANCHOR_CENTROID = config.ANCHOR_CENTROID
PELVIS_GAP = config.PELVIS_GAP
ROBUST = config.ROBUST
LOWPASS_HZ = config.LOWPASS_HZ
STATIC = config.STATIC
STATIC_SHARE = config.STATIC_SHARE
STATIC_W = config.STATIC_W
STATIC_MARKERS = config.STATIC_MARKERS
FOOT_CAP = config.FOOT_CAP
FLAT_W = config.FLAT_W


# Standing frames are calibrated with flat feet (ik.calibrate `stand`): the
# TOE marker sits on the shoe's toe box, which otherwise tilts the model foot
# toes-up ~5 deg (case study: toe tip 2 cm above the heel while standing).
# Placement varies most for wand / mid-segment markers, the trunk markers (EMG
# electrodes on the back and abdomen) and the head markers (K5 mask harness):
# a 6 cm cap for every marker pinned them (case study: T10 62 mm median error), so
# these may move up to 15 cm, foot markers FOOT_CAP (10 cm), the others 6 cm.
# Label endings of the Plug-in-Gait markers (mocap2myo.markerset).
LOOSE = ("THI", "TIB", "UPA", "FRM", "T10", "STRN", "RBAK", "FHD", "BHD")
# Foot markers sit on thick-soled slides (TOE on the toe box): cap FOOT_CAP
# (M2M_FOOT_CAP), see config.py.
FOOT = ("HEE", "TOE", "ANK")


def max_shift(labels):
  return np.array([0.15 if l.endswith(LOOSE) else FOOT_CAP if l.endswith(FOOT) else 0.06
                   for l in labels])


def standing_frames(loaded, per_trial=12, seed=0, min_s=0.3):
  """Flat-foot frames: all four sole markers (both heels and toes) slow and on
  the floor (ik.sole_stance) for >= min_s -- standing, turning on the spot,
  lifting. The whole body need not be still (quiet standing is only ~3 s per
  cycle and was missed entirely for S03 / S05 / S08 with a pelvis-speed test)."""
  from mocap2myo import ik as mk
  rng = np.random.default_rng(seed)
  out = []
  for P, rate in loaded:
    ok = np.all(list(mk.sole_stance(P, rate).values()), 0)
    k = int(min_s * rate)
    ok = np.convolve(ok, np.ones(k), "same") >= k  # inside >= min_s runs
    idx = np.flatnonzero(ok)
    if len(idx):
      out += [P[i] for i in rng.choice(idx, min(per_trial, len(idx)), replace=False)]
  print(f"  flat-foot calibration frames: {len(out)}", flush=True)
  return np.array(out) if out else None


def static_frames(subj, loaded, n=10):
  """[n, n_markers, 3] quietest frames of the subject's static trial (scaled like
  the trials), else quiet upright standing frames from its trials."""
  from mocap2myo import ik as mk
  f = os.path.join(NPZ, "static", f"{subj}.npz")
  if os.path.exists(f):
    P, _, _ = mk.load_trial(f, lowpass_hz=LOWPASS_HZ or None)
    sp = np.r_[np.inf, np.nanmax(np.linalg.norm(np.diff(P, axis=0), axis=-1), 1)]
    idx = np.sort(np.argsort(sp)[:max(5 * n, 50)])
    src = "static trial"
  else:
    cand = [(P, np.flatnonzero(mk.standing(P, rate))) for P, rate in loaded]
    P = np.concatenate([P[i] for P, i in cand])
    idx = np.arange(len(P))
    src = "quiet standing in the trials (no static trial)"
  out = P[idx[:: max(1, len(idx) // n)][:n]]
  print(f"  {subj} static frames: {len(out)} from {src}", flush=True)
  return out


def calib(subj):
  from mocap2myo import ik as mk
  f = os.path.join(OUT, f"{subj}_offsets.npz")
  if os.path.exists(f):
    return subj, "cached"
  loaded = [mk.load_trial(os.path.join(NPZ, f"{t}.npz"), lowpass_hz=LOWPASS_HZ or None)[:2]
            for t in config.trials(NPZ) if config.subject_of(t) == subj]
  trials = [P for P, _ in loaded]
  mm = mk.MarkerModel(neck=config.NECK)
  off = mk.calibrate(mm, trials, n_frames=40, iters=8, max_shift=max_shift(mk.LABELS),
                     outlier=OUTLIER, stand=standing_frames(loaded), flat=FLAT_W,
                     static=static_frames(subj, loaded) if STATIC else None,
                     static_share=STATIC_SHARE, static_w=STATIC_W,
                     static_segments={"legs": mk.STATIC_SEGMENTS_LEGS,
                                      "legs_only": mk.STATIC_SEGMENTS_LEGS_ONLY}.get(STATIC_MARKERS))
  # no robust loss here: in the offset fit it let the pose ignore a misfitting
  # marker, then moved that marker's offset to where it was -- case study: offsets of
  # two subjects drifted 34-40 mm (median), tracking error +4-5 mm; robust is for
  # tracking only
  np.savez(f[:-4] + ".part.npz", offsets=off, labels=np.array(mk.LABELS), static=STATIC,
           static_share=STATIC_SHARE, static_w=STATIC_W, static_markers=STATIC_MARKERS,
           neck=config.NECK)
  os.replace(f[:-4] + ".part.npz", f)
  return subj, "done"


def ik(name):
  from mocap2myo import ik as mk
  from mocap2myo.model import MYOFULLBODY_XML
  rng = config.IK_FRAMES
  f = os.path.join(OUT, f"{name}{'_' + rng.replace(':', '-') if rng else ''}.npz")
  if os.path.exists(f):
    return name, "cached", 0.0
  t0 = time.time()
  P, rate, s = mk.load_trial(os.path.join(NPZ, f"{name}.npz"), lowpass_hz=LOWPASS_HZ or None)
  if rng:
    a, b = (int(x) for x in rng.split(":"))
    P = P[a:b]
  mm = mk.MarkerModel(neck=config.NECK)
  mm.set_offsets(np.load(os.path.join(OUT, f"{config.subject_of(name)}_offsets.npz"))["offsets"])
  # 5 cm outlier rejection: label swaps (case study, a processed export: LWRB / C7
  # jump onto other markers, p95 > 0.9 m) and a differently placed TIB
  Q, E = mk.ik_trial(mm, P, rate=rate, contact=True, outlier=TRACK_OUTLIER or None,
                     mtp_prior=MTP_PRIOR,
                     anchor_centroid=ANCHOR_CENTROID, pelvis_gap=PELVIS_GAP,
                     robust=ROBUST or None, root_init=ROOT_INIT)
  tmp = f[:-4] + ".part.npz"
  Q, neck = mm.split_qpos(Q)  # the plain model's layout; the neck angles apart
  extra = dict(neck_qpos=neck.astype(np.float32), neck_joints=np.array(mk.NECK)) if mm.neck else {}
  solved = ~np.isnan(E).all(1)  # ik_trial leaves the error of unsolved frames all NaN
  rejected = np.isnan(E) & ~np.isnan(P[..., 0]) & solved[:, None]
  np.savez_compressed(tmp, qpos=Q.astype(np.float32), marker_err=E.astype(np.float32),
                      rejected=rejected, outlier=TRACK_OUTLIER, mtp_prior=MTP_PRIOR,
                      anchor_centroid=ANCHOR_CENTROID, pelvis_gap=PELVIS_GAP,
                      robust=ROBUST, robust_pelvis=False, root_init=ROOT_INIT,
                      lowpass_hz=LOWPASS_HZ,
                      unsolved=int((~solved).sum()), unsolved_frames=~solved,
                      scale=s, rate=rate, labels=np.array(mk.LABELS),
                      frame0=int(rng.split(":")[0]) if rng else 0,
                      model=str(MYOFULLBODY_XML),
                      joint_names=np.array([mm.m.joint(j).name for j in range(mm.m.njnt)
                                            if mm.m.joint(j).name not in mk.NECK]), **extra)
  os.replace(tmp, f)
  return name, (f"median err {1e3 * np.nanmedian(E):.1f} mm, rejected "
                f"{rejected.mean():.4f}, unsolved {int((~solved).sum())}"), time.time() - t0


def main():
  config.usage(__doc__)
  # --calib-only: the subject calibrations (<subject>_offsets.npz in OUT) without
  # the IK, to check them before a full run
  calib_only = "--calib-only" in sys.argv
  args = [a for a in sys.argv[1:] if a != "--calib-only"]
  os.makedirs(OUT, exist_ok=True)
  names = args or config.trials(NPZ)
  subjects = sorted({config.subject_of(n) for n in names})
  ctx = mp.get_context("spawn")
  workers = config.workers()  # ~1-2 GB each
  with ctx.Pool(min(len(subjects), workers)) as pool:
    for r in pool.imap_unordered(calib, subjects):
      print("calib", *r, flush=True)
  if calib_only:
    return
  with ctx.Pool(min(len(names), workers)) as pool:
    for name, msg, dt in pool.imap_unordered(ik, names):
      print("ik", name, msg, f"{dt:.0f}s", flush=True)


if __name__ == "__main__":
  main()
