"""Marker IK of the MMH Elevator trials onto MyoFullBody (same solver as MMH v2).

Input: ~/Projects/datasets/MMH_Elevator/npz/<S>_<floor>_NE.npz (elev_convert.py).
Stage A (per subject): marker-offset calibration (mmh_ik.calibrate) on frames
  drawn from all of the subject's trials -- more frames than MMH (40 vs 24)
  because one trial mixes stooping, walking, turning and overhead reaching.
Stage B (per trial): per-frame IK with the v2 contact terms (floor, stance sole
  lock, joint speed limit), mmh_ik.ik_trial(contact=True).
Output: ~/Projects/datasets/MMH_Elevator/ik/<S>_offsets.npz, <trial>.npz with
  qpos [T, nq] (scaled-subject frame: markers x scale, model not scaled),
  marker_err [T, 39] (m), scale, rate, labels, joint_names, model.

  OMP_NUM_THREADS=1 uv run python logs/rsl_rl/fullbody_liftbox/scripts/elev_ik.py [trial ...]
ELEV_IK_FRAMES=a:b restricts stage B to a frame range (pilot / timing).
"""

import glob
import multiprocessing as mp
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
NPZ = os.environ.get("ELEV_NPZ_DIR", os.path.join(ROOT, "npz"))  # npz_tak: raw takes
OUT = os.path.expanduser(os.environ.get("ELEV_IK_OUT", os.path.join(ROOT, "ik")))


# Standing frames are calibrated with flat feet (mmh_ik.calibrate `stand`): the
# TOE marker sits on the shoe's toe box, which otherwise tilts the model foot
# toes-up ~5 deg (S09 v2: toe tip 2 cm above the heel while standing).
# Placement varies most for wand / mid-segment markers, the trunk markers (EMG
# electrodes on the back and abdomen) and the head markers (K5 mask harness):
# the 6 cm cap of the MMH calibration pinned them (S09: T10 62 mm median error).
LOOSE = ("THI", "TIB", "UPA", "FRM", "T10", "STRN", "RBAK", "FHD", "BHD")


OUTLIER = 0.05  # m
# tracking IK: marker error beyond which 1-3 markers count as mislabelled (0 = none;
# the calibration keeps OUTLIER)
TRACK_OUTLIER = float(os.environ.get("ELEV_OUTLIER", OUTLIER))
# mmh_ik.ik_trial root_init; "pelvis" since IK v10 (2026-10-13): pelvis markers > 5 cm
# off 954 -> 265 frames (0.36 -> 0.10 %), the rest in deep squats past the model's
# joint limits (left to the dataset QA, elev_build_dataset.py)
ROOT_INIT = os.environ.get("ELEV_ROOT_INIT", "pelvis")
MTP_PRIOR = float(os.environ.get("ELEV_MTP_PRIOR", 0.1))
ANCHOR_CENTROID = os.environ.get("ELEV_ANCHOR_CENTROID", "1") == "1"  # mmh_ik.ik_trial  # see mmh_ik.MarkerModel.solve
PELVIS_GAP = os.environ.get("ELEV_PELVIS_GAP", "solve")  # mmh_ik.ik_trial: box hides the pelvis
# mmh_ik.MarkerModel.solve: Cauchy marker loss (m) instead of the outlier re-solve
# (markers flipping in / out at the threshold made the root jump); 0 = re-solve.
# Tracking IK only (the subject calibration keeps the re-solve, see calib)
ROBUST = float(os.environ.get("ELEV_ROBUST", 0.025))
LOWPASS_HZ = float(os.environ.get("ELEV_LOWPASS_HZ", 6.0))  # mmh_ik.load_trial (0 = raw)
# opt-in: anchor the subject calibration on the static standing trial
# (mmh_ik.calibrate static=...; <NPZ>/static/S<ss>.npz from elev_convert.py, tak
# route). A subject without one (S07) uses quiet-standing frames of its trials.
STATIC = os.environ.get("ELEV_STATIC", "0") == "1"
STATIC_SHARE = float(os.environ.get("ELEV_STATIC_SHARE", 0.5))
STATIC_W = float(os.environ.get("ELEV_STATIC_W", 1.0))
# "legs": the static trial places only pelvis / leg markers (arm and trunk markers
# from the trials alone); "all": every marker (first try, 2026-10-15)
STATIC_MARKERS = os.environ.get("ELEV_STATIC_MARKERS", "legs")


# Foot markers sit on thick-soled slides (TOE on the toe box): S02 / S07 hit
# the 6 cm cap with the flat-foot calibration -> 10 cm (ELEV_FOOT_CAP).
FOOT = ("HEE", "TOE", "ANK")
FOOT_CAP = float(os.environ.get("ELEV_FOOT_CAP", 0.10))
FLAT_W = float(os.environ.get("ELEV_FLAT_W", 30.0))


def max_shift(labels):
  return np.array([0.15 if l.endswith(LOOSE) else FOOT_CAP if l.endswith(FOOT) else 0.06
                   for l in labels])


def standing_frames(loaded, per_trial=12, seed=0, min_s=0.3):
  """Flat-foot frames: all four sole markers (both heels and toes) slow and on
  the floor (mmh_ik.sole_stance) for >= min_s -- standing, turning on the spot,
  lifting. The whole body need not be still (quiet standing is only ~3 s per
  cycle and was missed entirely for S03 / S05 / S08 with a pelvis-speed test)."""
  import mmh_ik as mk
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
  """[n, 39, 3] quietest frames of the subject's static trial (scaled like the
  trials), else quiet upright standing frames from its trials."""
  import mmh_ik as mk
  f = os.path.join(NPZ, "static", f"{subj}.npz")
  if os.path.exists(f):
    P, _, _ = mk.load_trial(f, lowpass_hz=LOWPASS_HZ or None)
    sp = np.r_[np.inf, np.nanmax(np.linalg.norm(np.diff(P, axis=0), axis=-1), 1)]
    idx = np.sort(np.argsort(sp)[:max(5 * n, 50)])
    src = "static trial"
  else:
    import elev_report
    cand = [(P, np.flatnonzero(elev_report.standing(P, rate))) for P, rate in loaded]
    P = np.concatenate([P[i] for P, i in cand])
    idx = np.arange(len(P))
    src = "quiet standing in the trials (no static trial)"
  out = P[idx[:: max(1, len(idx) // n)][:n]]
  print(f"  {subj} static frames: {len(out)} from {src}", flush=True)
  return out


def calib(subj):
  import mmh_ik as mk
  f = os.path.join(OUT, f"{subj}_offsets.npz")
  if os.path.exists(f):
    return subj, "cached"
  loaded = [mk.load_trial(p, lowpass_hz=LOWPASS_HZ or None)[:2]
            for p in sorted(glob.glob(os.path.join(NPZ, f"{subj}_*.npz")))]
  trials = [P for P, _ in loaded]
  mm = mk.MarkerModel()
  off = mk.calibrate(mm, trials, n_frames=40, iters=8, max_shift=max_shift(mk.LABELS),
                     outlier=OUTLIER, stand=standing_frames(loaded), flat=FLAT_W,
                     static=static_frames(subj, loaded) if STATIC else None,
                     static_share=STATIC_SHARE, static_w=STATIC_W,
                     static_segments=mk.STATIC_SEGMENTS_LEGS if STATIC_MARKERS == "legs" else None)
  # no robust loss here: in the offset fit it let the pose ignore a misfitting
  # marker, then moved that marker's offset to where it was -- S05 / S10 offsets
  # drifted 34-40 mm (median), tracking error +4-5 mm; robust is for tracking only
  np.savez(f[:-4] + ".part.npz", offsets=off, labels=np.array(mk.LABELS), static=STATIC,
           static_share=STATIC_SHARE, static_w=STATIC_W, static_markers=STATIC_MARKERS)
  os.replace(f[:-4] + ".part.npz", f)
  return subj, "done"


def ik(name):
  import mmh_ik as mk
  from mocap2myo.model import MYOFULLBODY_XML
  rng = os.environ.get("ELEV_IK_FRAMES")
  f = os.path.join(OUT, f"{name}{'_' + rng.replace(':', '-') if rng else ''}.npz")
  if os.path.exists(f):
    return name, "cached", 0.0
  t0 = time.time()
  P, rate, s = mk.load_trial(os.path.join(NPZ, f"{name}.npz"), lowpass_hz=LOWPASS_HZ or None)
  if rng:
    a, b = (int(x) for x in rng.split(":"))
    P = P[a:b]
  mm = mk.MarkerModel()
  mm.set_offsets(np.load(os.path.join(OUT, f"{name.split('_')[0]}_offsets.npz"))["offsets"])
  # 5 cm outlier rejection: label swaps in the exports (S11 Solid: LWRB / C7
  # jump onto other markers, p95 > 0.9 m) and S01's differently placed TIB
  Q, E = mk.ik_trial(mm, P, rate=rate, contact=True, outlier=TRACK_OUTLIER or None,
                     mtp_prior=MTP_PRIOR,
                     anchor_centroid=ANCHOR_CENTROID, pelvis_gap=PELVIS_GAP,
                     robust=ROBUST or None, root_init=ROOT_INIT)
  tmp = f[:-4] + ".part.npz"
  solved = ~np.isnan(E).all(1)
  rejected = np.isnan(E) & ~np.isnan(P[..., 0]) & solved[:, None]
  np.savez_compressed(tmp, qpos=Q.astype(np.float32), marker_err=E.astype(np.float32),
                      rejected=rejected, outlier=TRACK_OUTLIER, mtp_prior=MTP_PRIOR,
                      anchor_centroid=ANCHOR_CENTROID, pelvis_gap=PELVIS_GAP,
                      robust=ROBUST, robust_pelvis=False, root_init=ROOT_INIT,
                      lowpass_hz=LOWPASS_HZ,
                      unsolved=int((~solved).sum()),
                      scale=s, rate=rate, labels=np.array(mk.LABELS),
                      frame0=int(rng.split(":")[0]) if rng else 0,
                      model=str(MYOFULLBODY_XML),
                      joint_names=np.array([mm.m.joint(j).name for j in range(mm.m.njnt)]))
  os.replace(tmp, f)
  return name, (f"median err {1e3 * np.nanmedian(E):.1f} mm, rejected "
                f"{rejected.mean():.4f}, unsolved {int((~solved).sum())}"), time.time() - t0


def main():
  # --calib-only: the subject calibrations (Sxx_offsets.npz in OUT) without the IK,
  # to check them (elev_ik_fullcheck.py --offsets) before a full rebuild
  calib_only = "--calib-only" in sys.argv
  args = [a for a in sys.argv[1:] if a != "--calib-only"]
  os.makedirs(OUT, exist_ok=True)
  names = args or sorted(os.path.basename(p)[:-4]
                         for p in glob.glob(os.path.join(NPZ, "S*_NE.npz")))
  subjects = sorted({n.split("_")[0] for n in names})
  ctx = mp.get_context("spawn")
  workers = int(os.environ.get("IK_WORKERS", 12))  # ~1-2 GB each (see mmh_batch_ik)
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
