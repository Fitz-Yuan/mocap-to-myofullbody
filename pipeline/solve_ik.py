"""Marker IK of every trial onto the model: subject calibration, then per-frame IK.

The model is M2M_MODEL (mocap2myo/profiles/, default MyoFullBody), the markers the
dataset's marker set (M2M_MARKER_SET, mocap2myo/markersets/).

Input: <M2M_NPZ_DIR>/<trial>.npz (prepare_markers.py), and with M2M_STATIC=1 the
  static standing trials <M2M_NPZ_DIR>/static/<subject>.npz.
Stage A (per subject): marker-offset calibration (mocap2myo.ik.calibrate) on frames
  drawn from all of the subject's trials -- 40 frames, because one trial mixes
  stooping, walking, turning and overhead reaching. With M2M_SCALE=segments, first
  the subject's segment scale factors (mocap2myo.scaling.measure, from the static
  trial, else quiet standing): <subject>_scale.json; the calibration and the IK then
  run on the model scaled to the subject, and <subject>_model.xml is that model (MJCF,
  with the subject's markers as sites mk_<label>).
Stage B (per trial): per-frame IK with the contact terms (floor, stance sole
  lock, joint speed limit), mocap2myo.ik.ik_trial(contact=True).
Output: <M2M_IK_DIR>/<subject>_offsets.npz, <trial>.npz with
  qpos [T, nq] (scaled-subject frame: markers x scale, model not scaled; with
  M2M_SCALE=segments the subject's own frame, scale 1, on the subject's model),
  marker_err [T, n] (m; NaN = marker missing or rejected), unsolved_frames [T] bool
  (frames the IK did not solve: before the first frame with all four pelvis markers,
  or no marker at all; their qpos repeats a solved pose and marker_err is all NaN),
  unsolved (their count), scale, rate, labels, joint_names, model (the model file),
  model_profile, marker_set and the settings; with M2M_NECK=1 also neck_qpos [T, 3] and
  neck_joints; with M2M_SCALE=segments also scale_mode ("segments", as in the offsets).
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
# Stage B on the GPU (mocap2myo/ik_gpu.py, needs torch with CUDA and warp): "gpu" solves
# the same per-frame problem with the same frame-to-frame recursion, the frames of all
# trials in parallel in one batch. The subject calibration (stage A) and each trial's
# first frame stay on the CPU (worker processes). Default "cpu". (Read here for now;
# its place is config.py's IK section.)
IK_DEVICE = os.environ.get("M2M_IK_DEVICE", "cpu")
if IK_DEVICE not in ("cpu", "gpu"):
  raise SystemExit(f"M2M_IK_DEVICE={IK_DEVICE!r}: expected one of cpu, gpu")


# Standing frames are calibrated with flat feet (ik.calibrate `stand`): the
# TOE marker sits on the shoe's toe box, which otherwise tilts the model foot
# toes-up ~5 deg (case study: toe tip 2 cm above the heel while standing).
# Placement varies most for wand / mid-segment markers, the trunk markers (EMG
# electrodes on the back and abdomen) and the head markers (K5 mask harness):
# a 6 cm cap for every marker pinned them (case study: T10 62 mm median error), so
# these may move up to 15 cm ("loose" in the marker set), foot markers (on thick-soled
# slides, TOE on the toe box; "foot") FOOT_CAP (M2M_FOOT_CAP, 10 cm), the others 6 cm.


def max_shift(ms):
  """Calibration cap per marker (m) from the marker set's groups."""
  return np.where(ms.LOOSE, 0.15, np.where(ms.FOOT, FOOT_CAP, 0.06))


def standing_frames(loaded, ms, per_trial=12, seed=0, min_s=0.3):
  """Flat-foot frames: all four sole markers (both heels and toes) slow and on
  the floor (ik.sole_stance) for >= min_s -- standing, turning on the spot,
  lifting. The whole body need not be still (quiet standing is only ~3 s per
  cycle and was missed entirely for S03 / S05 / S08 with a pelvis-speed test)."""
  from mocap2myo import ik as mk
  rng = np.random.default_rng(seed)
  out = []
  for P, rate in loaded:
    ok = np.all(list(mk.sole_stance(P, rate, ms=ms).values()), 0)
    k = int(min_s * rate)
    ok = np.convolve(ok, np.ones(k), "same") >= k  # inside >= min_s runs
    idx = np.flatnonzero(ok)
    if len(idx):
      out += [P[i] for i in rng.choice(idx, min(per_trial, len(idx)), replace=False)]
  print(f"  flat-foot calibration frames: {len(out)}", flush=True)
  return np.array(out) if out else None


def static_frames(subj, loaded, mm, n=10):
  """[n, n_markers, 3] quietest frames of the subject's static trial (scaled like
  the trials), else quiet upright standing frames from its trials."""
  from mocap2myo import ik as mk
  f = os.path.join(NPZ, "static", f"{subj}.npz")
  if os.path.exists(f):
    P, _, _ = mk.load_trial(f, lowpass_hz=LOWPASS_HZ or None, mm=mm,
                            max_gap_s=config.MAX_GAP_S, gap_fill=config.GAP_FILL)
    sp = np.r_[np.inf, np.nanmax(np.linalg.norm(np.diff(P, axis=0), axis=-1), 1)]
    idx = np.sort(np.argsort(sp)[:max(5 * n, 50)])
    src = "static trial"
  else:
    cand = [(P, np.flatnonzero(mk.standing(P, rate, ms=mm.ms))) for P, rate in loaded]
    P = np.concatenate([P[i] for P, i in cand])
    idx = np.arange(len(P))
    src = "quiet standing in the trials (no static trial)"
  out = P[idx[:: max(1, len(idx) // n)][:n]]
  print(f"  {subj} static frames: {len(out)} from {src}", flush=True)
  return out


def scale_subject(subj, mm0):
  """M2M_SCALE=segments: the subject's scale factors (mocap2myo.scaling) from its static
  frames (static_frames, here in the subject's own size), and the scaled model."""
  from mocap2myo import ik as mk
  from mocap2myo import landmarks
  from mocap2myo import scaling
  loaded = [mk.load_trial(os.path.join(NPZ, f"{t}.npz"), lowpass_hz=LOWPASS_HZ or None, mm=mm0,
                          max_gap_s=config.MAX_GAP_S, gap_fill=config.GAP_FILL)
            for t in config.trials(NPZ) if config.subject_of(t) == subj]
  s = loaded[0][2]  # model height / subject height: the frames back to the subject's size
  P = static_frames(subj, [(P, rate) for P, rate, _ in loaded], mm0) / s
  data = scaling.measure(mm0, P, mm0.model_height / s, keep_uniform=config.SCALE_UNIFORM)
  data["source"] = ("static trial" if os.path.exists(os.path.join(NPZ, "static", f"{subj}.npz"))
                    else "quiet standing in the trials")
  sc = scaling.Scaling(data)
  mm = mk.MarkerModel(mm0.profile, mm0.ms, neck=config.NECK, scaling=sc)
  h = landmarks.compute(mm.profile, m=mm.m).model_height
  data["scaled_model_height_m"] = None if h is None else round(float(h), 4)
  sc.save(config.scale_file(subj))
  print(f"  {subj} scale factors (lateral, anterior, superior): "
        + ", ".join(f"{g} {np.round(f, 3).tolist() if len(set(f)) > 1 else round(f[0], 3)}"
                    for g, f in data["factors"].items())
        + f"; uniform {data['uniform']:.3f}; scaled model height {data['scaled_model_height_m']} m "
        f"(subject {data['height_m']:.3f} m)", flush=True)
  return mm


def calib(subj):
  from mocap2myo import ik as mk
  f = os.path.join(OUT, f"{subj}_offsets.npz")
  if os.path.exists(f):
    config.check_scale(f, np.load(f))
    return subj, "cached"
  mm = config.marker_model(neck=config.NECK)
  if config.SCALE == "segments":
    mm = (config.marker_model(neck=config.NECK, subject=subj)
          if os.path.exists(config.scale_file(subj)) else scale_subject(subj, mm))
  loaded = [mk.load_trial(os.path.join(NPZ, f"{t}.npz"), lowpass_hz=LOWPASS_HZ or None, mm=mm,
                          max_gap_s=config.MAX_GAP_S, gap_fill=config.GAP_FILL)[:2]
            for t in config.trials(NPZ) if config.subject_of(t) == subj]
  trials = [P for P, _ in loaded]
  off = mk.calibrate(mm, trials, n_frames=40, iters=8, max_shift=max_shift(mm.ms),
                     outlier=OUTLIER, stand=standing_frames(loaded, mm.ms), flat=FLAT_W,
                     static=static_frames(subj, loaded, mm) if STATIC else None,
                     static_share=STATIC_SHARE, static_w=STATIC_W,
                     static_segments={"legs": mk.STATIC_SEGMENTS_LEGS,
                                      "legs_only": mk.STATIC_SEGMENTS_LEGS_ONLY}.get(STATIC_MARKERS))
  # no robust loss here: in the offset fit it let the pose ignore a misfitting
  # marker, then moved that marker's offset to where it was -- case study: offsets of
  # two subjects drifted 34-40 mm (median), tracking error +4-5 mm; robust is for
  # tracking only
  extra = {}
  if config.SCALE == "segments":
    from mocap2myo import scaling
    extra["scale_mode"] = config.SCALE
    scaling.write_mjcf(mm.profile, mm.scaling, os.path.join(OUT, f"{subj}_model.xml"),
                       markers=(mm.ms, off, mm.active))
  np.savez(f[:-4] + ".part.npz", offsets=off, labels=np.array(mm.ms.LABELS), static=STATIC,
           static_share=STATIC_SHARE, static_w=STATIC_W, static_markers=STATIC_MARKERS,
           neck=config.NECK, model_profile=mm.profile.name, marker_set=mm.ms.key,
           active=mm.active, **extra)
  os.replace(f[:-4] + ".part.npz", f)
  return subj, "done"


def load(name):
  """Stage B input of one trial: the marker model with its subject's offsets and active
  markers, the markers (frame range M2M_IK_FRAMES), rate, scale and the output file; None
  if that file exists."""
  from mocap2myo import ik as mk
  rng = config.IK_FRAMES
  f = os.path.join(OUT, f"{name}{'_' + rng.replace(':', '-') if rng else ''}.npz")
  if os.path.exists(f):
    return None
  mm = config.marker_model(neck=config.NECK, subject=config.subject_of(name))
  P, rate, s = mk.load_trial(os.path.join(NPZ, f"{name}.npz"), lowpass_hz=LOWPASS_HZ or None, mm=mm,
                             max_gap_s=config.MAX_GAP_S, gap_fill=config.GAP_FILL)
  if rng:
    a, b = (int(x) for x in rng.split(":"))
    P = P[a:b]
  calf = os.path.join(OUT, f"{config.subject_of(name)}_offsets.npz")
  cal = np.load(calf)
  if "model_profile" in cal.files and str(cal["model_profile"]) != mm.profile.name:
    raise SystemExit(f"{name}: the subject's calibration is for {cal['model_profile']}, "
                     f"not M2M_MODEL={mm.profile.name}")
  config.check_scale(calf, cal)
  mm.set_offsets(cal["offsets"])
  if "active" in cal.files:  # markers the calibration switched off (no landmark on the model)
    mm.active &= cal["active"].astype(bool)
    mm.w = np.where(mm.active, mm.w, 0.0)
  return dict(name=name, f=f, mm=mm, P=P, rate=rate, s=s)


def track_args():
  """The tracking IK's settings (ik.ik_trial's keyword arguments; the GPU takes the same).
  5 cm outlier rejection: label swaps (case study, a processed export: LWRB / C7 jump
  onto other markers, p95 > 0.9 m) and a differently placed TIB."""
  return dict(contact=True, outlier=TRACK_OUTLIER or None, mtp_prior=MTP_PRIOR,
              anchor_centroid=ANCHOR_CENTROID, pelvis_gap=PELVIS_GAP, robust=ROBUST or None,
              root_init=ROOT_INIT, smooth=config.IK_SMOOTH, ankle_smooth=config.IK_ANKLE_SMOOTH,
              recover=(os.environ.get("M2M_IK_RECOVER_BASIN") or True) if config.IK_RECOVER else False)


def save(t, mm, active, Q, E, device="cpu"):
  """Write one trial's IK (t from load(); mm its model, active its active markers)."""
  from mocap2myo import ik as mk
  P, f, rng = t["P"], t["f"], config.IK_FRAMES
  tmp = f[:-4] + ".part.npz"
  Q, neck = mm.split_qpos(Q)  # the plain model's layout; the neck angles apart
  extra = dict(neck_qpos=neck.astype(np.float32), neck_joints=np.array(mk.NECK)) if mm.neck else {}
  if device == "gpu":
    extra["ik_device"] = "gpu"
  if config.SCALE == "segments":
    extra["scale_mode"] = config.SCALE
  solved = ~np.isnan(E).all(1)  # ik_trial leaves the error of unsolved frames all NaN
  rejected = np.isnan(E) & ~np.isnan(P[..., 0]) & solved[:, None] & active
  np.savez_compressed(tmp, qpos=Q.astype(np.float32), marker_err=E.astype(np.float32),
                      rejected=rejected, outlier=TRACK_OUTLIER, mtp_prior=MTP_PRIOR,
                      anchor_centroid=ANCHOR_CENTROID, pelvis_gap=PELVIS_GAP,
                      robust=ROBUST, robust_pelvis=False, root_init=ROOT_INIT,
                      lowpass_hz=LOWPASS_HZ,
                      unsolved=int((~solved).sum()), unsolved_frames=~solved,
                      scale=t["s"], rate=t["rate"], labels=np.array(mm.ms.LABELS),
                      frame0=int(rng.split(":")[0]) if rng else 0,
                      model=str(mm.profile.xml), model_profile=mm.profile.name,
                      marker_set=mm.ms.key, active=active,
                      joint_names=np.array([mm.m.joint(j).name for j in range(mm.m.njnt)
                                            if mm.m.joint(j).name not in mk.NECK]), **extra)
  os.replace(tmp, f)
  return (f"median err {1e3 * np.nanmedian(E):.1f} mm, rejected "
          f"{rejected.mean():.4f}, unsolved {int((~solved).sum())}")


def ik(name):
  """Stage B of one trial on the CPU (load, ik.ik_trial, save)."""
  from mocap2myo import ik as mk
  t0 = time.time()
  t = load(name)
  if t is None:
    return name, "cached", 0.0
  mm = t["mm"]
  Q, E = mk.ik_trial(mm, t["P"], rate=t["rate"], **track_args())
  return name, save(t, mm, mm.active, Q, E), time.time() - t0


def load_gpu(name):
  """load() in a worker process for the GPU path, with the trial's first solved frame (the
  first with all pelvis markers) solved there as ik_trial does (ik.first_frame); the marker
  model stays behind (its offsets and active markers come along)."""
  from mocap2myo import ik as mk
  t = load(name)
  if t is None:
    return name, None
  mm, P = t.pop("mm"), t["P"]
  ok = np.flatnonzero(~np.isnan(P[:, mm.ms.PELVIS]).any((1, 2)))
  a = track_args()
  t.update(offsets=mm.m.site_pos[mm.sid].copy(), active=mm.active.copy(),
           first=mk.first_frame(mm, P[ok[0]], outlier=a["outlier"], mtp_prior=a["mtp_prior"],
                                robust=a["robust"]) if len(ok) else None)
  return name, t


def ik_gpu(names, ctx, workers):
  """Stage B on the GPU: every trial in one batch (mocap2myo.ik_gpu.ik_trials_gpu), the
  inputs and first frames prepared in worker processes."""
  from mocap2myo.ik_gpu import ik_trials_gpu
  t0 = time.time()
  with ctx.Pool(min(len(names), workers)) as pool:
    loaded = pool.map(load_gpu, names)
  todo = [(n, t) for n, t in loaded if t is not None]
  for n, t in loaded:
    if t is None:
      print("ik", n, "cached", flush=True)
  if not todo:
    return
  t1 = time.time()
  print(f"ik: {len(todo)} trials loaded ({t1 - t0:.0f}s); solving on the GPU", flush=True)
  mm = config.marker_model(neck=config.NECK)
  st = {}
  out = ik_trials_gpu(mm, [t["P"] for _, t in todo], rate=[t["rate"] for _, t in todo],
                      offsets=[t["offsets"] for _, t in todo], active=[t["active"] for _, t in todo],
                      first=[t["first"] for _, t in todo], stats=st, **track_args())
  for (n, t), (Q, E) in zip(todo, out):
    print("ik", n, save(t, mm, t["active"], Q, E, device="gpu"), flush=True)
  print(f"ik: GPU {time.time() - t1:.0f}s for {sum(len(t['P']) for _, t in todo)} frames "
        f"({st['passes']} passes), {time.time() - t0:.0f}s in all", flush=True)


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
  if IK_DEVICE == "gpu":  # every trial in one batch on the GPU
    if config.SCALE == "segments":  # each subject's own scaled model: one batch per subject
      for subj in subjects:
        ik_gpu([n for n in names if config.subject_of(n) == subj], ctx, workers)
    else:
      ik_gpu(names, ctx, workers)
    return
  with ctx.Pool(min(len(names), workers)) as pool:
    for name, msg, dt in pool.imap_unordered(ik, names):
      print("ik", name, msg, f"{dt:.0f}s", flush=True)


if __name__ == "__main__":
  main()
