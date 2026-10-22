"""Motion from video: GVHMR's reconstruction as virtual markers, then the marker pipeline.

`mocap2myo video` (mocap2myo/cli.py) runs, per video:
  1. prepare_video: re-encode at a constant 30 frames per second with the phone's rotation
     applied (GVHMR reads every frame as 1/30 s and ignores the rotation flag of phone videos);
  2. run_gvhmr: GVHMR (installed separately: it and the SMPL-X model files have their own
     licences and are not part of this repository) -> SMPL-X parameters in a gravity-aligned
     world;
  3. export_markers: mocap2myo/gvhmr_markers.py in GVHMR's Python -> 32 virtual markers on the
     SMPL mesh (markersets/smpl_opencap.toml), and the same body standing in SMPL's zero pose;
  4. to_trials: Z up; scaled to the subject's height (monocular video fixes the size only up
     to a scale: the measured height is the better scale); the floor at z = 0 with GVHMR's slow
     vertical drift removed (floor="flat"; "as-is" keeps its heights, for stairs or slopes);
     the foot in contact with the floor held in place (foot_lock); TRC files of the trial and
     of a static trial (the standing pose), the subject's height and mass, and a dataset
     profile. From there it is the marker pipeline (mocap2myo run).
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

FPS = 30.0  # GVHMR's frame rate
Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # (x, y, z) -> (x, -z, y)
DRIFT_WINDOW_S, DRIFT_HZ, CONTACT_BAND = 1.0, 0.25, 0.02  # floor="flat": s, Hz, m (floor_drift)
CONTACT_H, CONTACT_SOFT, CONTACT_V = 0.05, 0.015, 0.25  # foot_lock: m, m, m/s
LOCK_HZ = 6.0  # foot_lock: the correction's low-pass, as the markers'
NAME = re.compile(r"[A-Za-z0-9-]+")


def _run(cmd, **kw):
  print("  $ " + " ".join(str(c) for c in cmd), flush=True)
  subprocess.run([str(c) for c in cmd], check=True, **kw)


def prepare_video(src, dst):
  """src -> dst (mp4): constant 30 frames per second, the rotation flag applied, no audio."""
  if shutil.which("ffmpeg") is None:
    raise SystemExit("ffmpeg is needed (https://ffmpeg.org; conda install ffmpeg, apt install ffmpeg)")
  _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", f"fps={FPS:g}", "-c:v", "libx264",
        "-crf", "17", "-pix_fmt", "yuv420p", "-an", dst])


def run_gvhmr(video, gvhmr, python, out_root, static_cam=False):
  """GVHMR's demo on video (in GVHMR's folder, its Python) -> its hmr4d_results.pt."""
  demo = os.path.join(gvhmr, "tools", "demo", "demo.py")
  if not os.path.exists(demo):
    raise SystemExit(f"--gvhmr {gvhmr}: no tools/demo/demo.py (the GVHMR folder; see the docs, Video)")
  env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")  # torch >= 2.6 and GVHMR's checkpoints
  _run([python, demo, "--video", os.path.abspath(video), "--output_root", os.path.abspath(out_root)]
       + (["-s"] if static_cam else []), cwd=gvhmr, env=env)
  res = os.path.join(out_root, os.path.splitext(os.path.basename(video))[0], "hmr4d_results.pt")
  if not os.path.exists(res):
    raise SystemExit(f"GVHMR wrote no {res}")
  return res


def export_markers(result, gvhmr, python, out_npz, betas=None):
  """gvhmr_markers.py on a GVHMR result, in GVHMR's Python and folder (betas: a .npy of the
  body shape to use instead of this video's)."""
  env = dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
  _run([python, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gvhmr_markers.py"),
        os.path.abspath(result), os.path.abspath(out_npz)] + ([os.path.abspath(betas)] if betas else []),
       cwd=gvhmr, env=env)


def _lowpass(x, rate, hz, order=2):
  """Zero-phase Butterworth, padded by three periods of the cut-off (filtfilt's default, a
  few samples, leaves transients at the ends of a slow filter)."""
  from scipy.signal import butter, filtfilt
  b, a = butter(order, min(hz / (rate / 2), 0.99), btype="low")
  if len(x) <= 3 * max(len(a), len(b)):
    return x
  return filtfilt(b, a, x, axis=0, padlen=min(len(x) - 1, int(3 * rate / hz)))


def floor_drift(sole_z, rate):
  """The floor's height under the subject per frame. The frames with a foot on the floor are
  those whose lowest sole point is within CONTACT_BAND of its minimum over DRIFT_WINDOW_S
  (some foot is on the floor within it in walking, turning, lifting; jumps are shorter); the
  floor follows the sole height through those frames, smoothed below DRIFT_HZ (the slow drift
  of the reconstruction only). The minimum alone would lag behind a drifting floor."""
  w = max(1, int(round(DRIFT_WINDOW_S * rate / 2)))
  pad = np.pad(sole_z, w, mode="edge")
  env = np.array([pad[i:i + 2 * w + 1].min() for i in range(len(sole_z))])
  on = sole_z < env + CONTACT_BAND
  t = np.arange(len(sole_z))
  return _lowpass(np.interp(t, t[on], sole_z[on]), rate, DRIFT_HZ)


def foot_lock(X, names, rate, sole_offset):
  """X [T, n, 3] (Z up, floor at 0) -> X shifted horizontally per frame so the foot in contact
  with the floor does not slide, and a summary. A monocular reconstruction can move the body
  faster than its stance foot allows (the foot skates); the joint angles are right, the path is
  not. Every frame is translated by an accumulated offset o(t) = o(t-1) - sum_f w_f dp_f /
  max(sum_f w_f, 1), dp_f the horizontal step of foot f (mid heel-toe) since the last frame,
  w_f = low_f x softmin_f(speed): of two low feet the slower is the stance foot (a swinging
  foot also passes near the floor), both slow and low = double support, none low = flight
  (no correction). The offset is low-passed at LOCK_HZ. sole_offset: {marker: its height
  above the sole when standing}."""
  i = {n: k for k, n in enumerate(names)}
  T = len(X)
  pts, low, speed = {}, {}, {}
  for s in "lr":
    heel, toe = i[f"{s}_calc"], i[f"{s}_toe"]
    pts[s] = 0.5 * (X[:, heel, :2] + X[:, toe, :2])
    h = np.minimum(X[:, heel, 2] - sole_offset[f"{s}_calc"], X[:, toe, 2] - sole_offset[f"{s}_toe"])
    low[s] = 1.0 / (1.0 + np.exp((h - CONTACT_H) / CONTACT_SOFT))
    speed[s] = np.linalg.norm(np.gradient(pts[s], 1.0 / rate, axis=0), axis=-1)
  e = np.stack([np.exp(-speed[s] / CONTACT_V) for s in "lr"])
  soft = e / np.maximum(e.sum(0, keepdims=True), 1e-12)
  w = {s: np.clip(low[s] * soft[k] * 2.0, 0.0, 1.0) for k, s in enumerate("lr")}
  off = np.zeros((T, 2))
  for t in range(1, T):
    d, ws = np.zeros(2), 0.0
    for s in "lr":
      wt = 0.5 * (w[s][t] + w[s][t - 1])
      d += wt * (pts[s][t] - pts[s][t - 1])
      ws += wt
    off[t] = off[t - 1] - d / max(ws, 1.0)
  off = _lowpass(off, rate, LOCK_HZ, order=4)
  out = X.copy()
  out[..., :2] += off[:, None, :]

  def slide(Y):  # median horizontal speed of the slower foot while it is on the floor (m/s)
    v = []
    for s in "lr":
      p = 0.5 * (Y[:, i[f"{s}_calc"], :2] + Y[:, i[f"{s}_toe"], :2])
      sp = np.linalg.norm(np.gradient(p, 1.0 / rate, axis=0), axis=-1)
      v.append(sp[w[s] > 0.5])
    v = np.concatenate(v)
    return float(np.median(v)) if len(v) else float("nan")
  return out, dict(slide_before_m_s=slide(X), slide_after_m_s=slide(out),
                   contact_frac={s: float((w[s] > 0.5).mean()) for s in "lr"},
                   shift_end_m=float(np.linalg.norm(off[-1])))


def to_trials(npz, height_cm=None, floor="flat", lock=True):
  """gvhmr_markers.py's output -> (names, trial markers [T, n, 3], static markers [n, 3], info);
  Z up, metres, the subject's size, the floor at z = 0."""
  d = np.load(npz)
  names = [str(x) for x in d["names"]]
  stature = float(d["stature"])
  scale = (height_cm / 100.0) / stature if height_cm else 1.0
  X = d["markers"].astype(np.float64) @ Y_UP_TO_Z_UP.T * scale
  sole = d["lowest"].astype(np.float64) @ Y_UP_TO_Z_UP.T * scale
  S = d["static"].astype(np.float64) @ Y_UP_TO_Z_UP.T * scale
  s0 = (d["static_lowest"].astype(np.float64) @ Y_UP_TO_Z_UP.T * scale)[2]
  S[:, 2] -= s0
  S[:, :2] -= S[[names.index(f"{s}_{a}SIS") for s in "lr" for a in "AP"], :2].mean(0)  # pelvis over the origin
  info = dict(frames=len(X), rate=FPS, stature_video_m=round(stature, 3), scale=round(scale, 4),
              height_m=round(stature * scale, 3), floor=floor, foot_lock=lock)
  if floor == "flat":
    f = floor_drift(sole[:, 2], FPS)
    info["floor_drift_m"] = round(float(f.max() - f.min()), 3)
  else:
    f = np.full(len(X), np.percentile(sole[:, 2], 5))
  X[..., 2] -= f[:, None]
  if lock:
    X, li = foot_lock(X, names, FPS, {n: S[names.index(n), 2] for n in ("l_calc", "l_toe", "r_calc", "r_toe")})
    info.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in li.items()})
  return names, X, S, info


def write_trc(path, names, X, rate):
  from mocap2myo.tak import write_trc as _write
  _write(path, dict(names=names, xyz=X, rate=rate))


PROFILE = '''# Dataset profile written by "mocap2myo video": motion reconstructed from video (GVHMR),
# as virtual markers on the SMPL mesh. Joint angles only: the joint loads from video are not
# validated (no force plates, and the reconstruction's accelerations are not measured ones).
name = "{name}"
description = "Motion from video (GVHMR), virtual SMPL markers"

[paths]
root = "{root}"
raw_dir = "raw"

[input]
input = "trc"
up = "z"
units = "m"
rate = "native"

[naming]
raw_trials = "*/*.trc"
trial_regex = '(?P<subject>[A-Za-z0-9-]+)/(?P=subject)_(?!static\\.trc$)[A-Za-z0-9-]+\\.trc'
trial_name = "{{stem}}"
trial_glob = "*"
subject_sep = "_"
raw_static = "*/*_static.trc"
static_regex = '(?P<subject>[A-Za-z0-9-]+)/(?P=subject)_static\\.trc'
static_name = "{{subject}}_static"

[subjects]
anthro = "csv"
anthro_csv = "anthropometrics.csv"
anthro_columns = "Subject,Height,Weight"
marker_set = "smpl_opencap"

[task]
task = "none"

[ik]
static = true
static_markers = "legs"
'''


def add_video(video, subject, mass_kg, study, height_cm=None, gvhmr=None, python="python",
              static_cam=False, result=None, floor="flat", lock=True):
  """One video into the study folder: raw/<subject>/<subject>_<video>.trc, the subject's static
  trial, anthropometrics.csv and <study>/<study name>.toml. Returns the profile's path."""
  if not NAME.fullmatch(subject):
    raise SystemExit(f"--subject {subject!r}: letters, digits and '-' only (the '_' separates the subject)")
  stem = "-".join(NAME.findall(os.path.splitext(os.path.basename(video))[0])) or "video"
  trial = f"{subject}_{stem}"
  if not gvhmr:
    raise SystemExit("--gvhmr: GVHMR's folder (or M2M_GVHMR); see the docs, Video")
  work = os.path.join(study, "gvhmr")
  os.makedirs(work, exist_ok=True)
  if result is None:
    mp4 = os.path.join(work, f"{trial}.mp4")
    print(f"{video}: re-encoding at {FPS:g} fps -> {mp4}", flush=True)
    prepare_video(video, mp4)
    print("GVHMR (this takes a few minutes per minute of video on a GPU)", flush=True)
    result = run_gvhmr(mp4, gvhmr, python, work, static_cam)
  npz = os.path.join(work, f"{trial}_markers.npz")
  raw = os.path.join(study, "raw", subject)
  os.makedirs(raw, exist_ok=True)
  shape = os.path.join(raw, f"{subject}_betas.npy")  # the subject's body: from their first video
  export_markers(result, gvhmr, python, npz, betas=shape if os.path.exists(shape) else None)
  if not os.path.exists(shape):
    np.save(shape, np.load(npz)["betas"])
  names, X, S, info = to_trials(npz, height_cm, floor, lock)
  write_trc(os.path.join(raw, f"{trial}.trc"), names, X, FPS)
  write_trc(os.path.join(raw, f"{subject}_static.trc"), names, np.repeat(S[None], int(FPS), 0), FPS)
  info.update(video=os.path.abspath(video), gvhmr_result=os.path.abspath(result), subject=subject,
              mass_kg=mass_kg, height_given_cm=height_cm)
  json.dump(info, open(os.path.join(raw, f"{trial}.json"), "w"), indent=1)
  # heights and masses: one row per subject
  csv = os.path.join(study, "anthropometrics.csv")
  rows = {}
  if os.path.exists(csv):
    for line in open(csv).read().splitlines()[1:]:
      if line.strip():
        rows[line.split(",")[0]] = line
  rows[subject] = f"{subject},{100 * info['height_m']:.1f},{mass_kg:g}"
  open(csv, "w").write("Subject,Height,Weight\n" + "\n".join(rows[k] for k in sorted(rows)) + "\n")
  name = os.path.basename(os.path.abspath(study))
  prof = os.path.join(study, f"{name}.toml")
  if not os.path.exists(prof):
    open(prof, "w").write(PROFILE.format(name=name, root=os.path.abspath(study)))
  print(f"{trial}: {info['frames']} frames at {FPS:g} Hz, height {info['height_m']} m (video "
        f"{info['stature_video_m']} m), floor drift removed {info.get('floor_drift_m', 0)} m, "
        f"foot slide {info.get('slide_before_m_s', float('nan')):.3f} -> {info.get('slide_after_m_s', float('nan')):.3f} m/s",
        flush=True)
  return prof


if __name__ == "__main__":
  sys.exit("use: mocap2myo video (mocap2myo/cli.py)")
