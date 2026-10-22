"""Gap-filled markers as C3D files: what was measured, and where a marker was missing, the
same marker on the subject's calibrated model in the IK pose (a model-based fill, as Vicon
Nexus' Kinematic Fill), shifted to meet the measurement at both ends of the gap (the
measured-minus-model offsets before and after it, blended linearly: a pattern fill with the
model as the donor), for other software (OpenSim, Visual3D, ...).

  python pipeline/export_c3d.py OUT_DIR [trial ...]      (needs ezc3d)

Per trial OUT_DIR/<trial>.c3d: the marker set's markers (its labels, the order of the marker
set), millimetres, the Z-up world of pipeline/prepare_markers.py, at the markers' rate. A
sample is filled where the marker was not measured (or was rejected as mislabelled by the
IK) and the frame was solved; frames the IK did not solve keep their gaps (residual -1).
Residual 0 = measured, 1 = filled; parameter group M2M (FILLED_FRAMES per marker, SOURCE,
IK_DIR) says the same. Head markers are written as measured only (no neck joint: the model
does not follow them). Gaps the reader's short-gap filling closed (ik.load_trial) count as
measured here only if they were in the marker file; the file is written from the raw
markers, not the low-passed ones.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402


def filled_markers(name, ik_dir=None, npz_dir=None):
  """Labels [n], positions [T, n, 3] (m, NaN where nothing), filled [T, n] bool, rate."""
  ik_dir, npz_dir = ik_dir or config.IK_DIR, npz_dir or config.NPZ_DIR
  K = np.load(os.path.join(ik_dir, f"{name}.npz"))
  D = np.load(os.path.join(npz_dir, f"{name}.npz"))
  subj = config.subject_of(name)
  mm = config.run_marker_model(ik_dir, subj)
  mm.set_offsets(np.load(os.path.join(ik_dir, f"{subj}_offsets.npz"))["offsets"])
  labels = list(mm.ms.LABELS)
  have = list(D["marker_labels"])
  X = np.full((len(D["markers"]), len(labels), 3), np.nan)
  for k, lab in enumerate(labels):
    if lab in have:
      X[:, k] = D["markers"][:, have.index(lab)]
  E = K["marker_err"]
  rejected = K["rejected"].astype(bool) if "rejected" in K.files else np.zeros(E.shape, bool)
  X[rejected] = np.nan  # mislabelled samples: the model's marker replaces them
  s = float(K["scale"])
  Q = K["qpos"].astype(np.float64)
  solved = ~np.isnan(E).all(1)
  head = np.asarray(mm.ms.HEAD, bool)
  V = np.full(X.shape, np.nan)
  for t in np.flatnonzero(solved):
    V[t] = mm.markers(Q[t]) / s  # model space -> the subject's metres
  fill = np.zeros(X.shape[:2], bool)
  T = len(X)
  for k in np.flatnonzero(mm.active & ~head):
    idx = np.flatnonzero(np.isnan(X[:, k, 0]) & solved)
    for g in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1) if len(idx) else []:
      # the model's marker, shifted so that it meets the measurement at both ends: the
      # measured-minus-model offset just before and after the gap, blended linearly across
      # it (one end: that end's offset), as a pattern fill with the model as the donor
      a, b = g[0] - 1, g[-1] + 1
      both = lambda t: 0 <= t < T and np.isfinite(X[t, k]).all() and np.isfinite(V[t, k]).all()
      off_a = X[a, k] - V[a, k] if both(a) else None
      off_b = X[b, k] - V[b, k] if both(b) else None
      w = ((g - a) / (b - a))[:, None]
      off = (off_a * (1 - w) + off_b * w if off_a is not None and off_b is not None
             else off_a if off_a is not None else off_b if off_b is not None else 0.0)
      X[g, k] = V[g, k] + off
      fill[g, k] = True
  return labels, X, fill, float(D["marker_rate"])


def write_c3d(path, labels, X, fill, rate, source=""):
  import ezc3d  # optional dependency (pip install -e ".[c3d]")
  T, n = X.shape[:2]
  c = ezc3d.c3d()
  c["parameters"]["POINT"]["RATE"]["value"] = [rate]
  c["parameters"]["POINT"]["LABELS"]["value"] = labels
  c["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
  pts = np.ones((4, n, T))
  pts[:3] = np.transpose(X * 1000.0, (2, 1, 0))
  c["data"]["points"] = pts
  res = np.where(np.isnan(X[..., 0]), -1.0, fill.astype(float)).T[None]  # [1, n, T]
  c["data"]["meta_points"]["residuals"] = res
  c["data"]["meta_points"]["camera_masks"] = np.zeros((7, n, T), bool)
  c.add_parameter("M2M", "FILLED_FRAMES", [int(x) for x in fill.sum(0)])
  c.add_parameter("M2M", "SOURCE", [source])
  c.write(path)


def main():
  config.usage(__doc__)
  if len(sys.argv) < 2:
    raise SystemExit(__doc__.strip())
  out = sys.argv[1]
  os.makedirs(out, exist_ok=True)
  names = sys.argv[2:] or config.trials(config.IK_DIR)
  for n in names:
    labels, X, fill, rate = filled_markers(n)
    write_c3d(os.path.join(out, f"{n}.c3d"), labels, X, fill, rate,
              source=f"mocap2myo IK fill ({config.MODEL}, {os.path.basename(config.IK_DIR)})")
    gaps = np.isnan(X[..., 0]).mean()
    print(f"{n}: {len(labels)} markers, {fill.sum()} samples filled from the model "
          f"({100 * fill.mean():.2f} %), {100 * gaps:.2f} % left empty -> {out}/{n}.c3d")


if __name__ == "__main__":
  main()
