"""Box lifting, marker stage: box pose, shelf, platform and floor from the object markers.

Called by pipeline/prepare_markers.py (M2M_TASK=box_lift). Adds to each marker npz:
  box_markers [T, 6, 3] (template order), box_template [6, 3], box_pos [T, 3] /
  box_quat [T, 4] wxyz (Kabsch fit of the template; box frame: x long side, z up
  when resting, origin = box centre), box_fit_rms [T], box_size [3] (estimate,
  see BOX_SIZE), box_mass, shelf_markers [4, 3], elevator_markers [6, 3] (the
  platform the box is lifted from), floor_markers [k, 3] (median positions),
  floor_z, shelf_top_z.

Template: the box marker coordinates in the box frame, from the one trial whose
box markers are labelled (TEMPLATE_TRIAL; tak route: its raw take, with Motive's
rigid-body labels Box_k / Elevator_k; export route: its TRC export, Box_1..6). In
every other trial the box markers are unlabelled: the moving object columns are
assigned to template slots by the best rigid fit.
"""

import itertools
import os

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from mocap2myo.kinematics import kabsch
from pipeline import config

TEMPLATE_TRIAL = "S03_Solid_NE"  # the only export with labelled Box_1..6
MARKER_R = 0.007  # m, retro-reflective marker radius (centre above the surface)
# Clear plastic storage box (video), 30 lb (user). Marker layout: 3 markers on
# each long side face (2 high, 1 low and 1 high, 2 low); centres 0.23 m apart
# across, 0.33 m along, lowest 0.03 m and highest 0.165 m above the floor.
# Size = estimate from that layout (faces just inside the marker centres, ends
# ~1.5 cm beyond the outermost markers, lid ~2.5 cm above the top markers).
BOX_SIZE = np.array([0.36, 0.23 - 2 * MARKER_R, 0.19])
BOX_MASS = 30 * 0.45359237


def box_template(read):
  """Box marker coordinates in the box frame from the labelled template trial
  (read = prepare_markers.read_export)."""
  names, P, rate = read(os.path.join(config.RAW_TAK_DIR, f"{TEMPLATE_TRIAL}.tak")
                        if config.SOURCE == "tak"
                        else os.path.join(config.RAW_TRC_DIR, f"{TEMPLATE_TRIAL}.trc"))
  B = np.stack([P[:, names.index(f"Box_{i}")] for i in range(1, 7)], 1)
  c = B.mean(1)
  v = np.r_[0, np.linalg.norm(np.diff(c, axis=0), axis=1)] * rate
  rest = np.where(v[:100] < 0.01)[0]  # resting on the elevator floor at the start
  X = np.nanmean(B[rest], 0)
  el = np.stack([np.nanmedian(P[:, names.index(f"Elevator_{i}")], 0) for i in range(1, 7)])
  floor = float(np.median(el[:, 2])) - MARKER_R
  u = np.linalg.svd(X[:, :2] - X[:, :2].mean(0))[2][0]  # long horizontal axis
  R = np.array([[u[0], u[1], 0], [-u[1], u[0], 0], [0, 0, 1]])
  L = (X - X.mean(0)) @ R.T
  # box centre in that frame: mid-length, mid-way between the side faces,
  # half the height above the floor
  ctr = np.array([(L[:, 0].max() + L[:, 0].min()) / 2, (L[:, 1].max() + L[:, 1].min()) / 2,
                  floor - X.mean(0)[2] + BOX_SIZE[2] / 2])
  return L - ctr


def match_box(cols, tmpl):
  """Assign k (3..6) well-tracked moving columns [T, k, 3] to template slots
  (best rigid fit over all injective assignments) -> [T, 6, 3], rms."""
  k = cols.shape[1]
  ok = ~np.isnan(cols).any((1, 2))
  frames = np.where(ok)[0][:: max(1, ok.sum() // 50)][:50]
  best = None
  for slots in itertools.permutations(range(6), k):
    e = 0.0
    for f in frames:
      R, t = kabsch(tmpl[list(slots)], cols[f])
      e += np.sqrt(np.mean(np.sum((tmpl[list(slots)] @ R.T + t - cols[f]) ** 2, 1)))
    e /= len(frames)
    if best is None or e < best[0]:
      best = (e, slots)
  Bm = np.full((len(cols), 6, 3), np.nan)
  Bm[:, list(best[1])] = cols
  return Bm, best[0]


def merge_fragments(Bm, frags, tmpl, tol=0.03):
  """Fill empty template slots from fragmented moving columns: a fragment
  belongs to the slot nearest to its median position in the box frame."""
  pos, quat, _ = box_pose(Bm, tmpl)
  used = []
  for F in frags:
    ok = ~np.isnan(F[:, 0]) & ~np.isnan(pos[:, 0])
    if ok.sum() < 10:
      continue
    R = sRot.from_quat(quat[ok][:, [1, 2, 3, 0]]).as_matrix()
    loc = np.median(np.einsum("nji,nj->ni", R, F[ok] - pos[ok]), 0)
    dist = np.linalg.norm(tmpl - loc, axis=1)
    k = int(np.argmin(dist))
    if dist[k] < tol:
      fill = np.isnan(Bm[:, k, 0]) & ~np.isnan(F[:, 0])
      Bm[fill, k] = F[fill]
      used.append((k, int(fill.sum())))
  return Bm, used


def box_pose(Bm, tmpl, tol=0.012):
  """Per-frame Kabsch pose; a marker off by > tol whose removal makes the rest
  consistent (ghost / swapped label) is dropped for that frame."""
  T = len(Bm)
  pos, quat, rms = np.full((T, 3), np.nan), np.full((T, 4), np.nan), np.full(T, np.nan)
  for f in range(T):
    ok = ~np.isnan(Bm[f, :, 0])
    if ok.sum() < 3:
      continue
    R, t = kabsch(tmpl[ok], Bm[f, ok])
    e = np.linalg.norm(tmpl[ok] @ R.T + t - Bm[f, ok], axis=1)
    if e.max() > tol and ok.sum() > 3:
      keep = np.where(ok)[0]
      keep = np.delete(keep, np.argmax(e))
      R2, t2 = kabsch(tmpl[keep], Bm[f, keep])
      e2 = np.linalg.norm(tmpl[keep] @ R2.T + t2 - Bm[f, keep], axis=1)
      if e2.max() < tol:
        ok, R, t, e = np.isin(np.arange(6), keep), R2, t2, e2
    pos[f], quat[f] = t, sRot.from_matrix(R).as_quat()[[3, 0, 1, 2]]
    rms[f] = np.sqrt(np.mean(e ** 2))
  return pos, quat, rms


def prepare_setup(read):
  """Once per run: the box template."""
  tmpl = box_template(read)
  print("box template (box frame, m):\n", np.round(tmpl, 3))
  return tmpl


def prepare_trial(name, names, A, cols, rate, tmpl):
  """Box pose and scene of one trial from its object columns A[:, cols] (Z-up, m).
  Returns the extra npz fields and a log line."""
  valid = {i: np.mean(~np.isnan(A[:, i, 0])) for i in cols}
  obj = [i for i, v in valid.items() if v > 0.02]
  rng = {i: np.nanmax(np.nanmax(A[:, i], 0) - np.nanmin(A[:, i], 0)) for i in obj}
  moving = [i for i in obj if rng[i] > 1.0 and valid[i] > 0.5]
  frags = [i for i in obj if rng[i] > 1.0 and valid[i] <= 0.5]
  static = {i: np.nanmedian(A[:, i], 0) for i in obj if rng[i] < 0.1 and valid[i] > 0.5}
  if not 3 <= len(moving) <= 6:
    raise RuntimeError(f"{name}: {len(moving)} moving object columns, expected 3-6")
  Bm, match_rms = match_box(A[:, moving], tmpl)
  Bm, merged = merge_fragments(Bm, [A[:, i] for i in frags], tmpl)
  bpos, bquat, brms = box_pose(Bm, tmpl)
  start = np.nanmedian(bpos[:50, :2], 0)
  shelf = np.array([p for p in static.values() if p[2] > 1.2])
  floor_m = [p for p in static.values() if p[2] < 0.1]
  elev = np.array([p for p in floor_m if np.linalg.norm(p[:2] - start) < 1.8])
  other = np.array([p for p in floor_m if np.linalg.norm(p[:2] - start) >= 1.8])
  floor_z = float(np.median(elev[:, 2])) - MARKER_R
  fields = dict(
    box_markers=Bm.astype(np.float32), box_template=tmpl, box_pos=bpos, box_quat=bquat,
    box_fit_rms=brms, box_size=BOX_SIZE, box_mass=BOX_MASS, shelf_markers=shelf,
    elevator_markers=elev, floor_markers=other, floor_z=floor_z,
    shelf_top_z=float(np.median(shelf[:, 2])) - MARKER_R)
  note = (f"box match rms {1e3 * match_rms:.1f} mm ({len(moving)} cols, fragments merged "
          f"{merged}), fit rms median {1e3 * np.nanmedian(brms):.1f} / p99 "
          f"{1e3 * np.nanpercentile(brms, 99):.1f} mm, box pose missing {np.isnan(brms).mean():.3f}; "
          f"shelf {len(shelf)} top {np.median(shelf[:, 2]) - MARKER_R:.3f}, elevator {len(elev)}, "
          f"floor other {len(other)}, floor z {floor_z:+.3f}")
  return fields, note
