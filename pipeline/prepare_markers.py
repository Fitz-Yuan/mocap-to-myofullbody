"""Convert the MMH Elevator marker exports (our lab) to the MMH npz format.

Input: ~/Projects/datasets/MMH_Elevator/
  markers_export/S<ss>_<Solid|Spring>_NE.xlsx  the collaborators' TRC-as-xlsx export
    (100 Hz, metres, Y-up; body markers labelled, object markers unlabelled
    "Marker 00x" -- identity recovered geometrically below)
  raw_trc/S03_Solid_NE.trc  the only export with labelled Box_1..6 -> box template
  meta/anthropometrics.csv  height (cm), weight (kg)
  (xlsx are read through xlsx_cache/, see elev_xlsx_cache.py)
Output: ~/Projects/datasets/MMH_Elevator/npz/S<ss>_<floor>_NE.npz, Z-up, metres,
  unscaled, readable by mmh_ik.load_trial (markers, marker_labels, marker_rate,
  height_m) plus:
  box_markers [T, 6, 3] (template order), box_pos [T, 3] / box_quat [T, 4] wxyz
  (Kabsch fit of the template; box frame: x long side, z up when resting,
  origin = box centre), box_fit_rms [T], box_size [3] (estimate, see BOX_SIZE),
  shelf_markers [4, 3], elevator_markers [6, 3], floor_markers [k, 3] (median
  positions), floor_z, shelf_top_z, pelvis_fixed [T, 4] (rigid refill flags).

Pelvis: The export filled the long ASIS gaps (box against the belly) by
interpolation; in those frames the pelvis cluster is off rigid by 30-40 mm
(p95, S03 vs the raw TRC). Each frame's 4 pelvis markers are fitted to the
subject's rigid pelvis template; a marker off by > PELVIS_TOL whose removal
makes the other 3 consistent is replaced by its rigid reconstruction.

ELEV_SOURCE=tak (opt-in; default "export" = everything above, unchanged): input =
the raw Motive takes read without Motive (elev_tak_cache.py -> tak_cache/), so
nothing comes from the earlier export. Body labels are Motive's live skeleton
labels (S01: ISB_TO_PIG); the box template comes from the S03 raw take. What the
manual Motive processing adds to a raw take is gap filling (S03: the 5765
samples it has and the raw labels lack are in none of the unlabelled tracks),
which this pipeline does its own way (fix_pelvis, short-gap interpolation in
mmh_ik.load_trial, IK with missing markers). Output goes to ELEV_NPZ_DIR
(default npz_tak/ in this mode, npz/ otherwise). The static standing trials in
tak_cache/static/ become <ELEV_NPZ_DIR>/static/S<ss>.npz (body markers only, same
layout), read by elev_ik.py with ELEV_STATIC=1.

Usage: uv run python logs/rsl_rl/fullbody_liftbox/scripts/elev_convert.py [S09_Solid_NE ...]
"""

import glob
import itertools
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmh_ik as mk  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
SOURCE = os.environ.get("ELEV_SOURCE", "export")
assert SOURCE in ("export", "tak"), SOURCE
OUT = os.environ.get("ELEV_NPZ_DIR",
                     os.path.join(ROOT, "npz_tak" if SOURCE == "tak" else "npz"))
RATE = 100.0
MARKER_R = 0.007  # m, retro-reflective marker radius (centre above the surface)
# Clear plastic storage box (video), 30 lb (user). Marker layout: 3 markers on
# each long side face (2 high, 1 low and 1 high, 2 low); centres 0.23 m apart
# across, 0.33 m along, lowest 0.03 m and highest 0.165 m above the floor.
# Size = estimate from that layout (faces just inside the marker centres, ends
# ~1.5 cm beyond the outermost markers, lid ~2.5 cm above the top markers).
BOX_SIZE = np.array([0.36, 0.23 - 2 * MARKER_R, 0.19])
BOX_MASS = 30 * 0.45359237
PELVIS_TOL = 0.012  # m


def zup(p):
  """Y-up (x, y, z) -> Z-up (x, -z, y), right-handed."""
  return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def read_export(path):
  """xlsx/trc export -> (names, [T, n, 3] Z-up) incl. unlabelled columns."""
  if path.endswith(".xlsx"):  # via elev_xlsx_cache.py (no pandas in .venv)
    c = np.load(os.path.join(ROOT, "xlsx_cache", os.path.basename(path)[:-5] + ".npz"))
    D = c["data"]
    return list(c["names"]), zup(D[:, 2:].reshape(len(D), -1, 3))
  if path.endswith(".tak"):  # via elev_tak_cache.py (olefile not in .venv)
    rel = os.path.relpath(path, os.path.join(ROOT, "raw_tak"))[:-4]  # e.g. static/S03_Anatomy
    c = np.load(os.path.join(ROOT, "tak_cache", rel + ".npz"))
    D = c["data"]
    return list(c["names"]), zup(D[:, 2:].reshape(len(D), -1, 3))
  else:
    lines = open(path).read().splitlines()
    names = lines[3].split("\t")[2:]
    rows = [ln.split("\t") for ln in lines[5:] if ln.strip()]
    w = max(len(r) for r in rows)
    A = np.array([[float(x) if x.strip() else np.nan for x in r] + [np.nan] * (w - len(r))
                  for r in rows])
  A = A[~np.isnan(A[:, 1])]
  cols = [(i, n) for i, n in enumerate(names) if isinstance(n, str) and n.strip()]
  P = np.stack([A[:, 2 + i:5 + i] for i, _ in cols], 1)
  return [n.strip() for _, n in cols], zup(P)


def kabsch(A, B):
  """R, t minimising |R A + t - B| (rows = points)."""
  ca, cb = A.mean(0), B.mean(0)
  U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
  D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
  R = Vt.T @ D @ U.T
  return R, cb - R @ ca


def box_template():
  """Box marker coordinates in the box frame from the labelled S03 export (tak
  mode: the S03 raw take, Motive's rigid-body labels Box_k / Elevator_k)."""
  names, P = read_export(os.path.join(ROOT, "raw_tak", "S03_Solid_NE.tak") if SOURCE == "tak"
                         else os.path.join(ROOT, "raw_trc", "S03_Solid_NE.trc"))
  B = np.stack([P[:, names.index(f"Box_{i}")] for i in range(1, 7)], 1)
  c = B.mean(1)
  v = np.r_[0, np.linalg.norm(np.diff(c, axis=0), axis=1)] * RATE
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


def reject_mislabels(P, tol=0.04, n_partners=3):
  """Label swaps in the exports (S11 Solid: C7 on another marker in 40% of the
  frames, metres away). Each marker's partners = the n most rigid markers
  relative to it over the trial (smallest distance spread, i.e. same / adjacent
  segment); a sample is rejected (NaN) when its distance to >= 2 valid partners
  (to the only one, if one is valid) is off its median by > tol. Two passes:
  the second ignores partners rejected in the first."""
  T, n, _ = P.shape
  D = np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)  # T, n, n
  med = np.nanmedian(D, 0)
  spread = np.nanpercentile(D, 90, 0) - np.nanpercentile(D, 10, 0)
  np.fill_diagonal(spread, np.inf)
  head = np.array([l.endswith("HD") for l in mk.LABELS])
  spread[:, head] = np.inf  # no neck joint in the model, but head markers are
  partners = np.argsort(spread, 1)[:, :n_partners]  # fine as partners of each other
  for i in np.where(head)[0]:
    partners[i] = [j for j in np.argsort(spread[i]) if head[j]][:n_partners]
  dev = np.abs(D - med) > tol
  bad = np.zeros((T, n), bool)
  for _ in range(2):
    for i in range(n):
      pj = partners[i]
      valid = ~np.isnan(D[:, i, pj]) & ~bad[:, pj]
      nd = (dev[:, i, pj] & valid).sum(1)
      nv = valid.sum(1)
      bad[:, i] = (nd >= 2) | ((nv == 1) & (nd == 1))
  P = P.copy()
  P[bad] = np.nan
  return P, bad


# tak mode, S01: Motive labelled the subject's Plug-in-Gait markers live with a
# skeleton template that uses ISB landmark names (the firefighter-study set). Same
# physical markers, other names; the template's single elbow marker took either
# epicondyle name (the static trial also has a real medial one: the lateral
# HLE wins, as Plug-in-Gait's ELB is the lateral epicondyle), and its head names
# are rotated on this subject (assigned by geometry, assign_head). LHGT (left counterpart of RBAK) and TV2 are not in the
# model's marker set.
ISB_TO_PIG = {
  "CV7": "C7", "SJN": "CLAV", "SXS": "STRN", "TV7": "T10", "RHGT": "RBAK",
  "LCAJ": "LSHO", "RCAJ": "RSHO", "LUA": "LUPA", "RUA": "RUPA",
  "LHLE": "LELB", "LHME": "LELB", "RHLE": "RELB", "RHME": "RELB",
  "LRSP": "LWRA", "RRSP": "RWRA", "LUSP": "LWRB", "RUSP": "RWRB",
  "LHM2": "LFIN", "RHM2": "RFIN", "LIAS": "LASI", "RIAS": "RASI",
  "LIPS": "LPSI", "RIPS": "RPSI", "LTH": "LTHI", "RTH": "RTHI",
  "LFLE": "LKNE", "RFLE": "RKNE", "LTTC": "LTIB", "RTTC": "RTIB",
  "LFAL": "LANK", "RFAL": "RANK", "LFCC": "LHEE", "RFCC": "RHEE",
  "LDP1": "LTOE", "RDP1": "RTOE",
  "LAH": "HEAD", "RAH": "HEAD", "LPH": "HEAD", "RPH": "HEAD"}
HEAD_PIG = ("LFHD", "RFHD", "LBHD", "RBHD")


def body_markers(names, A):
  """tak mode: [T, 39, 3] in mk.LABELS order from the take's labels (Plug-in-Gait,
  or ISB_TO_PIG). Two labels on one model marker (elbow): per frame the one seen;
  both seen > 5 mm apart -> the lateral one (HLE). Returns P, head columns
  [T, 4, 3] (label order of HEAD_PIG for Plug-in-Gait takes) and the renamed
  labels."""
  T = len(A)
  cand = {l: [] for l in mk.LABELS}
  head, renamed = [], {}
  isb = any(n in ISB_TO_PIG for n in names)
  for i, n in enumerate(names):
    l = ISB_TO_PIG.get(n) if isb else n
    if l == "HEAD" or (not isb and l in HEAD_PIG):
      head.append((HEAD_PIG.index(l) if l in HEAD_PIG else len(head), i))
    elif l in cand:
      cand[l].append(i)
      if l != n:
        renamed[n] = l
  P = np.full((T, len(mk.LABELS), 3), np.nan)
  for k, l in enumerate(mk.LABELS):
    for i in sorted(cand[l], key=lambda i: not names[i].endswith("HLE")):  # lateral first
      new = np.isnan(P[:, k, 0]) & ~np.isnan(A[:, i, 0])
      P[new, k] = A[new, i]
  H = np.stack([A[:, i] for _, i in sorted(head)], 1) if len(head) == 4 else None
  return P, H, renamed


def assign_head(P, H):
  """Head markers by position: front / back and left / right of the head centre in
  the pelvis heading (ASIS - PSIS, left = LASI - RASI), best of the 24 orders.
  Returns the [T, 4, 3] columns in HEAD_PIG order and that order."""
  g = {l: P[:, mk.LABELS.index(l)] for l in ("LASI", "RASI", "LPSI", "RPSI")}
  fwd = (g["LASI"] + g["RASI"] - g["LPSI"] - g["RPSI"])[:, :2]
  left = (g["LASI"] - g["RASI"])[:, :2]
  ok = ~np.isnan(fwd[:, 0]) & ~np.isnan(left[:, 0]) & ~np.isnan(H[..., 0]).any(1)
  fw = fwd[ok] / np.linalg.norm(fwd[ok], axis=1, keepdims=True)
  lf = left[ok] / np.linalg.norm(left[ok], axis=1, keepdims=True)
  rel = (H[ok] - H[ok].mean(1, keepdims=True))[..., :2]
  pf = np.median(np.einsum("tki,ti->tk", rel, fw), 0)
  pl = np.median(np.einsum("tki,ti->tk", rel, lf), 0)
  want = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])  # LFHD RFHD LBHD RBHD
  order = max(itertools.permutations(range(4)),
              key=lambda o: sum(want[s] @ (pf[o[s]], pl[o[s]]) for s in range(4)))
  return H[:, list(order)], order


def fix_pelvis(P):
  """Rigid refill of pelvis markers that break the cluster or are missing
  while the other 3 are present (see docstring)."""
  idx = [mk.LABELS.index(l) for l in ("LASI", "RASI", "LPSI", "RPSI")]
  X = P[:, idx]
  ok = ~np.isnan(X).any((1, 2))
  D = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)
  med = np.nanmedian(D[ok], 0)
  good = ok & (np.abs(D - med).max((1, 2)) < 0.005)
  ref = X[np.where(good)[0][0]]
  acc = [X[f] @ kabsch(X[f], ref)[0].T + kabsch(X[f], ref)[1] for f in np.where(good)[0][::10]]
  tmpl = np.mean(acc, 0)
  fixed = np.zeros((len(P), 4), bool)
  three = np.isnan(X[..., 0]).sum(1) == 1
  for f in np.where(three)[0]:
    k = int(np.where(np.isnan(X[f, :, 0]))[0][0])
    keep = [j for j in range(4) if j != k]
    R3, t3 = kabsch(tmpl[keep], X[f, keep])
    if np.linalg.norm(tmpl[keep] @ R3.T + t3 - X[f, keep], axis=1).max() < PELVIS_TOL:
      P[f, idx[k]] = tmpl[k] @ R3.T + t3
      fixed[f, k] = True
  for f in np.where(ok)[0]:
    R, t = kabsch(tmpl, X[f])
    e = np.linalg.norm(tmpl @ R.T + t - X[f], axis=1)
    if e.max() < PELVIS_TOL:
      continue
    for k in np.argsort(-e):  # try dropping the worst marker first
      keep = [j for j in range(4) if j != k]
      R3, t3 = kabsch(tmpl[keep], X[f, keep])
      if np.linalg.norm(tmpl[keep] @ R3.T + t3 - X[f, keep], axis=1).max() < PELVIS_TOL:
        P[f, idx[k]] = tmpl[k] @ R3.T + t3
        fixed[f, k] = True
        break
  return P, fixed


def convert(name, tmpl, anthro):
  subj = int(name[1:3])
  names, A = read_export(os.path.join(ROOT, "raw_tak", f"{name}.tak") if SOURCE == "tak"
                         else os.path.join(ROOT, "markers_export", f"{name}.xlsx"))
  T = len(A)
  P = np.full((T, len(mk.LABELS), 3), np.nan)
  missing = []
  rel_note = ""
  if SOURCE == "tak":
    P, H, renamed = body_markers(names, A)
    hk = [mk.LABELS.index(l) for l in HEAD_PIG]
    P[:, hk], order = assign_head(P, H)
    missing = [l for k, l in enumerate(mk.LABELS) if np.isnan(P[:, k, 0]).all()]
    if renamed:
      rel_note += f"ISB labels renamed {renamed}, head by geometry {order}; "
    elif tuple(order) != (0, 1, 2, 3):
      rel_note += f"HEAD LABELS REORDERED BY GEOMETRY {order}; "
  else:
    for k, l in enumerate(mk.LABELS):
      if l in names:
        P[:, k] = A[:, names.index(l)]
      else:
        missing.append(l)
  P, mislabel = reject_mislabels(P)
  P, pel_fixed = fix_pelvis(P)
  # object columns: unlabelled exports ("Marker 00x") / takes (rigid-body markers)
  # (rigid-body markers are named <body>_<k> by optitrack_tak; body labels have no "_")
  is_obj = ((lambda n: n.startswith("Unlabeled") or "_" in n) if SOURCE == "tak"
            else (lambda n: n.startswith(("Marker", "Unlabeled"))))
  valid = {i: np.mean(~np.isnan(A[:, i, 0])) for i, n in enumerate(names) if is_obj(n)}
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
  a = anthro[subj]
  os.makedirs(OUT, exist_ok=True)
  out = os.path.join(OUT, f"{name}.npz")
  np.savez_compressed(
    out[:-4] + ".part.npz", markers=P.astype(np.float32), marker_labels=np.array(mk.LABELS),
    marker_rate=RATE, height_m=a["Height"] / 100, mass_kg=a["Weight"],
    box_markers=Bm.astype(np.float32), box_template=tmpl, box_pos=bpos, box_quat=bquat,
    box_fit_rms=brms, box_size=BOX_SIZE, box_mass=BOX_MASS, shelf_markers=shelf,
    elevator_markers=elev, floor_markers=other, floor_z=floor_z,
    shelf_top_z=float(np.median(shelf[:, 2])) - MARKER_R, pelvis_fixed=pel_fixed,
    mislabel_rejected=mislabel, source=SOURCE)
  os.replace(out[:-4] + ".part.npz", out)
  gaps = np.isnan(P[..., 0]).mean(0)
  print(f"{name}: {T} frames ({T / RATE:.0f} s), body missing {missing or '-'}, {rel_note}"
        f"gaps>1% { {l: round(g, 3) for l, g in zip(mk.LABELS, gaps) if g > 0.01} }, "
        f"mislabel-rejected { {mk.LABELS[i]: round(float(mislabel[:, i].mean()), 3) for i in np.argsort(-mislabel.mean(0))[:4] if mislabel[:, i].mean() > 0.005} }, "
        f"pelvis refilled {pel_fixed.any(1).mean():.3f} of frames; box match rms "
        f"{1e3 * match_rms:.1f} mm ({len(moving)} cols, fragments merged {merged}), fit rms median {1e3 * np.nanmedian(brms):.1f} / p99 "
        f"{1e3 * np.nanpercentile(brms, 99):.1f} mm, box pose missing {np.isnan(brms).mean():.3f}; "
        f"shelf {len(shelf)} top {np.median(shelf[:, 2]) - MARKER_R:.3f}, elevator {len(elev)}, "
        f"floor other {len(other)}, floor z {floor_z:+.3f}")


def convert_static(subj, anthro):
  """tak mode: static standing trial -> <OUT>/static/<subj>.npz (body markers only)."""
  names, A = read_export(os.path.join(ROOT, "raw_tak", "static", f"{subj}_Anatomy.tak"))
  P, H, renamed = body_markers(names, A)
  hk = [mk.LABELS.index(l) for l in HEAD_PIG]
  P[:, hk], order = assign_head(P, H)
  P, mislabel = reject_mislabels(P)
  a = anthro[int(subj[1:3])]
  os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
  out = os.path.join(OUT, "static", f"{subj}.npz")
  np.savez_compressed(out[:-4] + ".part.npz", markers=P.astype(np.float32),
                      marker_labels=np.array(mk.LABELS), marker_rate=RATE,
                      height_m=a["Height"] / 100, mass_kg=a["Weight"], source=SOURCE)
  os.replace(out[:-4] + ".part.npz", out)
  seen = np.mean(~np.isnan(P[..., 0]), 0)
  print(f"{subj} static: {len(P)} frames, renamed {len(renamed)} labels, head {order}, "
        f"markers seen < 95%: { {l: round(float(v), 2) for l, v in zip(mk.LABELS, seen) if v < 0.95} }")


def main():
  rows = np.genfromtxt(os.path.join(ROOT, "meta", "anthropometrics.csv"), delimiter=",",
                       names=True, dtype=None, encoding=None)
  anthro = {int(r["Subject"]): {"Height": float(r["Height"]), "Weight": float(r["Weight"])}
            for r in rows}
  tmpl = box_template()
  print("box template (box frame, m):\n", np.round(tmpl, 3))
  if SOURCE == "tak":
    names = sys.argv[1:] or sorted(os.path.basename(f)[:-4] for f in
                                   glob.glob(os.path.join(ROOT, "tak_cache", "S*_NE.npz")))
  else:
    names = sys.argv[1:] or sorted(os.path.basename(f)[:-5] for f in
                                   glob.glob(os.path.join(ROOT, "markers_export", "*.xlsx")))
  for n in names:
    convert(n, tmpl, anthro)
  if SOURCE == "tak" and not sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(ROOT, "tak_cache", "static", "S*_Anatomy.npz"))):
      convert_static(os.path.basename(p)[:3], anthro)


if __name__ == "__main__":
  main()
