"""Marker conditioning: labelled marker trajectories -> one marker npz per trial.

Input (M2M_SOURCE, see config.py):
  tak     the raw Motive takes read without Motive (cache_takes.py -> tak_cache/),
          the route run_all.sh uses. Body labels are Motive's live skeleton labels:
          the marker set's own labels or one of its alias sets
          (mocap2myo.markerset; case study, S01: ISB names).
  export  (default) markers_export/<trial>.xlsx, TRC-as-xlsx exports made in Motive
          (100 Hz, metres, Y-up; body markers labelled, object markers unlabelled
          "Marker 00x"), read through xlsx_cache/<trial>.npz.
  meta/anthropometrics.csv  height (cm), mass (kg) per subject (config.ANTHRO_*)
Output: <M2M_NPZ_DIR>/<trial>.npz, Z-up, metres, unscaled, readable by
  mocap2myo.ik.load_trial: markers [T, n, 3] (marker set order), marker_labels,
  marker_rate, height_m, mass_kg, pelvis_fixed [T, 4] (rigid refill flags),
  mislabel_rejected [T, n], source, plus the task's fields (M2M_TASK; box_lift:
  the box pose, shelf, platform and floor, tasks/box_lift/box_fit.py). The static
  standing trials in tak_cache/static/ become <M2M_NPZ_DIR>/static/<subject>.npz
  (body markers only, same layout), read by solve_ik.py with M2M_STATIC=1.

Per trial: labels mapped onto the marker set (alias sets, head markers by
position), mislabelled samples rejected, pelvis markers refilled as a rigid body.

Pelvis: an export may fill long ASIS gaps (box against the belly) by
interpolation; in those frames the pelvis cluster is off rigid by 30-40 mm (case
study: p95, S03 vs the raw TRC). Each frame's 4 pelvis markers are fitted to the
subject's rigid pelvis template; a marker off by > PELVIS_TOL whose removal
makes the other 3 consistent is replaced by its rigid reconstruction.

What the manual Motive processing adds to a raw take is gap filling (case study,
S03: the 5765 samples it has and the raw labels lack are in none of the
unlabelled tracks), which this pipeline does its own way (fix_pelvis, short-gap
interpolation in ik.load_trial, IK with missing markers).

  python pipeline/prepare_markers.py [trial ...]
(with trial names, the static trials are left as they are)
"""

import glob
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mocap2myo import markerset as ms  # noqa: E402
from mocap2myo.kinematics import kabsch  # noqa: E402
from pipeline import config  # noqa: E402

SOURCE = config.SOURCE
OUT = config.NPZ_DIR
RATE = 100.0  # export route: the exports' frame rate (takes carry their own)
PELVIS_TOL = 0.012  # m


def zup(p):
  """Y-up (x, y, z) -> Z-up (x, -z, y), right-handed."""
  return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def read_export(path):
  """Take / xlsx export / TRC -> (names, [T, n, 3] Z-up, rate) incl. unlabelled
  columns. Takes and xlsx exports are read through their caches (cache_takes.py,
  xlsx_cache/)."""
  if path.endswith(".xlsx"):  # xlsx_cache/<name>.npz: names [n], data [T, 2 + 3n]
    c = np.load(os.path.join(config.XLSX_CACHE_DIR, os.path.basename(path)[:-5] + ".npz"))
    D = c["data"]
    return list(c["names"]), zup(D[:, 2:].reshape(len(D), -1, 3)), RATE
  if path.endswith(".tak"):  # via cache_takes.py
    rel = os.path.relpath(path, config.RAW_TAK_DIR)[:-4]  # e.g. static/S03_Anatomy
    c = np.load(os.path.join(config.TAK_CACHE_DIR, rel + ".npz"))
    D = c["data"]
    return list(c["names"]), zup(D[:, 2:].reshape(len(D), -1, 3)), float(c["rate"])
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
  return [n.strip() for _, n in cols], zup(P), RATE


def object_column(n):
  """A column that is not a body marker: unlabelled tracks, rigid-body markers
  (named <body>_<k> by mocap2myo.tak; body labels have no "_"), the unlabelled
  "Marker 00x" columns of an export."""
  if SOURCE == "tak":
    return n.startswith("Unlabeled") or "_" in n
  return n.startswith(("Marker", "Unlabeled"))


def reject_mislabels(P, tol=0.04, n_partners=3):
  """Label swaps (case study, S11 Solid export: C7 on another marker in 40% of the
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
  head = ms.HEAD
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


def body_markers(names, A):
  """tak route: [T, n, 3] in marker set order from the take's labels (the set's own
  labels, or one of its alias sets, ms.ALIASES). Two labels on one model marker
  (case study: the ISB elbow): per frame the one seen; both seen -> the one in
  ms.PREFER (the lateral epicondyle). Returns P, the head columns [T, h, 3] (in
  ms.HEAD_LABELS order for takes with the set's own labels; reordered by position
  in assign_head) and the renamed labels."""
  T = len(A)
  cand = {l: [] for l in ms.LABELS}
  head, renamed = [], {}
  alias = next((a for a in ms.ALIASES.values() if any(n in a for n in names)), None)
  for i, n in enumerate(names):
    l = alias.get(n) if alias is not None else n
    if l == "HEAD" or (alias is None and l in ms.HEAD_LABELS):
      head.append((ms.HEAD_LABELS.index(l) if l in ms.HEAD_LABELS else len(head), i))
    elif l in cand:
      cand[l].append(i)
      if l != n:
        renamed[n] = l
  P = np.full((T, len(ms.LABELS), 3), np.nan)
  for k, l in enumerate(ms.LABELS):
    for i in sorted(cand[l], key=lambda i: names[i] not in ms.PREFER):  # preferred first
      new = np.isnan(P[:, k, 0]) & ~np.isnan(A[:, i, 0])
      P[new, k] = A[new, i]
  H = (np.stack([A[:, i] for _, i in sorted(head)], 1) if len(head) == len(ms.HEAD_LABELS)
       else None)
  return P, H, renamed


def assign_head(P, H):
  """Head markers by position: front / back and left / right of the head centre in
  the pelvis heading (ASIS - PSIS, left = left - right ASIS), best of all orders
  (layout: ms.MARKER_SET["head"]). Returns the [T, h, 3] columns in ms.HEAD_LABELS
  order and that order."""
  lasi, rasi, lpsi, rpsi = (P[:, k] for k in ms.PELVIS)
  fwd = (lasi + rasi - lpsi - rpsi)[:, :2]
  left = (lasi - rasi)[:, :2]
  ok = ~np.isnan(fwd[:, 0]) & ~np.isnan(left[:, 0]) & ~np.isnan(H[..., 0]).any(1)
  fw = fwd[ok] / np.linalg.norm(fwd[ok], axis=1, keepdims=True)
  lf = left[ok] / np.linalg.norm(left[ok], axis=1, keepdims=True)
  rel = (H[ok] - H[ok].mean(1, keepdims=True))[..., :2]
  pf = np.median(np.einsum("tki,ti->tk", rel, fw), 0)
  pl = np.median(np.einsum("tki,ti->tk", rel, lf), 0)
  want = np.array([ms.MARKER_SET["head"][l] for l in ms.HEAD_LABELS])  # (front, left) signs
  nh = len(ms.HEAD_LABELS)
  order = max(itertools.permutations(range(nh)),
              key=lambda o: sum(want[s] @ (pf[o[s]], pl[o[s]]) for s in range(nh)))
  return H[:, list(order)], order


def fix_pelvis(P):
  """Rigid refill of pelvis markers that break the cluster or are missing
  while the other 3 are present (see docstring)."""
  idx = ms.PELVIS
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


def convert(name, anthro, task=None, task_state=None):
  names, A, rate = read_export(os.path.join(config.RAW_TAK_DIR, f"{name}.tak") if SOURCE == "tak"
                               else os.path.join(config.EXPORT_DIR, f"{name}.xlsx"))
  T = len(A)
  P = np.full((T, len(ms.LABELS), 3), np.nan)
  missing = []
  rel_note = ""
  if SOURCE == "tak":
    P, H, renamed = body_markers(names, A)
    hk = [ms.LABELS.index(l) for l in ms.HEAD_LABELS]
    P[:, hk], order = assign_head(P, H)
    missing = [l for k, l in enumerate(ms.LABELS) if np.isnan(P[:, k, 0]).all()]
    if renamed:
      rel_note += f"alias labels renamed {renamed}, head by geometry {order}; "
    elif tuple(order) != tuple(range(len(ms.HEAD_LABELS))):
      rel_note += f"HEAD LABELS REORDERED BY GEOMETRY {order}; "
  else:
    for k, l in enumerate(ms.LABELS):
      if l in names:
        P[:, k] = A[:, names.index(l)]
      else:
        missing.append(l)
  P, mislabel = reject_mislabels(P)
  P, pel_fixed = fix_pelvis(P)
  fields, task_note = {}, ""
  if task is not None:  # objects / scene from the columns that are not body markers
    cols = [i for i, n in enumerate(names) if object_column(n)]
    fields, task_note = task.prepare_trial(name, names, A, cols, rate, task_state)
  a = anthro[config.subject_of(name)]
  os.makedirs(OUT, exist_ok=True)
  out = os.path.join(OUT, f"{name}.npz")
  np.savez_compressed(
    out[:-4] + ".part.npz", markers=P.astype(np.float32), marker_labels=np.array(ms.LABELS),
    marker_rate=rate, height_m=a["height_cm"] / 100, mass_kg=a["mass_kg"], **fields,
    pelvis_fixed=pel_fixed, mislabel_rejected=mislabel, source=SOURCE)
  os.replace(out[:-4] + ".part.npz", out)
  gaps = np.isnan(P[..., 0]).mean(0)
  print(f"{name}: {T} frames ({T / rate:.0f} s), body missing {missing or '-'}, {rel_note}"
        f"gaps>1% { {l: round(float(g), 3) for l, g in zip(ms.LABELS, gaps) if g > 0.01} }, "
        f"mislabel-rejected { {ms.LABELS[i]: round(float(mislabel[:, i].mean()), 3) for i in np.argsort(-mislabel.mean(0))[:4] if mislabel[:, i].mean() > 0.005} }, "
        f"pelvis refilled {pel_fixed.any(1).mean():.3f} of frames"
        + (f"; {task_note}" if task_note else ""))


def convert_static(subj, anthro):
  """tak route: static standing trial -> <OUT>/static/<subj>.npz (body markers only)."""
  names, A, rate = read_export(os.path.join(config.RAW_TAK_DIR, "static",
                                            f"{config.static_name(subj)}.tak"))
  P, H, renamed = body_markers(names, A)
  hk = [ms.LABELS.index(l) for l in ms.HEAD_LABELS]
  P[:, hk], order = assign_head(P, H)
  P, mislabel = reject_mislabels(P)
  a = anthro[subj]
  os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
  out = os.path.join(OUT, "static", f"{subj}.npz")
  np.savez_compressed(out[:-4] + ".part.npz", markers=P.astype(np.float32),
                      marker_labels=np.array(ms.LABELS), marker_rate=rate,
                      height_m=a["height_cm"] / 100, mass_kg=a["mass_kg"], source=SOURCE)
  os.replace(out[:-4] + ".part.npz", out)
  seen = np.mean(~np.isnan(P[..., 0]), 0)
  print(f"{subj} static: {len(P)} frames, renamed {len(renamed)} labels, head {order}, "
        f"markers seen < 95%: { {l: round(float(v), 2) for l, v in zip(ms.LABELS, seen) if v < 0.95} }")


def main():
  config.usage(__doc__)
  anthro = config.anthropometrics()
  task = config.task()
  task_state = task.prepare_setup(read_export) if task is not None else None
  if SOURCE == "tak":
    names = sys.argv[1:] or config.trials(config.TAK_CACHE_DIR)
  else:
    names = sys.argv[1:] or sorted(os.path.basename(f)[:-5] for f in
                                   glob.glob(os.path.join(config.EXPORT_DIR, "*.xlsx")))
  for n in names:
    convert(n, anthro, task, task_state)
  if SOURCE == "tak" and not sys.argv[1:]:
    for subj, _ in config.static_files(os.path.join(config.TAK_CACHE_DIR, "static"), ".npz"):
      convert_static(subj, anthro)


if __name__ == "__main__":
  main()
