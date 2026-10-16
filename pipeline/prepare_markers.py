"""Marker conditioning: labelled marker trajectories -> one marker npz per trial.

Input:
  <M2M_CACHE_DIR>/<trial>.npz  the raw files as arrays (read_raw.py; tak_cache/ for
          OptiTrack takes, raw_cache/ for C3D / TRC). Body labels are the
          recording's labels (a take: Motive's live skeleton labels): the marker set's
          own labels or one of its alias sets (M2M_MARKER_SET, M2M_MARKER_ALIASES,
          mocap2myo/markersets/; case study, S01: ISB names onto Plug-in-Gait). Labels the
          marker set does not know are not body markers.
  meta/anthropometrics.csv  height (cm), mass (kg) per subject (config.ANTHRO_*; with
          M2M_ANTHRO=c3d the one read_raw.py collected from the C3D files)
Output: <M2M_NPZ_DIR>/<trial>.npz, Z-up, metres, unscaled, at the recording's rate,
  readable by mocap2myo.ik.load_trial: markers [T, n, 3] (marker set order),
  marker_labels, marker_set (its name), marker_rate, height_m, mass_kg, pelvis_fixed [T, 4] (rigid refill
  flags), mislabel_rejected [T, n], source ("tak", "c3d", "trc"), plus the task's
  fields (M2M_TASK; box_lift: the box pose, shelf, platform and floor,
  tasks/box_lift/box_fit.py), and from a C3D the force plates in the markers' world
  frame (fp_force [T, p, 3] N on the body, fp_cop [T, p, 3] m, fp_free [T, p] N m,
  fp_corners [p, 4, 3] m, see read_raw.py) and the analog channels it kept (analog,
  analog_labels, analog_units, analog_rate). The static standing trials in
  <M2M_CACHE_DIR>/static/ become <M2M_NPZ_DIR>/static/<subject>.npz (body markers
  only, same layout, plates too), read by solve_ik.py with M2M_STATIC=1.

Per trial: labels mapped onto the marker set (alias sets, head markers by
position, if the set has head markers), mislabelled samples rejected, pelvis markers
refilled as a rigid body.

Pelvis: each frame's 4 pelvis markers are fitted to the subject's rigid pelvis
template; a marker off by > PELVIS_TOL whose removal makes the other 3 consistent
is replaced by its rigid reconstruction, and a single missing pelvis marker is
reconstructed from the other 3. (Gap filling in Motive interpolates long ASIS gaps,
when the box is against the belly, and leaves the cluster 30-40 mm off rigid there.)

What the manual Motive processing adds to a raw take is gap filling (case study,
S03: the 5765 samples it has and the raw labels lack are in none of the
unlabelled tracks), which this pipeline does its own way (fix_pelvis, short-gap
interpolation in ik.load_trial, IK with missing markers).

  python pipeline/prepare_markers.py [trial ...]
(with trial names, the static trials are left as they are)
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402  (first: it selects the marker set)
from mocap2myo.kinematics import kabsch  # noqa: E402

ms = config.marker_set()  # the dataset's marker set (M2M_MARKER_SET)

OUT = config.NPZ_DIR
SOURCE = "tak"  # written into each npz ("source"): the input is a raw take (or the cache's)
PELVIS_TOL = 0.012  # m
# cache fields passed on to the marker npz (C3D: force plates, analog channels)
EXTRA = ("fp_force", "fp_cop", "fp_free", "fp_corners", "analog", "analog_labels",
         "analog_units", "analog_rate")


def zup(p):
  """Y-up (x, y, z) -> Z-up (x, -z, y), right-handed."""
  return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def read_take(name):
  """A recording through its cache (read_raw.py) -> (names, [T, n, 3] Z-up, rate),
  unlabelled tracks included. name: a trial, or static/<static name>. A take's cache
  is in Motive's Y-up world; a C3D / TRC cache is Z-up already (zup)."""
  c = np.load(os.path.join(config.CACHE_DIR, name + ".npz"))
  D = c["data"]
  A = D[:, 2:].reshape(len(D), -1, 3)
  if not ("zup" in c.files and bool(c["zup"])):
    A = zup(A)
  return list(c["names"]), A, float(c["rate"])


def cache_extras(name):
  """The cache fields passed on (EXTRA; none for a take) and the input kind."""
  c = np.load(os.path.join(config.CACHE_DIR, name + ".npz"))
  src = str(c["source"]) if "source" in c.files else SOURCE
  return {k: c[k] for k in EXTRA if k in c.files}, src


def subject_size(anthro, subj):
  if subj not in anthro:
    raise SystemExit(f"subject {subj}: no height / mass in {config.ANTHRO_CSV}")
  return anthro[subj]


def object_column(n):
  """A column that is not a body marker: unlabelled tracks and rigid-body markers
  (named <body>_<k> by mocap2myo.tak; body labels have no "_")."""
  return n.startswith("Unlabeled") or "_" in n


def reject_mislabels(P, tol=0.04, n_partners=3):
  """Label swaps (case study, S11 Solid in a processed export: C7 on another marker in
  40% of the frames, metres away). Each marker's partners = the n most rigid markers
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


def choose_alias(names):
  """The alias set (take label -> marker label) the take's labels use, or None (the set's
  own labels): the allowed alias set (M2M_MARKER_ALIASES) that covers most of the labels,
  if it covers more than the set's own labels do."""
  own = sum(n in ms.markers for n in names)
  best, nbest = None, own
  for a in config.marker_aliases(ms).values():
    k = sum(n in a for n in names)
    if k > nbest:
      best, nbest = a, k
  return best


def body_markers(names, A):
  """[T, n, 3] in marker set order from the take's labels (the set's own
  labels, or one of its alias sets, choose_alias). Two labels on one model marker
  (case study: the ISB elbow): per frame the one seen; both seen -> the one in
  ms.PREFER (the lateral epicondyle). Returns P, the head columns [T, h, 3] (in
  ms.HEAD_LABELS order for takes with the set's own labels; reordered by position
  in assign_head) and the renamed labels."""
  T = len(A)
  cand = {l: [] for l in ms.LABELS}
  head, renamed = [], {}
  alias = choose_alias(names)
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
  H = (np.stack([A[:, i] for _, i in sorted(head)], 1)
       if head and len(head) == len(ms.HEAD_LABELS) else None)
  return P, H, renamed


def assign_head(P, H):
  """Head markers by position: front / back and left / right of the head centre in
  the pelvis heading (ASIS - PSIS, left = left - right ASIS), best of all orders
  (layout: the marker set's head role). Returns the [T, h, 3] columns in ms.HEAD_LABELS
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
  want = np.array([ms.head[l] for l in ms.HEAD_LABELS])  # (front, left) signs
  nh = len(ms.HEAD_LABELS)
  order = max(itertools.permutations(range(nh)),
              key=lambda o: sum(want[s] @ (pf[o[s]], pl[o[s]]) for s in range(nh)))
  return H[:, list(order)], order


def place_head(P, H):
  """Head columns of P by position (assign_head), in place; returns the order (() for a
  marker set without head markers). H None (not every head marker found): the columns
  stay as labelled."""
  if not ms.HEAD_LABELS:
    return ()
  hk = [ms.LABELS.index(l) for l in ms.HEAD_LABELS]
  if H is None:
    return tuple(range(len(hk)))
  P[:, hk], order = assign_head(P, H)
  return order


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
  names, A, rate = read_take(name)
  T = len(A)
  rel_note = ""
  P, H, renamed = body_markers(names, A)
  order = place_head(P, H)
  missing = [l for k, l in enumerate(ms.LABELS) if np.isnan(P[:, k, 0]).all()]
  if renamed:
    rel_note += f"alias labels renamed {renamed}, head by geometry {order}; "
  elif tuple(order) != tuple(range(len(ms.HEAD_LABELS))):
    rel_note += f"HEAD LABELS REORDERED BY GEOMETRY {order}; "
  P, mislabel = reject_mislabels(P)
  P, pel_fixed = fix_pelvis(P)
  fields, task_note = {}, ""
  if task is not None:  # objects / scene from the columns that are not body markers
    cols = [i for i, n in enumerate(names) if object_column(n)]
    fields, task_note = task.prepare_trial(name, names, A, cols, rate, task_state)
  a = subject_size(anthro, config.subject_of(name))
  extra, source = cache_extras(name)
  os.makedirs(OUT, exist_ok=True)
  out = os.path.join(OUT, f"{name}.npz")
  np.savez_compressed(
    out[:-4] + ".part.npz", markers=P.astype(np.float32), marker_labels=np.array(ms.LABELS),
    marker_set=ms.key, marker_rate=rate, height_m=a["height_cm"] / 100, mass_kg=a["mass_kg"], **fields,
    pelvis_fixed=pel_fixed, mislabel_rejected=mislabel, source=source, **extra)
  os.replace(out[:-4] + ".part.npz", out)
  gaps = np.isnan(P[..., 0]).mean(0)
  print(f"{name}: {T} frames ({T / rate:.0f} s), body missing {missing or '-'}, {rel_note}"
        f"gaps>1% { {l: round(float(g), 3) for l, g in zip(ms.LABELS, gaps) if g > 0.01} }, "
        f"mislabel-rejected { {ms.LABELS[i]: round(float(mislabel[:, i].mean()), 3) for i in np.argsort(-mislabel.mean(0))[:4] if mislabel[:, i].mean() > 0.005} }, "
        f"pelvis refilled {pel_fixed.any(1).mean():.3f} of frames"
        + (f"; {task_note}" if task_note else "")
        + (f"; {extra['fp_force'].shape[1]} force plates" if "fp_force" in extra else ""))


def convert_static(subj, anthro):
  """Static standing trial -> <OUT>/static/<subj>.npz (body markers only)."""
  name = f"static/{config.static_name(subj)}"
  names, A, rate = read_take(name)
  P, H, renamed = body_markers(names, A)
  order = place_head(P, H)
  P, mislabel = reject_mislabels(P)
  a = subject_size(anthro, subj)
  extra, source = cache_extras(name)
  os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
  out = os.path.join(OUT, "static", f"{subj}.npz")
  np.savez_compressed(out[:-4] + ".part.npz", markers=P.astype(np.float32),
                      marker_labels=np.array(ms.LABELS), marker_set=ms.key, marker_rate=rate,
                      height_m=a["height_cm"] / 100, mass_kg=a["mass_kg"], source=source,
                      **extra)
  os.replace(out[:-4] + ".part.npz", out)
  seen = np.mean(~np.isnan(P[..., 0]), 0)
  print(f"{subj} static: {len(P)} frames, renamed {len(renamed)} labels, head {order}, "
        f"markers seen < 95%: { {l: round(float(v), 2) for l, v in zip(ms.LABELS, seen) if v < 0.95} }")


def main():
  config.usage(__doc__)
  anthro = config.anthropometrics()
  task = config.task()
  task_state = task.prepare_setup(read_take) if task is not None else None
  names = sys.argv[1:] or config.trials(config.CACHE_DIR)
  for n in names:
    convert(n, anthro, task, task_state)
  if not sys.argv[1:]:
    for subj, _ in config.static_files(os.path.join(config.CACHE_DIR, "static"), ".npz"):
      convert_static(subj, anthro)


if __name__ == "__main__":
  main()
