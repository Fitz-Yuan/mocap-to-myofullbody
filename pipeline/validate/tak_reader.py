"""Check the take reader (mocap2myo/tak.py) against Motive's own TRC exports of the same takes.

For every (take, TRC) pair: markers matched by name, and over the samples present
in both, the share that are identical (< 0.01 mm) and the largest difference.
A processed take must match its TRC exactly. A raw take differs where the lab
edited it in Motive before exporting (gap filling adds samples; relabelling
moves a few), so for raw takes only the identical share is informative.

  python pipeline/validate/tak_reader.py DIR        (needs olefile)
DIR holds S<ss>_<take>.tak (raw), proc_S<ss>_<take>.tak (processed) and
S<ss>_<take>.trc (Motive export).
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mocap2myo import tak  # noqa: E402
from pipeline import config  # noqa: E402


def _identical_share(X, y):
  both = ~np.isnan(X[:, 0]) & ~np.isnan(y[:, 0])
  if not both.any():
    return 0.0
  return float(np.mean(np.abs(X[both] - y[both]).max(1) < 1e-5))


def check(take_path, trc_path):
  """Markers are paired by name; a TRC column whose name is not in the take, or
  whose same-name track is not identical, is paired with the take track that
  is identical to it (rigid-body markers: the reader names them <body>_<k> by
  Motive's index, the export keeps "Marker 00k" or a hand-given number)."""
  tk = tak.read_tak(take_path)
  names, frames, Y = tak.read_trc(trc_path)
  XT = tk["xyz"][frames]
  shared = same = 0
  worst, unmatched, by_pos, only_trc, only_take = 0.0, [], 0, 0, 0
  for j, n in enumerate(names):
    i = tk["names"].index(n) if n in tk["names"] else None
    if i is None or _identical_share(XT[:, i], Y[:, j]) < 0.99:
      cand = [(_identical_share(XT[:, k], Y[:, j]), k) for k in range(XT.shape[1])]
      s, k = max(cand)
      if s >= 0.99:
        i, by_pos = k, by_pos + 1
    if i is None:
      unmatched.append(n)
      continue
    X = XT[:, i]
    a, b = ~np.isnan(X[:, 0]), ~np.isnan(Y[:, j, 0])
    both = a & b
    d = np.abs(X[both] - Y[both, j]).max(1) if both.any() else np.zeros(0)
    shared += both.sum()
    same += (d < 1e-5).sum()
    worst = max(worst, float(d.max()) if len(d) else 0.0)
    only_trc += (b & ~a).sum()
    only_take += (a & ~b).sum()
  return dict(markers=len(names), unmatched=unmatched, by_position=by_pos, shared=int(shared),
              identical=same / max(shared, 1), max_diff_mm=1e3 * worst,
              trc_only=int(only_trc), take_only=int(only_take))


def main():
  config.usage(__doc__)
  d = sys.argv[1]
  rows = []
  for trc in sorted(glob.glob(os.path.join(d, "S*.trc"))):
    stem = os.path.basename(trc)[:-4]
    for kind, take in (("processed", os.path.join(d, f"proc_{stem}.tak")),
                       ("raw", os.path.join(d, f"{stem}.tak"))):
      if os.path.exists(take):
        r = check(take, trc)
        rows.append((stem, kind, r))
        print(f"{stem:16s} {kind:9s} markers {r['markers']:3d} (by position {r['by_position']:2d}) "
              f"unmatched {len(r['unmatched']):2d} "
              f"shared {r['shared']:8d} identical {r['identical']:.4f} max diff "
              f"{r['max_diff_mm']:9.3f} mm  only in TRC {r['trc_only']:6d}  only in take "
              f"{r['take_only']:6d}" + (f"  unmatched {r['unmatched']}" if r["unmatched"] else ""),
              flush=True)
  proc = [r for _, k, r in rows if k == "processed"]
  if proc:
    exact = sum(r["identical"] == 1.0 and not r["unmatched"] and not r["trc_only"]
                and not r["take_only"] for r in proc)
    print(f"processed takes identical to their TRC export: {exact}/{len(proc)}")


if __name__ == "__main__":
  main()
