""".tak -> npz cache for prepare_markers.py (no Motive, no export).

Reads the raw Motive takes of the trials (raw_tak/<trial>.tak, trials matching
M2M_TRIAL_GLOB) with mocap2myo.tak and writes tak_cache/<trial>.npz: names [n],
data [T, 2 + 3n] = Frame#, Time, X/Y/Z per marker (Motive world Y-up, metres;
NaN = not seen), rate (Hz), plus assets [n] (skeleton / rigid body
each marker is labelled to) and the rigid-body definitions (rb_<name> [k, 3],
marker k in the body frame). Marker names: the labels from Motive's live skeleton
labelling (Plug-in-Gait in the case study); rigid-body markers <body>_<k> (case
study: Box_1.., Shelf_.., Elevator_.., Platform_.., FP1_.., FP2_..; k = Motive's
index in the asset, not the hand-given numbers of processed exports); unlabelled
tracks "Unlabeled NNNN".

Static standing trials (raw_tak/static/<static name>.tak, M2M_STATIC_NAME; a subject
may have none) go to tak_cache/static/ the same way.

  python pipeline/cache_takes.py [trial ... | static/<static name> ...]   (needs olefile)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mocap2myo import tak  # noqa: E402
from pipeline import config  # noqa: E402

SRC = config.RAW_TAK_DIR
OUT = config.TAK_CACHE_DIR


def main():
  config.usage(__doc__)
  os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
  names = sys.argv[1:] or (
    config.trials(SRC, ".tak")
    + ["static/" + os.path.basename(p)[:-4]
       for _, p in config.static_files(os.path.join(SRC, "static"), ".tak")])
  for n in names:
    tk = tak.read_tak(os.path.join(SRC, f"{n}.tak"))
    T, X = tk["n_frames"], tk["xyz"]
    data = np.concatenate([np.arange(T)[:, None], np.arange(T)[:, None] / tk["rate"],
                           X.reshape(T, -1)], 1)
    rb = {f"rb_{k}": np.array([v[i] for i in sorted(v)]) for k, v in tk["rigid_bodies"].items()}
    out = os.path.join(OUT, f"{n}.npz")
    np.savez_compressed(out[:-4] + ".part.npz", names=np.array(tk["names"]),
                        assets=np.array(tk["assets"]), data=data, rate=tk["rate"], **rb)
    os.replace(out[:-4] + ".part.npz", out)
    lab = [m for m in tk["names"] if not m.startswith("Unlabeled")]
    seen = np.mean(~np.isnan(X[..., 0]), 0)
    body = [(m, s) for m, s, a in zip(tk["names"], seen, tk["assets"])
            if a and not m.startswith("Unlabeled") and "_" not in m]
    print(f"{n}: {T} frames @ {tk['rate']:g} Hz, {len(lab)} labelled ({len(body)} body) + "
          f"{len(tk['names']) - len(lab)} unlabelled; rigid bodies "
          f"{ {k: len(v) for k, v in tk['rigid_bodies'].items()} }; body markers seen <95%: "
          f"{ {m: round(float(s), 3) for m, s in body if s < 0.95} }", flush=True)


if __name__ == "__main__":
  main()
