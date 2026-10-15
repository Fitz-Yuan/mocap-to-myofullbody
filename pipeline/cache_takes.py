""".tak -> npz cache for elev_convert.py with ELEV_SOURCE=tak (no Motive, no earlier export).

Reads the raw Motive takes of the NE trials (lab storage <Subject>/Raw Take Files/, saved as
~/Projects/datasets/MMH_Elevator/raw_tak/S<ss>_<floor>_NE.tak) with optitrack_tak.py
and writes tak_cache/<trial>.npz in the xlsx_cache layout (names [n], data
[T, 2 + 3n] = Frame#, Time, X/Y/Z per marker, Motive world Y-up, metres; NaN =
not seen) plus assets [n] (skeleton / rigid body each marker is labelled to) and
the rigid-body definitions (rb_<name> [k, 3], marker k in the body frame).
Marker names: Plug-in-Gait labels from Motive's live skeleton labelling;
rigid-body markers <body>_<k> (Box_1.., Shelf_.., Elevator_.., Platform_..,
FP1_.., FP2_..; k = Motive's index in the asset, not the hand-given numbers of
the lab's processed exports); unlabelled tracks "Unlabeled NNNN".

Static standing trials (raw_tak/static/S<ss>_Anatomy.tak, lab storage <Subject>/Raw Take
Files/Subject<ss>_Anatomy.tak; S07 has none) go to tak_cache/static/ the same way.

  uv run --no-project --with olefile --with numpy python -I \
    logs/rsl_rl/fullbody_liftbox/scripts/elev_tak_cache.py [trial ...]
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optitrack_tak as tak  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
SRC = os.path.join(ROOT, "raw_tak")
OUT = os.path.join(ROOT, "tak_cache")


def main():
  os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
  names = sys.argv[1:] or (
    sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SRC, "S*_NE.tak")))
    + sorted("static/" + os.path.basename(p)[:-4]
             for p in glob.glob(os.path.join(SRC, "static", "S*_Anatomy.tak"))))
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
