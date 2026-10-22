"""Raw recordings -> npz cache for prepare_markers.py (no Motive, Nexus or QTM, no export).

Reads the dataset's trial files and static trials (M2M_INPUT: OptiTrack .tak, C3D or
TRC; which files, and how a file becomes a trial name and a subject: the dataset
profile M2M_DATASET or the settings in pipeline/config.py) and writes
<M2M_CACHE_DIR>/<trial>.npz (default tak_cache/ for .tak, raw_cache/ for C3D / TRC):
names [n], data [T, 2 + 3n] = Frame#, Time, X/Y/Z per marker (metres, NaN = not seen),
rate (Hz), and

  .tak (mocap2myo.tak): Motive's world (Y-up; prepare_markers.py turns it Z-up), plus
    assets [n] (skeleton / rigid body each marker is labelled to) and the rigid-body
    definitions (rb_<name> [k, 3], marker k in the body frame). Marker names: the
    labels from Motive's live skeleton labelling (Plug-in-Gait in the case study);
    rigid-body markers <body>_<k> (case study: Box_1.., Shelf_.., Elevator_..,
    Platform_.., FP1_.., FP2_..; k = Motive's index in the asset, not the hand-given
    numbers of processed exports); unlabelled tracks "Unlabeled NNNN".
  C3D / TRC (mocap2myo.readers): already Z-up (zup = True; M2M_UP is the file's up axis,
    "auto" = from the head and foot markers) and in metres (M2M_UNITS overrides the
    file's unit); "Subject:" label prefixes dropped; plus source, source_file (the path
    under M2M_RAW_DIR), up, units, and
    from a C3D when it has them:
      fp_force [T, p, 3] N (the force each plate exerts on the body), fp_cop [T, p, 3] m
        (NaN where the vertical force < M2M_FP_MIN_FZ), fp_free [T, p] N m (free moment
        about the vertical through the centre of pressure), fp_corners [p, 4, 3] m,
        fp_sign: the force plates (ezc3d) per marker frame (the analog samples of the
        frame averaged), rotated and scaled with the markers
      analog [S, k], analog_labels, analog_units, analog_rate: the other analog channels
        (EMG, ...; M2M_ANALOG) at their own rate
      other_labels, other_groups, other_units, other_data [T, k, 3]: points that are not
        markers, as in the file (its frame and units): the lab software's model outputs
        (POINT type groups ANGLES, FORCES, MOMENTS, POWERS, MODELED_MARKERS, ...) and the
        labels matching M2M_POINTS_EXCLUDE
      height_m, mass_kg (M2M_ANTHRO=c3d: from the C3D parameters, also collected per
        subject in <M2M_CACHE_DIR>/anthropometrics.csv: height cm, mass kg)
M2M_RATE (a number): keep every k-th frame of a file recorded at k times that rate.

Static standing trials (M2M_RAW_STATIC / M2M_STATIC_REGEX; a subject may have none) go
to <M2M_CACHE_DIR>/static/<static name>.npz (M2M_STATIC_NAME) the same way.

  python pipeline/read_raw.py [trial ... | static/<static name> ...]
(.tak needs olefile, C3D ezc3d: pip install -e ".[tak,c3d]")
"""

import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mocap2myo import readers, tak  # noqa: E402
from pipeline import config  # noqa: E402

OUT = config.CACHE_DIR


def decimation(rate):
  """Keep every k-th frame (M2M_RATE); 1 = the file's own rate."""
  if config.RATE is None or config.RATE == rate:
    return 1
  k = rate / config.RATE
  if abs(k - round(k)) > 1e-6 or round(k) < 2:
    raise SystemExit(f"M2M_RATE={config.RATE:g}: the file is at {rate:g} Hz, not an integer "
                     "multiple of it")
  return int(round(k))


def save(out, **arrays):
  np.savez_compressed(out[:-4] + ".part.npz", **arrays)
  os.replace(out[:-4] + ".part.npz", out)


def cache_tak(n, path):
  tk = tak.read_tak(path)
  T, X, rate = tk["n_frames"], tk["xyz"], tk["rate"]
  step = decimation(rate)
  if step > 1:
    X, rate = X[::step], rate / step
    T = len(X)
  data = np.concatenate([np.arange(T)[:, None], np.arange(T)[:, None] / rate,
                         X.reshape(T, -1)], 1)
  rb = {f"rb_{k}": np.array([v[i] for i in sorted(v)]) for k, v in tk["rigid_bodies"].items()}
  save(os.path.join(OUT, f"{n}.npz"), names=np.array(tk["names"]),
       assets=np.array(tk["assets"]), data=data, rate=rate, **rb)
  lab = [m for m in tk["names"] if not m.startswith("Unlabeled")]
  seen = np.mean(~np.isnan(X[..., 0]), 0)
  body = [(m, s) for m, s, a in zip(tk["names"], seen, tk["assets"])
          if a and not m.startswith("Unlabeled") and "_" not in m]
  print(f"{n}: {T} frames @ {rate:g} Hz, {len(lab)} labelled ({len(body)} body) + "
        f"{len(tk['names']) - len(lab)} unlabelled; rigid bodies "
        f"{ {k: len(v) for k, v in tk['rigid_bodies'].items()} }; body markers seen <95%: "
        f"{ {m: round(float(s), 3) for m, s in body if s < 0.95} }", flush=True)
  return dict(up="y")


def cache_file(n, path):
  """A C3D or TRC file -> the cache (see the docstring). Returns the up axis and the
  anthropometric values it holds."""
  r = readers.read_trial(path, config.INPUT, up=config.UP, units=config.UNITS or None,
                         exclude=config.POINTS_EXCLUDE, analog=config.ANALOG,
                         fp_sign=config.FP_SIGN, fp_min_fz=config.FP_MIN_FZ,
                         anthro=((config.ANTHRO_C3D_HEIGHT, config.ANTHRO_C3D_MASS)
                                 if config.ANTHRO == "c3d" else None))
  k = decimation(r["rate"])
  X, rate = r["xyz"][::k], r["rate"] / k
  T = len(X)
  data = np.concatenate([np.arange(T)[:, None], np.arange(T)[:, None] / rate,
                         X.reshape(T, -1)], 1)
  extra = dict(zup=True, source=r["kind"],
               source_file=os.path.relpath(path, config.RAW_DIR), up=r["up"], units=r["units"])
  if r["plates"] is not None and r["plates"].get("corners_estimated"):
    extra["fp_corners_estimated"] = True
  note = []
  pl = r["plates"]
  if pl is not None:
    extra.update(fp_force=pl["force"][::k].astype(np.float32),
                 fp_cop=pl["cop"][::k].astype(np.float32),
                 fp_free=pl["free"][::k].astype(np.float32), fp_corners=pl["corners"],
                 fp_sign=pl["sign"])
    peak = np.nanmax(np.nan_to_num(pl["force"][..., 2], nan=0.0), 0)
    note.append(f"{len(peak)} plates (peak vertical {np.round(peak).astype(int).tolist()} N, "
                f"sign {pl['sign']:+.0f}, centre of pressure vs ezc3d's "
                f"{pl['cop_check_mm']:.2f} mm)")
    if pl["cop_check_mm"] > 5:
      raise SystemExit(f"{path}: the centre of pressure differs from ezc3d's by "
                       f"{pl['cop_check_mm']:.1f} mm; the plate moments are not about the "
                       "plate's surface centre as mocap2myo.readers assumes")
  an = r["analog"]
  if an is not None:
    extra.update(analog=an["data"], analog_labels=np.array(an["labels"]),
                 analog_units=np.array(an["units"]), analog_rate=an["rate"])
    note.append(f"analog {len(an['labels'])} @ {an['rate']:g} Hz")
  ot = r["other"]
  if ot is not None:
    extra.update(other_labels=np.array(ot["labels"]), other_groups=np.array(ot["groups"]),
                 other_units=np.array(ot["units"]), other_data=ot["data"][::k])
    note.append(f"other points {dict(collections.Counter(ot['groups']))}")
  an_vals = r.get("anthro") or {}
  for key in ("height_m", "mass_kg", "sex"):
    if an_vals.get(key) is not None:
      extra[key] = an_vals[key]
  save(os.path.join(OUT, f"{n}.npz"), names=np.array(r["labels"]), data=data, rate=rate,
       **extra)
  seen = np.mean(~np.isnan(X[..., 0]), 0)
  unl = [m.startswith(("*", "Unlabeled")) or not m for m in r["labels"]]  # Vicon: *<k>
  low = {m: round(float(s), 3) for m, s, u in zip(r["labels"], seen, unl) if s < 0.95 and not u}
  print(f"{n}: {T} frames @ {rate:g} Hz{f' (1 in {k} frames kept)' if k > 1 else ''}, "
        f"{len(unl) - sum(unl)} labelled + {sum(unl)} unlabelled markers ({r['units']}, "
        f"up {r['up']}); " + "".join(f"{x}; " for x in note) + f"markers seen <95%: {low}"
        + (f", height {an_vals['height_m']:.3f} m" if an_vals.get("height_m") else "")
        + (f", mass {an_vals['mass_kg']:.1f} kg" if an_vals.get("mass_kg") else ""), flush=True)
  return dict(up=r["up"], **an_vals)


def write_anthropometrics(found):
  """Heights / masses from the C3D files, merged into ANTHRO_CSV (height cm, mass kg)."""
  sc, hc, mc = config.ANTHRO_COLUMNS
  rows = {}
  if os.path.exists(config.ANTHRO_CSV):
    with open(config.ANTHRO_CSV, newline="") as f:
      rows = {r[sc]: r for r in csv.DictReader(f)}
  for s, vals in sorted(found.items()):
    h = [v["height_m"] for v in vals if v.get("height_m")]
    m = [v["mass_kg"] for v in vals if v.get("mass_kg")]
    if not h or not m:
      print(f"  {s}: no height / mass in its C3D files ({config.ANTHRO_C3D_HEIGHT}, "
            f"{config.ANTHRO_C3D_MASS})")
      continue
    if np.ptp(h) > 0.01 or np.ptp(m) > 1.0:
      print(f"  {s}: heights {sorted(set(h))} m / masses {sorted(set(m))} kg differ between "
            "its files; the median is used")
    rows[s] = {sc: s, hc: f"{100 * np.median(h):.1f}", mc: f"{np.median(m):.2f}"}
    sx = {v.get("sex") for v in vals} - {None}
    if len(sx) == 1 and config.SEX_COLUMN:  # .b3d files carry the subject's sex
      rows[s][config.SEX_COLUMN] = sx.pop()
  os.makedirs(os.path.dirname(os.path.abspath(config.ANTHRO_CSV)), exist_ok=True)
  cols = [sc, hc, mc] + ([config.SEX_COLUMN] if config.SEX_COLUMN else [])
  with open(config.ANTHRO_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for s in sorted(rows):
      w.writerow({c: rows[s].get(c, "") for c in cols})
  print(f"anthropometrics ({len(rows)} subjects) -> {config.ANTHRO_CSV}")


def main():
  config.usage(__doc__)
  trials, skipped = config.raw_trials()
  statics, skipped_s = config.raw_statics()
  static = {f"static/{config.static_name(s)}": (s, p) for s, p in statics.items()}
  files = {n: (config.subject_of(n), p) for n, p in trials.items()} | static
  names = sys.argv[1:]
  unknown = [n for n in names if n not in files]
  if unknown:
    raise SystemExit(f"{unknown}: no such trial / static trial in {config.RAW_DIR} "
                     f"(trials: {list(trials)[:5]}..., static: {list(static)[:3]}...)")
  names = names or list(files)
  if not names:
    raise SystemExit(f"no trial files: {config.RAW_DIR}/{config.RAW_TRIALS}")
  if (skipped or skipped_s) and not sys.argv[1:]:
    print(f"skipped (no match of M2M_TRIAL_REGEX / M2M_STATIC_REGEX): "
          f"{len(skipped)} trial files {skipped[:3]}, {len(skipped_s)} static files "
          f"{skipped_s[:3]}")
  os.makedirs(os.path.join(OUT, "static"), exist_ok=True)
  ups, found = collections.Counter(), collections.defaultdict(list)
  for n in names:
    subj, path = files[n]
    try:
      r = cache_tak(n, path) if config.INPUT == "tak" else cache_file(n, path)
    except (OSError, ValueError) as e:  # a file the readers cannot take
      raise SystemExit(f"{path}: {type(e).__name__}: {e}")
    ups[r["up"]] += 1
    if "height_m" in r or "mass_kg" in r:
      found[subj].append(r)
  if config.INPUT != "tak" and len(ups) > 1:
    print(f"WARNING: the files have different up axes {dict(ups)} (M2M_UP={config.UP})")
  if config.ANTHRO == "c3d":
    write_anthropometrics(found)


if __name__ == "__main__":
  main()
