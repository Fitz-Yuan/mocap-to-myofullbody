"""Read one trial file of any supported kind into one structure: C3D, TRC, OptiTrack .tak,
AddBiomechanics .b3d (one trial of it: "<file>.b3d#<trial>").

read_trial(path) -> dict
  kind        "c3d" | "trc" | "tak"
  labels      [n] marker labels ("Subject:" prefixes stripped)
  xyz         [T, n, 3] metres, NaN = missing (C3D: residual < 0; any kind: a sample
              that is exactly (0, 0, 0) or NaN)
  rate        Hz (C3D: POINT:RATE; TRC: DataRate; .tak: Motive's camera rate)
  up, R       the file's up axis ("z", "y", "-x", ...) and the rotation R [3, 3] applied
              to every position and vector (x_world = R @ x_file), so that the world is
              Z-up; up=None (the default) keeps the file's frame (R = identity)
  units       the file's length unit (C3D: POINT:UNITS; TRC: Units; .tak: m)
  other       C3D points that are not markers, as in the file (its frame, its units):
              the lab software's model outputs listed in the POINT type groups (ANGLES,
              FORCES, MOMENTS, POWERS, SCALARS, REACTIONS, MODELED_MARKERS) and the
              labels matching `exclude` (e.g. Plug-in-Gait's segment points PELO, ...):
              dict(labels, groups, units, data [T, k, 3]); None for TRC / .tak
  analog      C3D analog channels that are not force-plate channels and whose label
              matches `analog` (EMG, ...), at their own rate: dict(labels, units, rate,
              data [S, k]); None if there are none
  plates      C3D force plates (ezc3d, extract_forceplat_data=True) per marker frame (the
              analog samples of each frame averaged), in the world frame of the markers, SI:
                force   [T, p, 3] N, the force the plate exerts on the body (ground
                        reaction: +z while standing on a level plate)
                cop     [T, p, 3] m, centre of pressure on the plate surface (NaN where
                        |vertical force| < fp_min_fz)
                free    [T, p] N m, free moment about the vertical through the centre of
                        pressure (NaN where the centre of pressure is)
                corners [p, 4, 3] m
                sign    the factor applied to ezc3d's force and moment (+1 / -1; fp_sign)
                cop_check_mm  largest difference to ezc3d's own centre of pressure over
                        the loaded frames (a check of the moment's reference point)
              None if the file has none
  anthro      C3D only, with `anthro=(height spec, mass spec)`: dict(height_m, mass_kg)
              from the C3D parameters (values that are missing are None)
  assets, rigid_bodies   .tak only (mocap2myo.tak.read_tak)

Up axis: up="z" | "y" | "x" (or "-z", ...) names the axis that points up in the file;
"auto" finds it from the marker labels: the axis along which head / upper-trunk markers
(…FHD, C7, CV7, CLAV, SJN, SHO, SAE, ...) sit farthest above foot markers (…HEE, TOE,
FCC, FM1, ANK, ...), median over the frames.

C3D needs ezc3d (pip install -e ".[c3d]"), .tak olefile (".[tak]"), .b3d nimblephysics.

  python -m mocap2myo.readers FILE [--up auto|x|y|z]     (a summary of one file)
"""

import os
import re
import sys

import numpy as np

LENGTH = {"m": 1.0, "dm": 0.1, "cm": 0.01, "mm": 0.001}
MOMENT = {"nm": 1.0, "n.m": 1.0, "n·m": 1.0, "nmm": 0.001, "n.mm": 0.001, "n·mm": 0.001}
# POINT type groups of a C3D (model outputs, not markers) and their unit parameters
TYPE_GROUPS = {"ANGLES": "ANGLE_UNITS", "FORCES": "FORCE_UNITS", "MOMENTS": "MOMENT_UNITS",
               "POWERS": "POWER_UNITS", "SCALARS": "SCALAR_UNITS",
               "REACTIONS": "REACTION_UNITS", "MODELED_MARKERS": "MODELED_MARKER_UNITS"}
_UPPER = re.compile(r"(FHD|BHD|HEAD|C7|CV7|CLAV|SJN|STRN|SXS|SHO|SAE|CAJ|ACR)$")
_FOOT = re.compile(r"(HEE|HEEL|TOE|FCC|FM[1-5]|MT[1-5]|MTH[1-5]?|ANK|FAL|TAM)$")
_AXES = {"x": 0, "y": 1, "z": 2}


def kind_of(path):
  ext = os.path.splitext(path.partition("#")[0])[1].lower().lstrip(".")
  if ext not in ("c3d", "trc", "tak", "b3d"):
    raise ValueError(f"{path}: unknown file kind {ext!r} (c3d, trc, tak, b3d)")
  return ext


def strip_prefix(label):
  """"Subject:LASI" -> "LASI" (Vicon / Qualisys subject prefixes)."""
  return label.rsplit(":", 1)[-1].strip()


def up_rotation(up):
  """Rotation R [3, 3] (x_world = R @ x_file) that turns the file's `up` axis ("z", "-y",
  ...) into +z. "y" gives (x, y, z) -> (x, -z, y), as for a Motive take."""
  m = re.fullmatch(r"([+-]?)([xyz])", up)
  if m is None:
    raise ValueError(f"up axis {up!r}: expected x, y, z (optionally with a sign) or auto")
  neg, ax = m.group(1) == "-", m.group(2)
  R = {"z": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
       "y": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
       "x": [[0, 0, -1], [0, 1, 0], [1, 0, 0]]}[ax]
  R = np.array(R, float)
  if neg:  # half a turn about the world's x axis (z -> -z)
    R = np.diag([1.0, -1.0, -1.0]) @ R
  return R


def detect_up(labels, xyz):
  """The file's up axis from head / upper-trunk vs foot marker labels (see module doc)."""
  up = [i for i, l in enumerate(labels) if _UPPER.search(l.upper())]
  foot = [i for i, l in enumerate(labels) if _FOOT.search(l.upper())]
  if not up or not foot:
    raise ValueError("up axis: no head / upper-trunk or no foot marker labels to tell it "
                     "from; set it (z, y, x, or with a sign)")
  with np.errstate(all="ignore"):
    d = np.nanmedian(np.nanmean(xyz[:, up], 1) - np.nanmean(xyz[:, foot], 1), 0)
  if np.isnan(d).any():
    raise ValueError("up axis: head and foot markers never seen together; set it")
  k = int(np.argmax(np.abs(d)))
  rest = np.delete(np.abs(d), k)
  if not np.abs(d[k]) > 2 * rest.max():
    raise ValueError(f"up axis: head - foot = {np.round(d, 3)} (file units) has no clear "
                     "largest axis; set it")
  return ("" if d[k] > 0 else "-") + "xyz"[k]


def _labels(group, key, n=None):
  """C3D labels of a group: LABELS, LABELS2, ... (more than 255 entries)."""
  out, k = [], 1
  while (key if k == 1 else f"{key}{k}") in group:
    out += list(group[key if k == 1 else f"{key}{k}"]["value"])
    k += 1
  if n is not None:
    out = (out + [f"_{i}" for i in range(len(out), n)])[:n]
  return [str(x) for x in out]


def _value(params, spec):
  """A C3D parameter value "GROUP:NAME": the entry NAME of the group's LABELS (its VALUES
  and UNITS), or the parameter NAME itself. Returns (value, unit) or (None, "")."""
  g, _, name = spec.partition(":")
  grp = params.get(g)
  if grp is None:
    return None, ""
  if "LABELS" in grp and "VALUES" in grp:
    labs = [str(x).strip().lower() for x in grp["LABELS"]["value"]]
    if name.lower() in labs:
      i = labs.index(name.lower())
      vals = np.ravel(grp["VALUES"]["value"])
      units = list(grp["UNITS"]["value"]) if "UNITS" in grp else []
      return float(vals[i]), (str(units[i]).strip() if i < len(units) else "")
  for k, v in grp.items():
    if k.lower() == name.lower() and isinstance(v, dict) and "value" in v:
      val = np.ravel(v["value"])
      return (float(val[0]) if len(val) else None), ""
  return None, ""


def _height_m(v, unit):
  if v is None or not v > 0:
    return None
  u = unit.lower()
  if u in LENGTH:
    return v * LENGTH[u]
  return v if v < 3 else v / 100 if v < 300 else v / 1000  # no unit: by magnitude


def _mass_kg(v, unit):
  if v is None or not v > 0:
    return None
  u = unit.lower()
  return v / 1000 if u == "g" else v * 0.45359237 if u in ("lb", "lbs") else v


def _missing(xyz):
  bad = np.isnan(xyz).any(-1) | (xyz == 0).all(-1)
  xyz = xyz.copy()
  xyz[bad] = np.nan
  return xyz


def _plates(c, T, ratio, R, sign, min_fz):
  """Force plates of an ezc3d c3d (extract_forceplat_data=True) per marker frame."""
  pfs = c["data"]["platform"]
  if not pfs:
    return None
  F, M, CE, corners = [], [], [], []
  for pf in pfs:
    if str(pf["unit_force"]).strip().lower() != "n":
      raise ValueError(f"force plate unit {pf['unit_force']!r}: only N is handled")
    mu = MOMENT.get(str(pf["unit_moment"]).strip().lower())
    pu = LENGTH.get(str(pf["unit_position"]).strip().lower())
    if mu is None or pu is None:
      raise ValueError(f"force plate units {pf['unit_moment']!r} / {pf['unit_position']!r}")

    def frames(a):  # [3, S] at the analog rate -> [T, 3], the samples of each frame averaged
      a = np.asarray(a, float).T
      n = min(len(a) // ratio, T)
      out = np.full((T, 3), np.nan)
      out[:n] = a[:n * ratio].reshape(n, ratio, 3).mean(1)
      return out

    F.append(frames(pf["force"]))
    M.append(frames(pf["moment"]) * mu)  # about the plate's surface centre (mean corners)
    CE.append(frames(pf["center_of_pressure"]) * pu)
    corners.append(np.asarray(pf["corners"], float).T * pu)
  F, M, CE = (np.stack(x, 1) @ R.T for x in (F, M, CE))  # [T, p, 3], world
  corners = np.stack(corners) @ R.T  # [p, 4, 3]
  centre = corners.mean(1)
  # each plate's surface normal (pointing up) and in-plane axes
  n_ = np.zeros((len(pfs), 3))
  u_ = np.zeros((len(pfs), 3))
  for k in range(len(pfs)):
    q = corners[k] - centre[k]
    nk = np.linalg.svd(q)[2][2]
    n_[k] = nk if nk[2] > 0 else -nk
    u_[k] = q[0] - (q[0] @ n_[k]) * n_[k]
    u_[k] /= np.linalg.norm(u_[k])
  v_ = np.cross(n_, u_)
  Fn = np.einsum("tpi,pi->tp", F, n_)
  if sign == "auto":  # the ground pushes: on a loaded plate, the reaction points out of it
    load = np.abs(Fn) > min_fz
    s = 1.0 if not load.any() or np.nansum(Fn[load]) > 0 else -1.0
  else:
    s = float(sign)
  F, M, Fn = s * F, s * M, s * Fn
  with np.errstate(all="ignore"):
    a = -np.einsum("tpi,pi->tp", M, v_) / Fn
    b = np.einsum("tpi,pi->tp", M, u_) / Fn
  cop = centre[None] + a[..., None] * u_[None] + b[..., None] * v_[None]
  m_cop = M - np.cross(cop - centre[None], F)  # moment about the centre of pressure
  free = m_cop[..., 2]  # about the vertical through it
  off = ~(np.abs(F[..., 2]) >= min_fz)
  cop[off] = np.nan
  free[off] = np.nan
  big = np.abs(F[..., 2]) > 10 * min_fz
  chk = float(np.nanmax(np.linalg.norm(cop[big] - CE[big], axis=-1)) * 1e3) if big.any() else 0.0
  return dict(force=F, cop=cop, free=free, corners=corners, sign=s, cop_check_mm=chk)


def read_c3d(path, up=None, units=None, exclude="", analog=".*", fp_sign="auto",
             fp_min_fz=20.0, anthro=None):
  import ezc3d  # optional dependency (pip install -e ".[c3d]"): only C3D input needs it
  c = ezc3d.c3d(path, extract_forceplat_data=True)
  P = c["parameters"]
  pt = P["POINT"]
  pts = np.asarray(c["data"]["points"], float)  # [4, n, T]
  n, T = pts.shape[1], pts.shape[2]
  labels = [strip_prefix(x) for x in _labels(pt, "LABELS", n)]
  rate = float(np.ravel(pt["RATE"]["value"])[0])
  unit = (units or (str(pt["UNITS"]["value"][0]).strip() if "UNITS" in pt else "")).lower()
  if unit not in LENGTH:
    raise ValueError(f"{path}: point unit {unit!r}; set it (m, cm, mm)")
  X = pts[:3].transpose(2, 1, 0)  # T, n, 3
  res = np.asarray(c["data"]["meta_points"]["residuals"], float)[0].T  # T, n
  X[res < 0] = np.nan
  X = _missing(X)
  group = {}
  for g in TYPE_GROUPS:
    if g in pt:
      for x in _labels(pt, g):
        group.setdefault(strip_prefix(x), g)
  excl = re.compile(exclude) if exclude else None
  is_other = [l in group or bool(excl and excl.fullmatch(l)) for l in labels]
  mk = [i for i, o in enumerate(is_other) if not o]
  seen = set()
  for i in list(mk):  # the same label twice (several subjects): the first one
    if labels[i] in seen:
      mk.remove(i)
      is_other[i] = True
    seen.add(labels[i])
  ot = [i for i, o in enumerate(is_other) if o]

  def unit_of(g):
    k = TYPE_GROUPS.get(g)
    if k and k in pt:
      return str(pt[k]["value"][0]).strip()
    return unit

  other = None
  if ot:
    og = [group.get(labels[i], "excluded") for i in ot]
    other = dict(labels=[labels[i] for i in ot], groups=og, units=[unit_of(g) for g in og],
                 data=X[:, ot].astype(np.float32))
  labels = [labels[i] for i in mk]
  X = X[:, mk] * LENGTH[unit]
  if up == "auto":
    up = detect_up(labels, X)
  R = up_rotation(up) if up else np.eye(3)
  X = X @ R.T
  # analog channels (not the force plates')
  an = P.get("ANALOG", {})
  A = np.asarray(c["data"]["analogs"], float)[0]  # [k, S]
  arate = float(np.ravel(an["RATE"]["value"])[0]) if "RATE" in an else rate
  ratio = int(round(arate / rate)) if arate else 1
  if A.shape[0] and abs(arate / rate - ratio) > 1e-6:
    raise ValueError(f"{path}: analog rate {arate:g} Hz is not a multiple of the point rate "
                     f"{rate:g} Hz")
  fpg = P.get("FORCE_PLATFORM", {})
  fp_ch = set()
  if "CHANNEL" in fpg:
    fp_ch = {int(x) - 1 for x in np.ravel(fpg["CHANNEL"]["value"]) if int(x) > 0}
  alab = _labels(an, "LABELS", A.shape[0]) if A.shape[0] else []
  aunit = _labels(an, "UNITS", A.shape[0]) if A.shape[0] else []
  keep = [i for i, l in enumerate(alab)
          if i not in fp_ch and analog and re.fullmatch(analog, strip_prefix(l))]
  ana = None
  if keep:
    ana = dict(labels=[strip_prefix(alab[i]) for i in keep], units=[aunit[i] for i in keep],
               rate=arate, data=A[keep].T.astype(np.float32))
  plates = _plates(c, T, max(ratio, 1), R, fp_sign, fp_min_fz) if fp_ch else None
  out = dict(kind="c3d", labels=labels, xyz=X, rate=rate, up=up, R=R, units=unit,
             other=other, analog=ana, plates=plates)
  if anthro is not None:
    h, hu = _value(P, anthro[0])
    m, mu = _value(P, anthro[1])
    out["anthro"] = dict(height_m=_height_m(h, hu), mass_kg=_mass_kg(m, mu))
  return out


def read_trc_file(path, up=None, units=None):
  from mocap2myo.tak import read_trc
  with open(path) as f:
    head = [next(f).rstrip("\n").split("\t") for _ in range(3)]
  h = {k.strip(): v.strip() for k, v in zip(head[1], head[2])}
  rate = float(h.get("DataRate") or h.get("CameraRate"))
  unit = (units or h.get("Units", "")).lower()
  if unit not in LENGTH:
    raise ValueError(f"{path}: unit {unit!r}; set it (m, cm, mm)")
  names, _, X = read_trc(path)
  labels = [strip_prefix(x) for x in names]
  X = _missing(X) * LENGTH[unit]
  if up == "auto":
    up = detect_up(labels, X)
  R = up_rotation(up) if up else np.eye(3)
  return dict(kind="trc", labels=labels, xyz=X @ R.T, rate=rate, up=up, R=R, units=unit,
              other=None, analog=None, plates=None)


def read_tak_file(path, up=None):
  from mocap2myo.tak import read_tak
  tk = read_tak(path)
  R = up_rotation(up) if up else np.eye(3)
  X = tk["xyz"] @ R.T if up else tk["xyz"]
  return dict(kind="tak", labels=list(tk["names"]), xyz=X, rate=tk["rate"], up=up, R=R,
              units="m", other=None, analog=None, plates=None, assets=tk["assets"],
              rigid_bodies=tk["rigid_bodies"])


def b3d_trials(path):
  """Trial names of an AddBiomechanics .b3d file (nimblephysics)."""
  import nimblephysics as nimble  # optional dependency (pip install nimblephysics): .b3d only
  sub = nimble.biomechanics.SubjectOnDisk(path)
  return [sub.getTrialName(i) for i in range(sub.getNumTrials())]


def read_b3d(path, up=None, fp_min_fz=20.0):
  """One trial of an AddBiomechanics .b3d file, path "<file>.b3d#<trial name>": the markers as
  observed (names stripped: some carry a "\r"), the force plates (OpenSim world, Y up; the
  torque is the free moment about the vertical through the centre of pressure; the file has
  no plate corners, so a 0.6 x 0.4 m rectangle on the plate's surface plane, centred on the
  median centre of pressure, stands in for them: corners_estimated; a plate whose centre of
  pressure lies at several heights, more than 5 cm apart, is split into one plate per
  height) and the subject's height, mass and sex."""
  import nimblephysics as nimble  # optional dependency (pip install nimblephysics): .b3d only
  file, _, trial = path.partition("#")
  sub = nimble.biomechanics.SubjectOnDisk(file)
  names = [sub.getTrialName(i) for i in range(sub.getNumTrials())]
  if trial not in names:
    raise ValueError(f"{file}: no trial {trial!r} (trials: {names[:5]}...)")
  t = names.index(trial)
  T = sub.getTrialLength(t)
  fr = sub.readFrames(t, 0, T, includeSensorData=True, includeProcessingPasses=False)
  labels = sorted({n.strip() for f in fr for n, _ in f.markerObservations})
  idx = {n: i for i, n in enumerate(labels)}
  X = np.full((T, len(labels), 3), np.nan)
  for k, f in enumerate(fr):
    for n, x in f.markerObservations:
      X[k, idx[n.strip()]] = np.ravel(x)
  X = _missing(X)
  if up == "auto":
    up = detect_up(labels, X)
  R = up_rotation(up) if up else np.eye(3)
  P = sub.getNumForcePlates(t)
  plates = None
  if P:
    F = np.array([np.ravel(f.rawForcePlateForces) for f in fr]).reshape(T, P, 3) @ R.T
    C = np.array([np.ravel(f.rawForcePlateCenterOfPressures) for f in fr]).reshape(T, P, 3) @ R.T
    M = np.array([np.ravel(f.rawForcePlateTorques) for f in fr]).reshape(T, P, 3) @ R.T
    sign = 1.0
    loaded = F[..., 2] >= fp_min_fz
    C = np.where(loaded[..., None], C, np.nan)
    free = np.where(loaded, M[..., 2], np.nan)
    # a plate whose centre of pressure sits at more than one height (one channel for the
    # steps of a staircase) becomes one plate per height: each surface its own plane
    Fs, Cs, Ms = [], [], []
    for p in range(P):
      z = C[loaded[:, p], p, 2]
      levels = []
      for v in np.sort(z):
        if not levels or v - levels[-1][-1] > 0.05:
          levels.append([v])
        else:
          levels[-1].append(v)
      for lv in levels or [[0.0]]:
        on = loaded[:, p] & (np.abs(C[:, p, 2] - np.median(lv)) <= 0.05) if len(levels) > 1 else np.ones(T, bool)
        Fs.append(np.where(on[:, None], F[:, p], 0.0))
        Cs.append(np.where(on[:, None], C[:, p], np.nan))
        Ms.append(np.where(on, free[:, p], np.nan))
    F, C, free = np.stack(Fs, 1), np.stack(Cs, 1), np.stack(Ms, 1)
    corners = np.zeros((F.shape[1], 4, 3))
    half = np.array([[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]])
    for p in range(F.shape[1]):
      c = np.nanmedian(C[:, p], 0) if np.isfinite(C[:, p, 0]).any() else np.zeros(3)
      corners[p, :, :2] = c[:2] + half
      corners[p, :, 2] = c[2]
    plates = dict(force=F, cop=C, free=free, corners=corners, sign=sign, cop_check_mm=0.0,
                  corners_estimated=True, split_from=P)
  sex = str(sub.getBiologicalSex()).lower()
  anthro = dict(height_m=float(sub.getHeightM()) or None, mass_kg=float(sub.getMassKg()) or None,
                sex="m" if sex.startswith("m") else "f" if sex.startswith("f") else None)
  return dict(kind="b3d", labels=labels, xyz=X @ R.T, rate=1.0 / sub.getTrialTimestep(t), up=up, R=R,
              units="m", other=None, analog=None, plates=plates, anthro=anthro)


def read_trial(path, kind=None, up=None, units=None, exclude="", analog=".*",
               fp_sign="auto", fp_min_fz=20.0, anthro=None):
  """One trial file -> the structure in the module docstring. kind: from the extension
  when None. up: None (file frame), "auto", or the file's up axis. units: overrides the
  file's length unit. exclude: regex of C3D point labels that are not markers. analog:
  regex of the analog labels to keep ("" = none). fp_sign: "auto" (the vertical force
  on a loaded plate points up), 1 or -1 (factor on ezc3d's force and moment).
  fp_min_fz: N, no centre of pressure below. anthro: (height spec, mass spec) of
  C3D parameters, "GROUP:NAME" (e.g. "SUBJECT:height")."""
  kind = kind or kind_of(path)
  if kind == "c3d":
    return read_c3d(path, up, units, exclude, analog, fp_sign, fp_min_fz, anthro)
  if kind == "trc":
    return read_trc_file(path, up, units)
  if kind == "tak":
    return read_tak_file(path, up)
  if kind == "b3d":
    return read_b3d(path, up, fp_min_fz)
  raise ValueError(f"kind {kind!r}: expected c3d, trc, tak or b3d")


if __name__ == "__main__":
  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip())
    sys.exit(0)
  up = sys.argv[sys.argv.index("--up") + 1] if "--up" in sys.argv else "auto"
  r = read_trial(sys.argv[1], up=up, anthro=("SUBJECT:height", "SUBJECT:weight"))
  seen = np.mean(~np.isnan(r["xyz"][..., 0]), 0)
  print(f"{sys.argv[1]}: {r['kind']}, {len(r['xyz'])} frames @ {r['rate']:g} Hz, unit {r['units']}, "
        f"up {r['up']}; {len(r['labels'])} markers: "
        + ", ".join(f"{l} {s:.2f}" for l, s in zip(r["labels"], seen)))
  if r["other"]:
    print(f"other points: {len(r['other']['labels'])} ({sorted(set(r['other']['groups']))})")
  if r["analog"]:
    print(f"analog: {r['analog']['labels']} @ {r['analog']['rate']:g} Hz")
  if r["plates"]:
    pl = r["plates"]
    print(f"plates: {pl['force'].shape[1]}, sign {pl['sign']:+.0f}, peak vertical "
          f"{np.round(np.nanmax(pl['force'][..., 2], 0)).tolist()} N, centre of pressure vs "
          f"ezc3d {pl['cop_check_mm']:.2f} mm")
  if r.get("anthro"):
    print("anthropometrics:", r["anthro"])
