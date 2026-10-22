"""Where the pipeline reads and writes, how the dataset names its files, and the
stage settings: every environment variable of the pipeline is read here.

Each setting is taken from, in this order: its environment variable (named next to
it), the dataset profile, the default here. The defaults are the layout and
settings of the case study (the MMH Elevator box-lifting data, OptiTrack .tak); with
no profile and no variable set, nothing differs from that.

Dataset profile (M2M_DATASET=path/to/profile.toml; examples in datasets/ of the
repository): a TOML file with the settings of one dataset. Its keys are the setting
names below without "M2M_", in lower case (trial_glob = "S*_NE" is M2M_TRIAL_GLOB);
tables ([input], [naming], ...) only group them, and a key may appear once. "name"
and "description" are free text. Values: strings, numbers, true / false (= 1 / 0),
lists (joined with commas). A relative path in a profile is relative to M2M_ROOT
(root itself: to the profile's folder). An unknown key stops the run.

Input (M2M_INPUT): OptiTrack .tak (mocap2myo.tak), C3D or TRC (mocap2myo.readers).
Trial files: M2M_RAW_TRIALS (a glob under M2M_RAW_DIR, ** allowed); a trial's name is
M2M_TRIAL_NAME filled with {stem} (the file name without extension) and the named
groups of M2M_TRIAL_REGEX (matched against the whole path relative to M2M_RAW_DIR,
"/" separated; files it does not match are skipped). The subject is the trial name
up to the first M2M_SUBJECT_SEP, so the name must start with it, and the names must
match M2M_TRIAL_GLOB (how the later stages find them). Static trials: M2M_RAW_STATIC
and M2M_STATIC_REGEX (a group "subject"), one per subject at most, cached and named
M2M_STATIC_NAME. Defaults, as the case study:

  $M2M_ROOT/
    raw_tak/<trial>.tak                    trial takes (M2M_RAW_DIR, M2M_RAW_TRIALS);
                                           trials match M2M_TRIAL_GLOB
    raw_tak/static/<static name>.tak       one static standing trial per subject
                                           (M2M_STATIC_NAME, M2M_RAW_STATIC), optional
    meta/anthropometrics.csv               subject, height (cm), mass (kg) per subject
                                           (M2M_ANTHRO_CSV, M2M_ANTHRO_COLUMNS; with
                                           M2M_ANTHRO=c3d read from the C3D files)
    opensim_setup/                         OpenSim model, marker set and scale setup,
                                           only for the optional OpenSim comparison
                                           (M2M_OSIM_SETUP_DIR, M2M_OSIM_SCALE_SETUP)
    raw_daq/                               case study only: the force-plate recordings
                                           (tasks/box_lift/force_plates.py)
  written by the stages:
    tak_cache/   pipeline/read_raw.py            the raw files as arrays (static/ too;
                                                 M2M_CACHE_DIR, default raw_cache/ for
                                                 C3D / TRC)
    markers/     pipeline/prepare_markers.py     conditioned markers per trial, and
                                                 static/<subject>.npz (M2M_NPZ_DIR)
    ik/          pipeline/solve_ik.py            <subject>_offsets.npz, <trial>.npz;
                                                 qa.py, report.py and validate/lomo.py
                                                 add qa.json, report.json / .md,
                                                 lomo.json (M2M_IK_DIR)
    ref/         tasks/<task>/build_reference.py (M2M_REF_DIR)
    kin/         pipeline/kinetics.py            joint loads per trial (M2M_KIN_DIR)
    opensim_ik/, opensim_id/, compare_osim/   validate/opensim_ik.py,
                                                 pipeline/opensim_id.py,
                                                 validate/opensim_compare.py
                                                 (optional OpenSim comparison)
  and the motion dataset in data/<M2M_DATA>/ of the repository.

Model and marker set: M2M_MODEL names the model profile (mocap2myo/profiles/<name>.toml,
default myofullbody; the others need models/fetch.py first, M2M_MODELS_DIR), and
M2M_MARKER_SET the marker set (mocap2myo/markersets/<name>.toml, default plug_in_gait),
M2M_MARKER_ALIASES which of its alias label sets apply. Both are written back to the
environment, where the library reads them (mocap2myo.profile.selected,
mocap2myo.markerset.selected). Read elsewhere (they belong to the library, not to the
stages): MYOFULLBODY_MODEL (mocap2myo/model.py, the MyoFullBody profile's variants),
M2M_MODELS_DIR (mocap2myo/profile.py). run_all.sh also reads PY and OSIM_PY (the Python
interpreters it runs the stages with). IK_WORKERS is not a dataset setting (no profile key).

Settings with a fixed set of values fail at import on anything else (a typo would
otherwise quietly fall back to another behaviour).
"""

import glob
import importlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _text(v, where):
  """A profile value as the text its environment variable would hold."""
  if isinstance(v, bool):
    return "1" if v else "0"
  if isinstance(v, (int, float, str)):
    return str(v)
  if isinstance(v, list) and all(isinstance(x, (int, float, str)) and not isinstance(x, bool)
                                 for x in v):
    return ",".join(str(x) for x in v)
  raise SystemExit(f"{where}: {v!r} is not a string, number, true / false or list")


def _load_profile(path):
  """{M2M_<KEY>: text} from a dataset profile (TOML; see the module docstring)."""
  try:
    import tomllib
  except ModuleNotFoundError:  # Python 3.10
    try:
      import tomli as tomllib
    except ModuleNotFoundError:
      raise SystemExit("M2M_DATASET: a profile needs Python >= 3.11 or tomli (pip install tomli)")
  try:
    with open(os.path.expanduser(path), "rb") as f:
      data = tomllib.load(f)
  except (OSError, tomllib.TOMLDecodeError) as e:
    raise SystemExit(f"M2M_DATASET={path}: {e}")
  out = {}

  def walk(table, where):
    for k, v in table.items():
      if isinstance(v, dict):
        walk(v, f"{where}{k}.")
      elif not (where == "" and k in ("name", "description")):
        name = "M2M_" + k.upper()
        if name in out:
          raise SystemExit(f"{path}: {k} is given twice")
        out[name] = _text(v, f"{path}: {where}{k}")
  walk(data, "")
  return out, str(data.get("name", os.path.splitext(os.path.basename(path))[0]))


# the dataset profile: settings that no environment variable sets
PROFILE = os.environ.get("M2M_DATASET", "")
_PROFILE, PROFILE_NAME = _load_profile(PROFILE) if PROFILE else ({}, "")
_USED = set()  # settings read (a profile key that is none of them stops the run)


def _env(name, default):
  _USED.add(name)
  if name in os.environ:
    return os.environ[name]
  return _PROFILE.get(name, default)


def _path(name, default):
  """A folder / file setting: the variable as given (~ expanded), a profile path
  (relative to ROOT), or the default."""
  _USED.add(name)
  if name in os.environ:
    return os.path.expanduser(os.environ[name])
  if name in _PROFILE:
    p = os.path.expanduser(_PROFILE[name])
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
  return default


def _regex(name, default, groups=()):
  """A regular-expression setting; must compile (and have the named groups)."""
  v = _env(name, default)
  try:
    rx = re.compile(v)
  except re.error as e:
    raise SystemExit(f"{name}={v!r}: {e}")
  missing = [g for g in groups if g not in rx.groupindex]
  if missing:
    raise SystemExit(f"{name}={v!r}: needs the named group(s) {missing}, e.g. (?P<subject>...)")
  return v


def _choice(name, default, allowed):
  """An enum setting: its value must be one of `allowed`."""
  v = _env(name, default)
  if v not in allowed:
    raise SystemExit(f"{name}={v!r}: expected one of {', '.join(allowed)}")
  return v


def _flag(name, default):
  """An on / off setting: "1" or "0"."""
  return _choice(name, default, ("0", "1")) == "1"


# --- data folders --------------------------------------------------------------

_USED.add("M2M_ROOT")
if "M2M_ROOT" in os.environ or "M2M_ROOT" not in _PROFILE:
  ROOT = os.path.expanduser(os.environ.get("M2M_ROOT", "~/data/mocap2myo"))
else:  # a relative root in a profile: relative to the profile's folder
  ROOT = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(PROFILE))),
                      os.path.expanduser(_PROFILE["M2M_ROOT"]))
# input: the raw recordings (M2M_RAW_DIR -> read_raw.py -> M2M_CACHE_DIR)
INPUT = _choice("M2M_INPUT", "tak", ("tak", "c3d", "trc"))  # kind of the raw files
RAW_DIR = _path("M2M_RAW_DIR", os.path.join(ROOT, "raw_tak"))
CACHE_DIR = _path("M2M_CACHE_DIR",
                  os.path.join(ROOT, "tak_cache" if INPUT == "tak" else "raw_cache"))
NPZ_DIR = _path("M2M_NPZ_DIR", os.path.join(ROOT, "markers"))
IK_DIR = _path("M2M_IK_DIR", os.path.join(ROOT, "ik"))
REF_DIR = _path("M2M_REF_DIR", os.path.join(ROOT, "ref"))
DATA_NAME = _env("M2M_DATA", "box_lift")  # dataset folder in data/ of the repository
DATA_DIR = os.path.join(REPO, "data", DATA_NAME)
DATA_STEM = _env("M2M_DATA_STEM", "box_lift")  # motion_<stem>[_ok]_100hz.npz

# --- dataset naming ------------------------------------------------------------

TRIAL_GLOB = _env("M2M_TRIAL_GLOB", "S*_NE")  # trial names (in every stage folder)
SUBJECT_SEP = _env("M2M_SUBJECT_SEP", "_")  # subject = trial name up to the first one
STATIC_NAME = _env("M2M_STATIC_NAME", "{subject}_Anatomy")  # static trial of a subject
if STATIC_NAME.count("{subject}") != 1:
  raise SystemExit(f"M2M_STATIC_NAME={STATIC_NAME!r}: needs {{subject}} once")
_EXT = "." + INPUT
# trial files: a glob under RAW_DIR (** = any folders); their names, see the docstring
RAW_TRIALS = _env("M2M_RAW_TRIALS", TRIAL_GLOB + _EXT)
TRIAL_REGEX = _regex("M2M_TRIAL_REGEX", "")  # "" = every file the glob finds
TRIAL_NAME = _env("M2M_TRIAL_NAME", "{stem}")
# static trial files and the subject of each (the regex's group "subject")
_pre, _post = STATIC_NAME.split("{subject}")
RAW_STATIC = _env("M2M_RAW_STATIC", "static/" + _pre + "*" + _post + _EXT)
STATIC_REGEX = _regex("M2M_STATIC_REGEX", "static/" + re.escape(_pre) + "(?P<subject>.+)"
                      + re.escape(_post + _EXT), groups=("subject",))
# read_raw.py: the raw files' up axis ("y" for a Motive take, whose world is Y-up and which
# prepare_markers.py turns Z-up; C3D / TRC: "auto" = from the head and foot markers, or
# x / y / z, optionally with a sign) and length unit ("" = the file's own)
UP = _choice("M2M_UP", "y" if INPUT == "tak" else "auto",
             ("y",) if INPUT == "tak" else ("auto", "x", "y", "z", "-x", "-y", "-z"))
UNITS = _choice("M2M_UNITS", "", ("", "m", "cm", "mm"))
# frame rate: "native" = the file's; a number = keep every k-th frame, k = native / it
# (an integer, e.g. 200 Hz -> 100 Hz; the plates come along, the analog channels keep
# their own rate)
_rate = _env("M2M_RATE", "native")
try:
  RATE = None if _rate == "native" else float(_rate)
except ValueError:
  raise SystemExit(f"M2M_RATE={_rate!r}: expected native or a rate in Hz")
if RATE is not None and not RATE > 0:
  raise SystemExit(f"M2M_RATE={_rate!r}: expected native or a rate > 0 Hz")
# C3D: point labels that are not markers although no POINT type group lists them (kept
# with the model outputs), e.g. Plug-in-Gait's segment points; analog channels to keep
# (EMG, ...; force-plate channels never: they become the plates)
POINTS_EXCLUDE = _regex("M2M_POINTS_EXCLUDE", "")
ANALOG = _regex("M2M_ANALOG", ".*")  # "" = none
# C3D force plates: sign of ezc3d's force and moment ("auto": the vertical force on a
# loaded plate points up) and the vertical force (N) below which there is no centre of
# pressure
FP_SIGN = _choice("M2M_FP_SIGN", "auto", ("auto", "1", "-1"))
FP_MIN_FZ = float(_env("M2M_FP_MIN_FZ", 20.0))
# the marker set the dataset uses (mocap2myo/markersets/<name>.toml; see marker_set()) and
# which of its alias label sets a recording may use ("auto": the one that matches its labels
# best, if it beats the set's own labels; "none"; or names, comma separated)
_SETS = sorted(f[:-5] for f in os.listdir(os.path.join(REPO, "mocap2myo", "markersets"))
               if f.endswith(".toml"))
MARKER_SET = _choice("M2M_MARKER_SET", "plug_in_gait", _SETS)
MARKER_ALIASES = _env("M2M_MARKER_ALIASES", "auto")
# the model (mocap2myo/profiles/<name>.toml; see model_profile())
_MODELS = sorted(f[:-5] for f in os.listdir(os.path.join(REPO, "mocap2myo", "profiles"))
                 if f.endswith(".toml") and f != "sources.toml")
MODEL = _choice("M2M_MODEL", "myofullbody", _MODELS)
# the library reads the same two (mocap2myo.markerset.selected, mocap2myo.profile.selected;
# also in the stages' worker processes)
os.environ["M2M_MARKER_SET"] = MARKER_SET
os.environ["M2M_MODEL"] = MODEL
# subject heights and masses: "csv" (ANTHRO_CSV) or "c3d" (C3D parameters, "GROUP:NAME":
# the entry NAME of the group's LABELS / VALUES / UNITS, or a parameter NAME; collected by
# read_raw.py into <M2M_CACHE_DIR>/anthropometrics.csv, which is then ANTHRO_CSV)
ANTHRO = _choice("M2M_ANTHRO", "csv", ("csv", "c3d"))
ANTHRO_C3D_HEIGHT = _env("M2M_ANTHRO_C3D_HEIGHT", "SUBJECT:height")
ANTHRO_C3D_MASS = _env("M2M_ANTHRO_C3D_MASS", "SUBJECT:weight")
if ANTHRO == "c3d" and INPUT != "c3d":
  raise SystemExit("M2M_ANTHRO=c3d needs M2M_INPUT=c3d")
ANTHRO_CSV = _path("M2M_ANTHRO_CSV", os.path.join(CACHE_DIR, "anthropometrics.csv")
                   if ANTHRO == "c3d" else os.path.join(ROOT, "meta", "anthropometrics.csv"))
# CSV columns: subject, height (cm), mass (kg)
ANTHRO_COLUMNS = tuple(c.strip() for c in
                       _env("M2M_ANTHRO_COLUMNS", "Subject,Height,Weight").split(","))
# the CSV's subject cell -> the subject as in the file names (1 -> "S01"); a cell that
# is not a number (e.g. already "S01") is taken as it is (and every cell with
# M2M_ANTHRO=c3d)
ANTHRO_SUBJECT = _env("M2M_ANTHRO_SUBJECT", "S{:02d}")


def subject_of(trial):
  """Subject of a trial (or static trial) name."""
  return trial.split(SUBJECT_SEP)[0]


# files the stages write next to the trials: <subject>_offsets.npz (solve_ik.py),
# <trial>_<a>-<b>.npz (solve_ik.py with M2M_IK_FRAMES), <name>.part.npz (being written)
_NOT_TRIAL = re.compile(r"(_offsets|_\d+-\d+|\.part)$")


def _raw_files(pattern, regex):
  """[(path, stem, regex groups)] of the files under RAW_DIR matching the glob (and the
  regex, on the "/"-separated path relative to RAW_DIR), and the paths it skipped."""
  out, skipped = [], []
  for p in sorted(glob.glob(os.path.join(RAW_DIR, pattern), recursive=True)):
    if not os.path.isfile(p):
      continue
    rel = os.path.relpath(p, RAW_DIR).replace(os.sep, "/")
    m = re.fullmatch(regex, rel) if regex else None
    if regex and m is None:
      skipped.append(rel)
      continue
    out.append((p, os.path.splitext(os.path.basename(p))[0], m.groupdict() if m else {}))
  return out, skipped


def raw_trials():
  """({trial name: path}, skipped paths) of the trial files (sorted by name)."""
  import fnmatch
  found, skipped = _raw_files(RAW_TRIALS, TRIAL_REGEX)
  out = {}
  for p, stem, g in found:
    try:
      name = TRIAL_NAME.format(stem=stem, **g)
    except (KeyError, IndexError, ValueError) as e:
      raise SystemExit(f"M2M_TRIAL_NAME={TRIAL_NAME!r} for {p}: {e!r} (fields: stem and the "
                       f"named groups of M2M_TRIAL_REGEX {sorted(g)})")
    if _NOT_TRIAL.search(name):
      skipped.append(os.path.relpath(p, RAW_DIR))
      continue
    if "subject" in g and subject_of(name) != g["subject"]:
      raise SystemExit(f"{p}: trial name {name!r} does not start with its subject "
                       f"{g['subject']!r} + M2M_SUBJECT_SEP {SUBJECT_SEP!r}")
    if not fnmatch.fnmatchcase(name, TRIAL_GLOB):
      raise SystemExit(f"{p}: trial name {name!r} does not match M2M_TRIAL_GLOB "
                       f"{TRIAL_GLOB!r}, which the later stages find trials by")
    if name in out:
      raise SystemExit(f"two files give the trial name {name!r}: {out[name]}, {p}")
    out[name] = p
  return dict(sorted(out.items())), skipped


def raw_statics():
  """({subject: path}, skipped paths) of the static trial files."""
  found, skipped = _raw_files(RAW_STATIC, STATIC_REGEX)
  out = {}
  for p, stem, g in found:
    s = g["subject"]
    if s in out:
      raise SystemExit(f"two static trials for subject {s}: {out[s]}, {p} "
                       "(M2M_RAW_STATIC / M2M_STATIC_REGEX must pick one)")
    out[s] = p
  return dict(sorted(out.items())), skipped


def trials(folder, ext=".npz"):
  """Sorted trial names with a file <name><ext> in folder (matching M2M_TRIAL_GLOB)."""
  names = (os.path.basename(p)[:-len(ext)]
           for p in glob.glob(os.path.join(folder, TRIAL_GLOB + ext)))
  return sorted(n for n in names if not _NOT_TRIAL.search(n))


def static_name(subject):
  return STATIC_NAME.format(subject=subject)


def static_files(folder, ext):
  """Sorted (subject, path) of the static trials in folder."""
  pre, post = STATIC_NAME.split("{subject}")
  pat = re.compile(re.escape(pre) + "(.+)" + re.escape(post + ext) + "$")
  out = []
  for p in sorted(glob.glob(os.path.join(folder, static_name("*") + ext))):
    m = pat.match(os.path.basename(p))
    if m:
      out.append((m.group(1), p))
  return out


def _anthro_subject(cell):
  """The CSV's subject cell -> the subject as in the file names (ANTHRO_SUBJECT)."""
  if ANTHRO == "c3d":
    return cell  # read_raw.py wrote the subjects as named in the trials
  try:
    num = float(cell)
  except ValueError:
    return cell  # already a name, e.g. "S01"
  return ANTHRO_SUBJECT.format(int(num))


def _anthro_rows():
  """(subject, row dict) per CSV row with a subject (header names as written)."""
  import csv
  with open(ANTHRO_CSV, newline="", encoding="utf-8-sig") as f:
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(f)]
  sc = ANTHRO_COLUMNS[0]
  need = list(ANTHRO_COLUMNS) + ([CARRIED_COLUMN] if CARRIED_COLUMN else [])
  have = set(rows[0]) if rows else set()
  missing = [c for c in need if c not in have]
  if missing and set(missing) != {CARRIED_COLUMN}:
    raise SystemExit(f"{ANTHRO_CSV}: no column {missing} (columns: {sorted(have)}; "
                     "set M2M_ANTHRO_COLUMNS / M2M_CARRIED_COLUMN)")
  return [(_anthro_subject(r[sc]), r) for r in rows if r.get(sc)]


def anthropometrics():
  """{subject: {"height_cm", "mass_kg"}} from ANTHRO_CSV."""
  _, hc, mc = ANTHRO_COLUMNS
  return {s: {"height_cm": float(r[hc]), "mass_kg": float(r[mc])} for s, r in _anthro_rows()}


def usage(doc):
  """-h / --help on a stage script: print its docstring and exit."""
  if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(doc.strip())
    sys.exit(0)


def workers():
  """Parallel processes of every parallel stage (IK_WORKERS, default 12); ~1-2 GB each
  for the IK."""
  return int(os.environ.get("IK_WORKERS", 12))


# --- task ----------------------------------------------------------------------

# tasks/<M2M_TASK>/: the task-specific parts (objects, scene, reference, dataset);
# "none" = body motion only
TASK = _env("M2M_TASK", "box_lift")


def task():
  """The task package (tasks/<TASK>/), or None."""
  if TASK in ("", "none"):
    return None
  if not os.path.exists(os.path.join(REPO, "tasks", TASK, "__init__.py")):
    raise SystemExit(f"M2M_TASK={TASK!r}: no package tasks/{TASK}/ (or 'none' for the body only)")
  return importlib.import_module(f"tasks.{TASK}")


# --- IK (solve_ik.py; lomo.py repeats the tracking settings) ----------------------

OUTLIER = 0.05  # m, subject calibration: 1-3 markers farther than this = mislabelled
# tracking IK: marker error beyond which 1-3 markers count as mislabelled (0 = none;
# the calibration keeps OUTLIER)
TRACK_OUTLIER = float(_env("M2M_OUTLIER", OUTLIER))
# ik.ik_trial root_init ("prev" or "pelvis"); with "pelvis" (case study): pelvis
# markers > 5 cm off 954 -> 265 frames (0.36 -> 0.10 %), the rest in deep squats past
# the model's joint limits (left to the dataset QA, tasks/box_lift/build_dataset.py)
ROOT_INIT = _choice("M2M_ROOT_INIT", "pelvis", ("prev", "pelvis"))
MTP_PRIOR = float(_env("M2M_MTP_PRIOR", 0.1))
ANCHOR_CENTROID = _flag("M2M_ANCHOR_CENTROID", "1")  # ik.ik_trial, see ik.MarkerModel.solve
# ik.ik_trial pelvis_gap ("carry" or "solve"): frames with a pelvis marker missing
# (the carried box hides it)
PELVIS_GAP = _choice("M2M_PELVIS_GAP", "solve", ("carry", "solve"))
# ik.MarkerModel.solve: Cauchy marker loss (m) instead of the outlier re-solve
# (markers flipping in / out at the threshold made the root jump); 0 = re-solve.
# Tracking IK only (the subject calibration keeps the re-solve, see solve_ik.calib)
ROBUST = float(_env("M2M_ROBUST", 0.025))
LOWPASS_HZ = float(_env("M2M_LOWPASS_HZ", 6.0))  # ik.load_trial: marker low-pass (0 = raw)
if LOWPASS_HZ < 0:
  raise SystemExit(f"M2M_LOWPASS_HZ={LOWPASS_HZ}: expected >= 0 (0 = no filter)")
# anchor the subject calibration on the static standing trial (ik.calibrate static=...;
# <M2M_NPZ_DIR>/static/<subject>.npz from prepare_markers.py). A subject without one
# uses quiet-standing frames of its trials. Off by default here; run_all.sh turns it
# on (M2M_STATIC=1), as in the case study's results.
STATIC = _flag("M2M_STATIC", "0")
STATIC_SHARE = float(_env("M2M_STATIC_SHARE", 0.5))
STATIC_W = float(_env("M2M_STATIC_W", 1.0))
# "legs": the static trial places only pelvis / leg markers (arm and trunk markers
# from the trials alone); "legs_only": leg markers only (pelvis from the trials);
# "all": every marker
STATIC_MARKERS = _choice("M2M_STATIC_MARKERS", "legs", ("legs", "legs_only", "all"))
# calibration: how far (m) a foot marker may move from its anatomical guess. Case
# study: the foot markers sit on thick-soled slides (TOE on the toe box), and two
# subjects hit a 6 cm cap with the flat-foot calibration -> 10 cm
FOOT_CAP = float(_env("M2M_FOOT_CAP", 0.10))
FLAT_W = float(_env("M2M_FLAT_W", 30.0))
# frame-to-frame smoothing of the profile's smooth_heavy joints (ankle, subtalar), per frame at
# 100 Hz (ik.ik_trial ankle_smooth). 0.3 held the ankle still in the case study's deep squats;
# walking moves the ankle faster, and gait datasets set less in their profile
IK_ANKLE_SMOOTH = float(_env("M2M_IK_ANKLE_SMOOTH", 0.3))
# opt-in: a 3-hinge neck in the IK model (ik.NECK), so the head follows its markers; the
# IK output keeps the plain model's qpos layout plus neck_qpos [T, 3]
NECK = _flag("M2M_NECK", "0")
IK_FRAMES = _env("M2M_IK_FRAMES", "")  # "a:b": tracking IK on that frame range only (pilot / timing)
LOMO_EVERY_S = float(_env("M2M_LOMO_EVERY_S", 1.0))  # validate/lomo.py: one frame per this many s

# --- inverse dynamics (kinetics.py) and the OpenSim comparison ---------------------

KIN_DIR = _path("M2M_KIN_DIR", os.path.join(ROOT, "kin"))
# low-pass (Hz) of the body (and carried object) positions before they are
# differentiated twice; also the OpenSim ID's filter (pipeline/opensim_id.py)
KIN_LOWPASS_HZ = float(_env("M2M_KIN_LOWPASS_HZ", 6.0))
if not KIN_LOWPASS_HZ > 0:
  raise SystemExit(f"M2M_KIN_LOWPASS_HZ={KIN_LOWPASS_HZ}: the inverse dynamics needs a "
                   "low-pass above 0 Hz (default 6)")
# force plates in the inverse dynamics (C3D inputs): low-passed with the body positions'
# filter (KIN_LOWPASS_HZ, mocap2myo.kinetics.lowpass_plates), so that force and motion carry
# the same frequencies; filtered differently they put impact artefacts into the joint
# moments around heel strike (Kristianslund et al. 2012, J Biomech 45:666). "0": the plates
# unfiltered (as before; for comparison)
KIN_FILTER_PLATES = _flag("M2M_KIN_FILTER_PLATES", "1")
# the ground reaction shared between the feet off the plates (mocap2myo.kinetics
# Kinetics.ground_split): "lever" = the lever rule along the line between the feet, every
# component in the vertical's proportion; "vpp" = graded sole contact (a heel landing fast
# counts) and the feet's forces converging 0.35 m above the centre of mass, so the trailing
# foot pushes forward while the leading one brakes. Against the plates of two public gait
# datasets (double support, the plates left out): hip / knee / ankle moment RMS error
# 59 / 66 / 88 -> 23 / 36 / 53 N m (level walking, 36 trials) and 49 / 41 / 39 -> 13 / 17 /
# 20 N m (Moissenet, 40 trials); the case study's right-foot share of the vertical force
# against its plates: RMS error 0.080 -> 0.070, r 0.88 both
KIN_SPLIT = _choice("M2M_KIN_SPLIT", "lever", ("lever", "vpp"))
# Mass while recorded, if the subjects wore equipment (here the COSMED K5 metabolic
# system, about 1-3 kg): a column of the anthropometrics CSV with the total mass; the
# difference to the body mass is lumped at CARRIED_BODY's centre of mass. "" = none.
CARRIED_COLUMN = _env("M2M_CARRIED_COLUMN", "Wgt+K5")
CARRIED_BODY = _env("M2M_CARRIED_BODY", "torso")


def carried_mass():
  """{subject: kg of equipment worn} from CARRIED_COLUMN (empty if there is none)."""
  if not CARRIED_COLUMN:
    return {}
  mc = ANTHRO_COLUMNS[2]
  return {s: float(r[CARRIED_COLUMN]) - float(r[mc]) for s, r in _anthro_rows()
          if r.get(CARRIED_COLUMN)}


# --- validation inputs ---------------------------------------------------------

XSENS_TRIAL = _env("M2M_XSENS_TRIAL", "S11_Spring_NE")  # trial with a usable Xsens c3d
XSENS_C3D = _path("M2M_XSENS_C3D", os.path.join(ROOT, "xsens_s11", f"{XSENS_TRIAL}.c3d"))
# validate/opensim_ik.py: Model.osim, markers.xml and the ScaleTool setup (see README)
OSIM_SETUP_DIR = _path("M2M_OSIM_SETUP_DIR", os.path.join(ROOT, "opensim_setup"))
OSIM_SCALE_SETUP = os.path.join(OSIM_SETUP_DIR, os.path.expanduser(
  _env("M2M_OSIM_SCALE_SETUP", "Sub_09__Scalingsetup_09.xml")))
OSIM_DIR = os.path.join(ROOT, "opensim_ik")  # validate/opensim_ik.py output
OSIM_ID_DIR = os.path.join(ROOT, "opensim_id")  # opensim_id.py output
OSIM_COMPARE_DIR = os.path.join(ROOT, "compare_osim")  # validate/opensim_compare.py output


def marker_set():
  """The dataset's marker set (M2M_MARKER_SET; a mocap2myo.markerset.MarkerSet), with
  M2M_MARKER_ALIASES checked against its alias sets."""
  from mocap2myo import markerset as ms
  s = ms.load(MARKER_SET)
  if MARKER_ALIASES not in ("auto", "none"):
    bad = [a for a in MARKER_ALIASES.split(",") if a.strip() not in s.ALIASES]
    if bad:
      raise SystemExit(f"M2M_MARKER_ALIASES={MARKER_ALIASES!r}: {MARKER_SET} has no alias set(s) {bad} "
                       f"(it has {sorted(s.ALIASES) or 'none'})")
  return s


def marker_aliases(s):
  """{name: alias map} of the marker set's alias sets a recording may use."""
  if MARKER_ALIASES == "none":
    return {}
  if MARKER_ALIASES == "auto":
    return dict(s.ALIASES)
  return {a.strip(): s.ALIASES[a.strip()] for a in MARKER_ALIASES.split(",")}


def model_profile():
  """The model (M2M_MODEL; a mocap2myo.profile.Profile)."""
  from mocap2myo import profile
  return profile.load(MODEL)


def marker_model(neck=False):
  """ik.MarkerModel of the dataset's model and marker set."""
  from mocap2myo import ik
  return ik.MarkerModel(model_profile(), marker_set(), neck=neck)


# a profile key that is no setting (a typo would otherwise be ignored quietly)
_unknown = sorted(set(_PROFILE) - _USED)
if _unknown:
  raise SystemExit(f"{PROFILE}: unknown setting(s) {[k[4:].lower() for k in _unknown]}; "
                   f"known: {sorted(k[4:].lower() for k in _USED if k.startswith('M2M_'))}")
