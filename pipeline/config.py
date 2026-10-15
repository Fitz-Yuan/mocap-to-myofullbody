"""Where the pipeline reads and writes, how the dataset names its files, and the
stage settings: every environment variable of the pipeline is read here.

Each setting can be overridden with the environment variable named next to it.
The defaults are the layout and settings of the case study (the MMH Elevator
box-lifting data); another dataset usually needs M2M_ROOT and the naming settings
below (or files named like the case study's):

  $M2M_ROOT/
    raw_tak/<trial>.tak                    trial takes; trials match M2M_TRIAL_GLOB
    raw_tak/static/<static name>.tak       one static standing trial per subject
                                           (M2M_STATIC_NAME), optional
    meta/anthropometrics.csv               subject, height (cm), mass (kg) per subject
                                           (M2M_ANTHRO_CSV, M2M_ANTHRO_COLUMNS)
    opensim_setup/                         OpenSim model, marker set and scale setup,
                                           only for the optional OpenSim comparison
                                           (M2M_OSIM_SETUP_DIR, M2M_OSIM_SCALE_SETUP)
    raw_daq/                               case study only: the force-plate recordings
                                           (tasks/box_lift/force_plates.py)
  written by the stages:
    tak_cache/   pipeline/cache_takes.py         the takes as arrays (static/ too)
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

Read elsewhere (they belong to the library, not to the stages): MYOFULLBODY_MODEL
(mocap2myo/model.py). The marker set is chosen in mocap2myo/markerset.py (MARKER_SET,
a source constant, not an environment variable). run_all.sh also reads PY and OSIM_PY
(the Python interpreters it runs the stages with).

Settings with a fixed set of values fail at import on anything else (a typo would
otherwise quietly fall back to another behaviour).
"""

import glob
import importlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
  return os.environ.get(name, default)


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

ROOT = os.path.expanduser(_env("M2M_ROOT", "~/data/mocap2myo"))
# input: the raw Motive takes (raw_tak/ -> cache_takes.py -> tak_cache/)
RAW_TAK_DIR = os.path.join(ROOT, "raw_tak")
TAK_CACHE_DIR = os.path.join(ROOT, "tak_cache")
NPZ_DIR = os.path.expanduser(_env("M2M_NPZ_DIR", os.path.join(ROOT, "markers")))
IK_DIR = os.path.expanduser(_env("M2M_IK_DIR", os.path.join(ROOT, "ik")))
REF_DIR = os.path.expanduser(_env("M2M_REF_DIR", os.path.join(ROOT, "ref")))
DATA_NAME = _env("M2M_DATA", "box_lift")  # dataset folder in data/ of the repository
DATA_DIR = os.path.join(REPO, "data", DATA_NAME)
DATA_STEM = _env("M2M_DATA_STEM", "box_lift")  # motion_<stem>[_ok]_100hz.npz

# --- dataset naming ------------------------------------------------------------

TRIAL_GLOB = _env("M2M_TRIAL_GLOB", "S*_NE")  # trial names (file stems)
SUBJECT_SEP = _env("M2M_SUBJECT_SEP", "_")  # subject = trial name up to the first one
STATIC_NAME = _env("M2M_STATIC_NAME", "{subject}_Anatomy")  # static trial of a subject
ANTHRO_CSV = os.path.expanduser(_env("M2M_ANTHRO_CSV",
                                     os.path.join(ROOT, "meta", "anthropometrics.csv")))
# CSV columns: subject, height (cm), mass (kg)
ANTHRO_COLUMNS = tuple(c.strip() for c in
                       _env("M2M_ANTHRO_COLUMNS", "Subject,Height,Weight").split(","))
# the CSV's subject cell -> the subject as in the file names (1 -> "S01"); a cell that
# is not a number (e.g. already "S01") is taken as it is
ANTHRO_SUBJECT = _env("M2M_ANTHRO_SUBJECT", "S{:02d}")


def subject_of(trial):
  """Subject of a trial (or static trial) name."""
  return trial.split(SUBJECT_SEP)[0]


# files the stages write next to the trials: <subject>_offsets.npz (solve_ik.py),
# <trial>_<a>-<b>.npz (solve_ik.py with M2M_IK_FRAMES), <name>.part.npz (being written)
_NOT_TRIAL = re.compile(r"(_offsets|_\d+-\d+|\.part)$")


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
  return int(_env("IK_WORKERS", 12))


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
# opt-in: a 3-hinge neck in the IK model (ik.NECK), so the head follows its markers; the
# IK output keeps the plain model's qpos layout plus neck_qpos [T, 3]
NECK = _flag("M2M_NECK", "0")
IK_FRAMES = _env("M2M_IK_FRAMES", "")  # "a:b": tracking IK on that frame range only (pilot / timing)
LOMO_EVERY_S = float(_env("M2M_LOMO_EVERY_S", 1.0))  # validate/lomo.py: one frame per this many s

# --- inverse dynamics (kinetics.py) and the OpenSim comparison ---------------------

KIN_DIR = os.path.expanduser(_env("M2M_KIN_DIR", os.path.join(ROOT, "kin")))
# low-pass (Hz) of the body (and carried object) positions before they are
# differentiated twice; also the OpenSim ID's filter (pipeline/opensim_id.py)
KIN_LOWPASS_HZ = float(_env("M2M_KIN_LOWPASS_HZ", 6.0))
if not KIN_LOWPASS_HZ > 0:
  raise SystemExit(f"M2M_KIN_LOWPASS_HZ={KIN_LOWPASS_HZ}: the inverse dynamics needs a "
                   "low-pass above 0 Hz (default 6)")
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
XSENS_C3D = os.path.expanduser(_env("M2M_XSENS_C3D",
                                    os.path.join(ROOT, "xsens_s11", f"{XSENS_TRIAL}.c3d")))
# validate/opensim_ik.py: Model.osim, markers.xml and the ScaleTool setup (see README)
OSIM_SETUP_DIR = os.path.expanduser(_env("M2M_OSIM_SETUP_DIR", os.path.join(ROOT, "opensim_setup")))
OSIM_SCALE_SETUP = os.path.join(OSIM_SETUP_DIR, os.path.expanduser(
  _env("M2M_OSIM_SCALE_SETUP", "Sub_09__Scalingsetup_09.xml")))
OSIM_DIR = os.path.join(ROOT, "opensim_ik")  # validate/opensim_ik.py output
OSIM_ID_DIR = os.path.join(ROOT, "opensim_id")  # opensim_id.py output
OSIM_COMPARE_DIR = os.path.join(ROOT, "compare_osim")  # validate/opensim_compare.py output
