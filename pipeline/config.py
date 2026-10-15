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
  written by the stages:
    tak_cache/   pipeline/cache_takes.py         the takes as arrays (static/ too)
    markers/     pipeline/prepare_markers.py     conditioned markers per trial, and
                                                 static/<subject>.npz (M2M_NPZ_DIR)
    ik/          pipeline/solve_ik.py            <subject>_offsets.npz, <trial>.npz;
                                                 qa.py, report.py and validate/lomo.py
                                                 add qa.json, report.json / .md,
                                                 lomo.json (M2M_IK_DIR)
    ref/         tasks/<task>/build_reference.py (M2M_REF_DIR)
  and the motion dataset in data/<M2M_DATA>/ of the repository.

Read elsewhere (they belong to the library, not to the stages): MYOFULLBODY_MODEL
(mocap2myo/model.py). The marker set is chosen in mocap2myo/markerset.py.
"""

import glob
import importlib
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
  return os.environ.get(name, default)


# --- data folders --------------------------------------------------------------

ROOT = os.path.expanduser(_env("M2M_ROOT", "~/data/mocap2myo"))
# "tak": the raw Motive takes (raw_tak/ -> cache_takes.py -> tak_cache/), the route
# run_all.sh uses. "export": TRC-as-xlsx exports made in Motive (markers_export/
# <trial>.xlsx, read through xlsx_cache/<trial>.npz with the same layout as tak_cache/)
SOURCE = _env("M2M_SOURCE", "export")
assert SOURCE in ("export", "tak"), SOURCE
RAW_TAK_DIR = os.path.join(ROOT, "raw_tak")
TAK_CACHE_DIR = os.path.join(ROOT, "tak_cache")
EXPORT_DIR = os.path.join(ROOT, "markers_export")
XLSX_CACHE_DIR = os.path.join(ROOT, "xlsx_cache")
RAW_TRC_DIR = os.path.join(ROOT, "raw_trc")
NPZ_DIR = os.path.expanduser(_env("M2M_NPZ_DIR", os.path.join(
  ROOT, "markers" if SOURCE == "tak" else "markers_from_export")))
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
ANTHRO_COLUMNS = tuple(_env("M2M_ANTHRO_COLUMNS", "Subject,Height,Weight").split(","))
# the CSV's subject cell -> the subject as in the file names (1 -> "S01")
ANTHRO_SUBJECT = _env("M2M_ANTHRO_SUBJECT", "S{:02d}")


def subject_of(trial):
  """Subject of a trial (or static trial) name."""
  return trial.split(SUBJECT_SEP)[0]


def trials(folder, ext=".npz"):
  """Sorted trial names with a file <name><ext> in folder."""
  return sorted(os.path.basename(p)[:-len(ext)]
                for p in glob.glob(os.path.join(folder, TRIAL_GLOB + ext)))


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


def anthropometrics():
  """{subject: {"height_cm", "mass_kg"}} from ANTHRO_CSV."""
  import numpy as np
  rows = np.genfromtxt(ANTHRO_CSV, delimiter=",", names=True, dtype=None, encoding=None)
  sc, hc, mc = ANTHRO_COLUMNS
  return {ANTHRO_SUBJECT.format(int(r[sc])): {"height_cm": float(r[hc]), "mass_kg": float(r[mc])}
          for r in rows}


def usage(doc):
  """-h / --help on a stage script: print its docstring and exit."""
  if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print(doc.strip())
    sys.exit(0)


def workers(default):
  """Parallel processes (IK_WORKERS); ~1-2 GB each for the IK."""
  return int(_env("IK_WORKERS", default))


# --- task ----------------------------------------------------------------------

# tasks/<M2M_TASK>/: the task-specific parts (objects, scene, reference, dataset);
# "none" = body motion only
TASK = _env("M2M_TASK", "box_lift")


def task():
  """The task package (tasks/<TASK>/), or None."""
  if TASK in ("", "none"):
    return None
  return importlib.import_module(f"tasks.{TASK}")


# --- IK (solve_ik.py; lomo.py repeats the tracking settings) ----------------------

OUTLIER = 0.05  # m, subject calibration: 1-3 markers farther than this = mislabelled
# tracking IK: marker error beyond which 1-3 markers count as mislabelled (0 = none;
# the calibration keeps OUTLIER)
TRACK_OUTLIER = float(_env("M2M_OUTLIER", OUTLIER))
# ik.ik_trial root_init; "pelvis" since IK v10 (2026-10-13): pelvis markers > 5 cm
# off 954 -> 265 frames (0.36 -> 0.10 %), the rest in deep squats past the model's
# joint limits (left to the dataset QA, tasks/box_lift/build_dataset.py)
ROOT_INIT = _env("M2M_ROOT_INIT", "pelvis")
MTP_PRIOR = float(_env("M2M_MTP_PRIOR", 0.1))
ANCHOR_CENTROID = _env("M2M_ANCHOR_CENTROID", "1") == "1"  # ik.ik_trial, see ik.MarkerModel.solve
PELVIS_GAP = _env("M2M_PELVIS_GAP", "solve")  # ik.ik_trial: box hides the pelvis
# ik.MarkerModel.solve: Cauchy marker loss (m) instead of the outlier re-solve
# (markers flipping in / out at the threshold made the root jump); 0 = re-solve.
# Tracking IK only (the subject calibration keeps the re-solve, see solve_ik.calib)
ROBUST = float(_env("M2M_ROBUST", 0.025))
LOWPASS_HZ = float(_env("M2M_LOWPASS_HZ", 6.0))  # ik.load_trial (0 = raw)
# opt-in: anchor the subject calibration on the static standing trial
# (ik.calibrate static=...; <M2M_NPZ_DIR>/static/<subject>.npz from
# prepare_markers.py, tak route). A subject without one uses quiet-standing frames
# of its trials.
STATIC = _env("M2M_STATIC", "0") == "1"
STATIC_SHARE = float(_env("M2M_STATIC_SHARE", 0.5))
STATIC_W = float(_env("M2M_STATIC_W", 1.0))
# "legs": the static trial places only pelvis / leg markers (arm and trunk markers
# from the trials alone); "legs_only": leg markers only (pelvis from the trials);
# "all": every marker (first try, 2026-10-15)
STATIC_MARKERS = _env("M2M_STATIC_MARKERS", "legs")
# Foot markers sit on thick-soled slides (TOE on the toe box): S02 / S07 hit
# the 6 cm cap with the flat-foot calibration -> 10 cm
FOOT_CAP = float(_env("M2M_FOOT_CAP", 0.10))
FLAT_W = float(_env("M2M_FLAT_W", 30.0))
IK_FRAMES = _env("M2M_IK_FRAMES", "")  # "a:b": tracking IK on that frame range only (pilot / timing)
LOMO_EVERY_S = float(_env("M2M_LOMO_EVERY_S", 1.0))  # validate/lomo.py: one frame per this many s

# --- validation inputs ---------------------------------------------------------

XSENS_TRIAL = _env("M2M_XSENS_TRIAL", "S11_Spring_NE")  # trial with a usable Xsens c3d
XSENS_C3D = os.path.expanduser(_env("M2M_XSENS_C3D",
                                    os.path.join(ROOT, "xsens_s11", f"{XSENS_TRIAL}.c3d")))
OSIM_SETUP_DIR = os.path.join(ROOT, "opensim_setup")  # Model.osim, markers.xml, scale setup
OSIM_DIR = os.path.join(ROOT, "opensim_ik")  # validate/opensim_ik.py output
OSIM_COMPARE_DIR = os.path.join(ROOT, "compare_osim")  # validate/opensim_compare.py output
