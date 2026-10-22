#!/bin/bash
# The whole pipeline, from raw recordings to motion clips (the case study's settings).
#   M2M_ROOT=~/data/mocap2myo bash pipeline/run_all.sh
#   M2M_DATASET=datasets/ramp_vielemeyer2026.toml bash pipeline/run_all.sh   (a profile)
# Inputs under $M2M_ROOT: raw_tak/<trial>.tak, raw_tak/static/<static name>.tak
# (M2M_STATIC_NAME), meta/anthropometrics.csv (Subject, Height cm, Weight kg), or what
# the dataset profile says (C3D / TRC files); names and folders: see pipeline/config.py.
# Outputs: tak_cache/ (raw_cache/), markers/, ik/ (IK, QA, report), ref/, kin/ (joint
# loads), and data/<M2M_DATA> in the repository. With M2M_TASK=none only the body
# motion and its joint loads (no reference, no dataset, no external loads). CPU only.
# IK_WORKERS parallel processes (default 12, ~1-2 GB each). Without a profile the
# static-trial calibration is turned on here (M2M_STATIC=1); with one, the profile says.
# PY: the Python to run the stages with (default: python).
# --opensim: also run the OpenSim workflow on the same markers (Scale + Marker Placer +
# IK, then Inverse Dynamics with the same loads) for the side-by-side comparison; needs
# the opensim package (conda install -c opensim-org opensim), in OSIM_PY if that is
# another Python (default: the same as PY).
# --muscles: also the muscle forces by static optimisation on the joint loads
# (pipeline/muscles.py, settings M2M_MUSCLE_*), after the joint loads.
set -eu
OPENSIM=0
MUSCLES=0
for arg in "$@"; do
  case "$arg" in
    --opensim) OPENSIM=1 ;;
    --muscles) MUSCLES=1 ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PY:-python}
OSIM_PY=${OSIM_PY:-$PY}
export OMP_NUM_THREADS=1
if [ -z "${M2M_DATASET:-}" ]; then
  export M2M_STATIC=${M2M_STATIC:-1}          # calibrate on the static standing trials (config.py: 0)
  export M2M_STATIC_MARKERS=${M2M_STATIC_MARKERS:-legs}
fi
# the task as config.py resolves it (variable, profile, default)
TASK=$(cd "$REPO" && $PY -c "from pipeline import config; print(config.TASK)")
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "read raw files"; $PY -u "$HERE/read_raw.py"         # .tak: olefile, C3D: ezc3d
say "markers";       $PY -u "$HERE/prepare_markers.py"
say "IK";            $PY -u "$HERE/solve_ik.py"
if [ -f "$REPO/tasks/$TASK/build_reference.py" ]; then
  say "references";  $PY -u "$REPO/tasks/$TASK/build_reference.py"
fi
say "QA";            $PY -u "$HERE/qa.py"
if [ -f "$REPO/tasks/$TASK/build_dataset.py" ]; then
  say "dataset";     $PY -u "$REPO/tasks/$TASK/build_dataset.py"
fi
say "report";        $PY -u "$HERE/report.py"
say "kinetics";      $PY -u "$HERE/kinetics.py"
if [ "$MUSCLES" = 1 ]; then
  say "muscles";     $PY -u "$HERE/muscles.py"
fi
if [ "$OPENSIM" = 1 ]; then
  say "OpenSim IK";  $OSIM_PY -u "$HERE/validate/opensim_ik.py"
  say "OpenSim ID";  $OSIM_PY -u "$HERE/opensim_id.py"
  say "compare";     $PY -u "$HERE/validate/opensim_compare.py" && $PY -u "$HERE/validate/kinetics_compare.py"
fi
say "done"
