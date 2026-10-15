#!/bin/bash
# The whole pipeline, from raw OptiTrack takes to motion clips (the case study's settings).
#   M2M_ROOT=~/data/mocap2myo bash pipeline/run_all.sh
# Inputs under $M2M_ROOT: raw_tak/<trial>.tak, raw_tak/static/<subject>_Anatomy.tak,
# meta/anthropometrics.csv (Subject, Height cm, Weight kg); names and folders: see
# pipeline/config.py. Outputs: tak_cache/, markers/, ik/ (IK, QA, report), ref/, and
# data/<M2M_DATA> in the repository. With M2M_TASK=none only the body motion (no
# reference, no dataset). CPU only. IK_WORKERS parallel processes (~1-2 GB each).
# --opensim: also run the OpenSim workflow on the same markers (Scale + Marker Placer +
# IK, then Inverse Dynamics with the same loads) for the side-by-side comparison; needs
# the opensim package (conda install -c opensim-org opensim), in OSIM_PY if that is
# another Python (default: the same as PY).
set -eu
OPENSIM=0
for arg in "$@"; do
  case "$arg" in
    --opensim) OPENSIM=1 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
PY=${PY:-python}
OSIM_PY=${OSIM_PY:-$PY}
export M2M_ROOT=${M2M_ROOT:-$HOME/data/mocap2myo}
export OMP_NUM_THREADS=1 IK_WORKERS=${IK_WORKERS:-12} M2M_SOURCE=tak
export M2M_STATIC=${M2M_STATIC:-1}            # calibrate on the static standing trials
export M2M_STATIC_MARKERS=${M2M_STATIC_MARKERS:-legs}
TASK=${M2M_TASK:-box_lift}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "read takes";    $PY -u "$HERE/cache_takes.py"       # needs olefile
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
if [ "$OPENSIM" = 1 ]; then
  say "OpenSim IK";  $OSIM_PY -u "$HERE/validate/opensim_ik.py"
  say "OpenSim ID";  $OSIM_PY -u "$HERE/opensim_id.py"
  say "compare";     $PY -u "$HERE/validate/opensim_compare.py" && $PY -u "$HERE/validate/kinetics_compare.py"
fi
say "done"
