#!/bin/bash
# The whole pipeline for the MMH Elevator data, from raw OptiTrack takes to motion clips.
#   ELEV_ROOT=~/data/mmh_elevator bash pipeline/run_all.sh
# Inputs under $ELEV_ROOT: raw_tak/S<ss>_<floor>_NE.tak, raw_tak/static/S<ss>_Anatomy.tak,
# meta/anthropometrics.csv (Subject, Height cm, Weight kg). Outputs: tak_cache/, npz_tak/,
# ik_tak/ (IK, QA, report), ref_tak/, and data/<ELEV_DATA> in the repository.
# CPU only. IK_WORKERS parallel processes (~1-2 GB each).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
export ELEV_ROOT=${ELEV_ROOT:-$HOME/data/mmh_elevator}
export OMP_NUM_THREADS=1 IK_WORKERS=${IK_WORKERS:-12} ELEV_SOURCE=tak
export ELEV_STATIC=${ELEV_STATIC:-1}            # calibrate on the static standing trials
export ELEV_STATIC_MARKERS=${ELEV_STATIC_MARKERS:-legs}
export ELEV_NPZ_DIR=$ELEV_ROOT/npz_tak ELEV_IK_OUT=$ELEV_ROOT/ik_tak ELEV_IK_DIR=$ELEV_ROOT/ik_tak
export ELEV_REF_DIR=$ELEV_ROOT/ref_tak ELEV_DATA=${ELEV_DATA:-lift_elev_tak}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "read takes";  $PY -u "$HERE/elev_tak_cache.py"      # needs olefile
say "convert";     $PY -u "$HERE/elev_convert.py"
say "IK";          $PY -u "$HERE/elev_ik.py"
say "references";  $PY -u "$HERE/elev_build_reference.py"
say "QA";          $PY -u "$HERE/elev_qa.py"
say "dataset";     $PY -u "$HERE/elev_build_dataset.py"
say "report";      $PY -u "$HERE/elev_report.py"
say "done"
