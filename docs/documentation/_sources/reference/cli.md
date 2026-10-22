(run)=
# Command line

## `mocap2myo`

The command installed by `pip install -e .`:

| Command | What it does |
|---|---|
| `mocap2myo init [PROFILE.toml]` | writes a dataset profile from a few questions: the folder, the file kind (found), the marker set (guessed from the labels), how subjects and static trials are named, where heights and masses come from |
| `mocap2myo check --dataset P` | what the pipeline would read, before it runs (`pipeline/check.py`) |
| `mocap2myo run --dataset P [trials]` | the steps in order: read, markers, IK, the task's reference, QA, the task's dataset, report, joint loads |
| `mocap2myo demo [DIR]` | one public walking trial end to end (about 0.4 MB) |
| `mocap2myo models [PROFILE ...]` | fetch and build the models other than MyoFullBody (`models/fetch.py`) |

Options of `run`: `--from STEP` (start there), `--only STEP,STEP`, `--gpu`, `--workers N`, `--muscles`, `--opensim` (with `--osim-python` if `opensim` is in another Python), `--root DIR`. Every option has a `M2M_*` setting behind it ({doc}`settings`).

## Scripts

Every step is a script that runs on its own, with the same settings (`--help` prints its documentation):


```
# install (Python 3.10+)
pip install -e ".[c3d,tak]"

# the stages as a shell script (what mocap2myo run does)
M2M_ROOT=~/data/mocap2myo IK_WORKERS=22 bash pipeline/run_all.sh
M2M_IK_DEVICE=gpu bash pipeline/run_all.sh        # the IK on the GPU, all trials in one batch

# check the take reader against Motive's TRC exports (folder of .tak / .trc pairs)
python pipeline/validate/tak_reader.py DIR

# exports: gap-filled C3D for other software
python pipeline/export_c3d.py OUT_DIR

# AddBiomechanics: read .b3d, compare with its own IK and ID (needs nimblephysics)
M2M_DATASET=datasets/addbiomechanics_falisse2017.toml python pipeline/validate/addbio_compare.py IK_DIR KIN_DIR

# validation: leave-one-marker-out, Xsens, static-trial diagnosis
python pipeline/validate/lomo.py
python pipeline/validate/xsens.py $M2M_ROOT/ik
python pipeline/validate/static_pose.py

# viewer files: the model once, then clips (optionally with a ghost from another run)
python pipeline/export_viewer.py model OUT_DIR
python pipeline/export_viewer.py clip S01_Solid_NE OUT_DIR --clip 1 --ghost-ref-dir OTHER_REF_DIR

# the same plus the OpenSim workflow (IK and Inverse Dynamics) for the comparison
M2M_ROOT=~/data/mocap2myo bash pipeline/run_all.sh --opensim    # opensim from conda

# inverse dynamics alone (after the IK and the reference), and its checks
python pipeline/kinetics.py
python tasks/box_lift/force_plates.py           # case study: the lab's force plates
python pipeline/validate/kinetics_compare.py    # vs OpenSim ID (after --opensim)

# another model, another dataset: fetch the models once, then any stage with M2M_MODEL
M2M_MODELS_DIR=~/data/mocap2myo/models python models/fetch.py
python models/check.py mass                      # landmarks, kinetics axes, symmetry
M2M_DATASET=datasets/moissenet2019.toml M2M_MODEL=ms_human_700 bash pipeline/run_all.sh

# public C3D with the lab software's model outputs: compare with Vicon Plug-in-Gait
M2M_DATASET=datasets/ramp_vielemeyer2026.toml python pipeline/validate/pig_compare.py

# a de-identified demo clip for a public page (no trial name, clip position, subject scale or QA grades)
python pipeline/export_viewer.py clip S01_Solid_NE OUT_DIR --clip 1 --public "Demo 1"
```

Data layout under `$M2M_ROOT`: `raw_tak/` (trial takes), `raw_tak/static/` (static trials), `meta/anthropometrics.csv`, and for `--opensim` `opensim_setup/` (see {ref}`OpenSim setup files <opensim-setup>`), or what a dataset profile says (`M2M_DATASET`, `datasets/*.toml`); written by the pipeline: `tak_cache/` (`raw_cache/` for C3D / TRC), `markers/`, `ik/` (with `qa.json`, `report.json` and `report.md`; `lomo.json` when `validate/lomo.py` is run), `ref/`, `kin/`, and the motion dataset in `data/<M2M_DATA>/` of the repository; with `--opensim` also `opensim_ik/`, `opensim_id/` and `compare_osim/`. Every folder, file-naming rule and `M2M_*` setting is in `pipeline/config.py`. The only data from the case study in this repository is one de-identified demo clip that ships with the viewer; the viewer's other demo is one public walking trial (Moissenet & Schreiber 2019, CC BY 4.0).
