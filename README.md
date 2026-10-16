# Mocap to MSK

**Motion capture in, musculoskeletal motion and joint loads out.**

Mocap to MSK turns optical motion capture (OptiTrack `.tak`, C3D, TRC) into joint angles and joint loads of a MuJoCo musculoskeletal model: MyoFullBody by default, or MS-Human-700, MASS, Rajagopal 2016, Lai–Uhlrich 2022, Gait2354 and MyoLeg.

**[Website](https://fitz-yuan.github.io/mocap-to-myofullbody/)** · **[Documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/)** · **[Viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/)**

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img alt="Pipeline: raw recording → reader → markers → calibration on a static trial → IK, then reference and QA, inverse dynamics, and optionally OpenSim" src="docs/assets/pipeline-light.svg">
</picture>

| Stage | Script | What it does |
|---|---|---|
| 1. Read | `pipeline/read_raw.py` | `.tak` (no Motive needed), C3D or TRC → marker trajectories, force plates, analog channels |
| 2. Markers | `pipeline/prepare_markers.py` | labels mapped onto the marker set, mislabels dropped, gaps refilled |
| 3. Calibration + IK | `pipeline/solve_ik.py` | where each marker sits on the body (anchored on a static standing trial), then joint angles per frame |
| 4. QA | `pipeline/qa.py`, `pipeline/report.py` | a pass / warn / fail grade per trial |
| 5. Joint loads | `pipeline/kinetics.py` | net joint moments and forces by inverse dynamics |
| optional | `run_all.sh --opensim`, `M2M_IK_DEVICE=gpu` | the OpenSim workflow on the same data for comparison; the IK on the GPU |

The code is in four parts:

```
mocap2myo/    the library: readers, IK, inverse dynamics, model profiles and marker sets (TOML)
pipeline/     the stages above, run_all.sh, and the validation scripts (validate/)
datasets/     one TOML profile per dataset: file naming, units, subjects, marker set, task
tasks/        task-specific steps, e.g. box_lift/ (the box, carry cycles, hand loads)
```

## Quick start

**1. Install** (Python 3.10+)

```bash
git clone https://github.com/Fitz-Yuan/mocap-to-myofullbody
cd mocap-to-myofullbody
pip install -e ".[tak,c3d]"
```

**2. Put the recordings in one folder**

```
~/data/mocap2myo/
  raw_tak/S01_Solid_NE.tak        trials
  raw_tak/static/S01_Anatomy.tak  one static standing trial per subject
  meta/anthropometrics.csv        Subject, Height (cm), Weight (kg)
```

These are the case study's names; for other names, C3D or TRC files, write a small profile in `datasets/` (see "Using your own data" in the [documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/#own-data)).

**3. Run**

```bash
export M2M_ROOT=~/data/mocap2myo
IK_WORKERS=8 bash pipeline/run_all.sh
```

Results land in the same folder: `ik/` (joint angles per frame), `kin/` (joint loads) and `ik/report.md` (the QA grades).

**4. Look at a trial**

```bash
python pipeline/export_viewer.py model out/                  # once
python pipeline/export_viewer.py clip S01_Solid_NE out/       # one trial
```

Open the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/) and drop the `.elvw` files on it. Nothing is uploaded.

**Other datasets and models**

Each profile in `datasets/` says at its top where its data can be downloaded and where to put it.

```bash
M2M_DATASET=datasets/moissenet2019.toml bash pipeline/run_all.sh                         # a public C3D dataset
python models/fetch.py                                                                 # the other models, once
M2M_DATASET=datasets/moissenet2019.toml M2M_MODEL=ms_human_700 bash pipeline/run_all.sh  # on another model
```

Every setting, the validation (against OpenSim, Xsens, force plates and Vicon Plug-in-Gait) and the known limits are in the [documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/).

## License

- Code: Apache-2.0 (`LICENSE`).
- MyoFullBody (`mocap2myo/assets/myofullbody/`): Apache-2.0 by the MuscleMimic team; its arm and hand parts derive from MoBL-ARMS, **released for non-commercial use only**. See its `NOTICE`, and `MODIFICATIONS.md` for the changes made here.
- The other models are not included and keep their own licences (`mocap2myo/profiles/sources.toml`). The viewer uses three.js (MIT).
