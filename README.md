# Mocap to MSK

**Motion capture in, musculoskeletal motion and joint loads out.**

Mocap to MSK turns optical motion capture (OptiTrack `.tak`, C3D, TRC) into joint angles and joint loads of a MuJoCo musculoskeletal model: MyoFullBody by default, or MS-Human-700, MASS, Rajagopal 2016, Lai–Uhlrich 2022, Gait2354 and MyoLeg. It reads the raw recordings itself (no Motive or Nexus needed), calibrates each subject on a static standing trial, solves the inverse kinematics frame by frame, computes the joint loads by inverse dynamics, and grades every trial so you can see how far to trust it.

**[Website](https://fitz-yuan.github.io/mocap-to-myofullbody/)** · **[Documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/)** · **[Viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/)**

## What you get

- **Joint angles** per frame for the chosen model (`ik/<trial>.npz`: the model's `qpos`, each marker's residual, the frames that could not be solved).
- **Joint loads** per frame (`kin/<trial>.npz`): low back (L5/S1) moment, force and compression; shoulder and elbow moments; hip, knee and ankle moments; the ground reaction and its centre of pressure.
- **A QA report** (`ik/report.md`): every trial graded pass / warn / fail on marker fit, standing posture, marker gaps and joint flips, with the reasons.
- **Viewer clips** (`.elvw`) to look at any trial in the browser: bones, muscles, markers coloured by fit error, joint-angle and joint-load curves, force arrows.

## What it reads

| | Supported |
|---|---|
| Recordings | OptiTrack Motive `.tak` (read directly), C3D (Vicon, Qualisys, ...; with force plates and EMG), TRC |
| Marker sets | Plug-in-Gait (39 markers), ISB / Dumas (52 markers), or your own as one TOML file |
| Models | MyoFullBody (416 muscles, default), MS-Human-700 (700) and its legs (100), MASS (284), Rajagopal 2016 (80), Lai–Uhlrich 2022 (80), Gait2354 (54), MyoLeg (80) |

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img alt="Pipeline: raw recording → reader → markers → calibration on a static trial → IK, then reference and QA, inverse dynamics, and optionally OpenSim" src="docs/assets/pipeline-light.svg">
</picture>

| Stage | Script | What it does |
|---|---|---|
| 1. Read | `pipeline/read_raw.py` | `.tak`, C3D or TRC → marker trajectories in metres, Z up; force plates; analog channels |
| 2. Markers | `pipeline/prepare_markers.py` | labels mapped onto the marker set, mislabelled samples dropped, pelvis gaps refilled |
| 3. Calibration + IK | `pipeline/solve_ik.py` | where each marker sits on the body (anchored on the static trial), then joint angles per frame by bounded least squares |
| 4. QA | `pipeline/qa.py`, `pipeline/report.py` | the per-trial grades |
| 5. Joint loads | `pipeline/kinetics.py` | inverse dynamics on the model, at the subject's size and mass |
| optional | `run_all.sh --opensim`, `M2M_IK_DEVICE=gpu` | the OpenSim workflow on the same data, for comparison; the IK on the GPU (about 30× faster per trial) |

```
mocap2myo/    the library: readers (.tak, C3D, TRC), IK, inverse dynamics, model profiles and marker sets
pipeline/     the stages above, run_all.sh (all of them in order), validate/ (the checks)
datasets/     one TOML profile per dataset: file naming, units, subjects, marker set, task
tasks/        task-specific steps, e.g. box_lift/ (the box, carry cycles, the box's load on the hands)
models/       fetch.py (the models other than MyoFullBody, at pinned versions), check.py
docs/         the website, the documentation and the viewer
```

## Quick start

**1. Install** (Python 3.10+)

```bash
git clone https://github.com/Fitz-Yuan/mocap-to-myofullbody
cd mocap-to-myofullbody
pip install -e ".[tak,c3d]"        # add ",gpu" for the GPU IK
```

**2. Put the recordings in one folder**

```
~/data/mocap2myo/
  raw_tak/S01_Solid_NE.tak        trials
  raw_tak/static/S01_Anatomy.tak  one static standing trial per subject
  meta/anthropometrics.csv        Subject, Height (cm), Weight (kg)
```

These are the default names; other names and formats are set in a dataset profile (step 5).

**3. Run**

```bash
export M2M_ROOT=~/data/mocap2myo
IK_WORKERS=8 bash pipeline/run_all.sh            # all stages; IK_WORKERS = trials solved in parallel
```

Results land in the same folder: `ik/`, `kin/` and `ik/report.md`. Each stage also runs on its own (`python pipeline/<stage>.py --help`).

**4. Look at a trial**

```bash
python pipeline/export_viewer.py model out/                  # once per model
python pipeline/export_viewer.py clip S01_Solid_NE out/       # one trial
```

Open the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/) and drop the `.elvw` files on it. The files are read in the browser; nothing is uploaded.

**5. Your own data**

A dataset needs labelled markers (the labels the recording software assigned are enough), a known marker set, a static standing trial per subject, and each subject's height and mass. Describe it in a profile, `datasets/<name>.toml`, and run with `M2M_DATASET=datasets/<name>.toml`. For C3D files laid out as `raw/<subject>/<subject>_walk<nn>.c3d` and `raw/<subject>/<subject>_static.c3d`:

```toml
[paths]
root = "~/data/my_study"                  # outputs go here
raw_dir = "raw"                           # raw/<subject>/<subject>_walk<nn>.c3d
[input]
input = "c3d"                             # tak, c3d or trc
up = "auto"                               # from head vs foot markers
[naming]
raw_trials = "*/*_walk*.c3d"              # the trials: a glob under raw_dir
trial_regex = '(?P<subject>[^/]+)/(?P=subject)_walk\d+\.c3d'
trial_name = "{stem}"                     # S01_walk03
trial_glob = "*_walk*"
raw_static = "*/*_static.c3d"             # one static standing trial per subject
static_regex = '(?P<subject>[^/]+)/(?P=subject)_static\.c3d'
static_name = "{subject}_static"
[subjects]
anthro = "c3d"                            # height and mass from the C3D (or a CSV)
marker_set = "plug_in_gait"
[task]
task = "none"                             # body motion and joint loads only
[ik]
static = true                             # calibrate on the static trials
```

Three profiles ship as examples: the case study (`mmh_elevator`) and two public gait datasets (`ramp_vielemeyer2026`, Vicon; `moissenet2019`, Qualisys, ISB markers), each with its download link at the top. A new marker set is one TOML file in `mocap2myo/markersets/`.

## Common settings

Settings are environment variables, or keys in the dataset profile (the name without `M2M_`). All of them: [documentation, settings reference](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/#settings).

| Setting | Default | |
|---|---|---|
| `M2M_ROOT` | `~/data/mocap2myo` | the data folder |
| `M2M_DATASET` | (none) | a dataset profile |
| `M2M_MODEL` | `myofullbody` | the model (`python models/fetch.py` first for the others) |
| `M2M_MARKER_SET` | `plug_in_gait` | the marker set |
| `M2M_TASK` | `box_lift` | the task package, or `none` |
| `M2M_IK_DEVICE` | `cpu` | `gpu`: the IK on the GPU (MyoFullBody, Plug-in-Gait, 100 Hz for now) |
| `IK_WORKERS` | `12` | trials solved in parallel on the CPU |

## Validation at a glance

| Check | Result |
|---|---|
| Marker fit, 22 lifting trials | median 9.2 mm; OpenSim IK on the same markers 19.8 mm |
| Quiet-standing knee flexion | 0.1°; OpenSim 10.6° |
| Low-back moment vs OpenSim Inverse Dynamics | r = 0.984 |
| Total ground force vs force plates | 1.8 % of body weight |
| Knee flexion vs Vicon Plug-in-Gait, 36 public walking trials | r = 0.996 |

Method, caveats and every number: [documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/).

## License

- Code: Apache-2.0 (`LICENSE`).
- MyoFullBody (`mocap2myo/assets/myofullbody/`): Apache-2.0 by the MuscleMimic team; its arm and hand parts derive from MoBL-ARMS, **released for non-commercial use only**. See its `NOTICE`, and `MODIFICATIONS.md` for the changes made here.
- The other models are not included and keep their own licences (`mocap2myo/profiles/sources.toml`). The viewer uses three.js (MIT).
