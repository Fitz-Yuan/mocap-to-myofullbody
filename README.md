# Mocap to MSK

**Motion capture in, musculoskeletal motion and joint loads out.**

Mocap to MSK turns optical motion capture (OptiTrack `.tak`, C3D, TRC, AddBiomechanics `.b3d`) into joint angles and joint loads of a MuJoCo musculoskeletal model: MyoFullBody by default, or MS-Human-700, MASS, Rajagopal 2016, Lai–Uhlrich 2022, Gait2354 and MyoLeg. It reads the raw recordings itself (no Motive or Nexus needed), calibrates each subject, solves the inverse kinematics frame by frame, computes the joint loads by inverse dynamics, and grades every trial so you can see how far to trust it.

**[Website](https://fitz-yuan.github.io/mocap-to-myofullbody/)** · **[Documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/)** · **[Viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/)**

## What you get

- **Joint angles** per frame (`ik/<trial>.npz`), with each marker's fit error.
- **Joint loads** per frame (`kin/<trial>.npz`): low back (L5/S1), shoulders, elbows, hips, knees, ankles, and the ground reaction.
- **A report** (`ik/report.md`) that grades every trial PASS / WARN / FAIL and says why.
- **Viewer clips** (`.elvw`) to watch any trial in the browser, and **gap-filled C3D files** for other software (OpenSim, Visual3D, ...).

## Step by step

### 1. Install

You need Python 3.10 or newer and git.

```bash
git clone https://github.com/Fitz-Yuan/mocap-to-myofullbody
cd mocap-to-myofullbody
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[c3d,tak]"
mocap2myo --help                     # the command is installed
```

`c3d` reads C3D files, `tak` reads OptiTrack takes. Add `gpu` for the GPU IK (NVIDIA GPU, CUDA) and `b3d` for AddBiomechanics files.

### 2. Try the demo

```bash
mocap2myo demo
```

This downloads one public walking trial (about 0.4 MB, Moissenet & Schreiber 2019, CC BY 4.0) into `mocap2myo_demo/` and runs every step on it, in about a minute. Then open the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/) and drop `mocap2myo_demo/viewer/2014001_C3_03.elvw` on it: you see the model walk, its markers against the measured ones, and the joint-angle and joint-load curves. The file is read in your browser; nothing is uploaded.

### 3. Prepare your recordings

Put the files of one study in one folder. You need:

- **Labelled markers.** The labels your capture software assigned are enough. Gaps are fine.
- **A known marker set.** Plug-in-Gait and ISB / Dumas are built in. Another set is one TOML file in `mocap2myo/markersets/`.
- **Each subject's height and mass**, in the C3D files or in a CSV (subject, height in cm, mass in kg).
- **A static standing trial per subject** (recommended). Without one, the calibration uses quiet standing from the trials.

For example:

```
my_study/
  raw/S01/S01_walk01.c3d
  raw/S01/S01_walk02.c3d
  raw/S01/S01_static.c3d
  raw/S02/...
```

### 4. Describe your dataset

```bash
mocap2myo init my_study.toml
```

It asks where the files are, finds their format, guesses the marker set from the labels, and asks how subjects and static trials are named. It writes a dataset profile, `my_study.toml`. You can also copy one of the examples in `datasets/` and edit it: each one is commented, and each has its download link at the top.

### 5. Check before running

```bash
mocap2myo check --dataset my_study.toml
```

It lists what the pipeline would read: the trials and static trials it finds, every subject's height and mass, the first trial's frame rate, units and force plates, and how many of the marker set's markers the labels match. Lines marked `x` must be fixed before running, and each says what to set; lines marked `!` are worth a look.

### 6. Run

```bash
mocap2myo run --dataset my_study.toml
```

It runs every step in order: read the files, prepare the markers, calibrate and solve the IK, grade the trials, compute the joint loads. Useful options:

| Option | What it does |
|---|---|
| `--workers 8` | trials solved in parallel on the CPU (default 12; about 1–2 GB of memory each) |
| `--gpu` | the IK on the GPU, all trials at once (tens of times faster) |
| `--muscles` | also each muscle's force, by static optimisation |
| `--from kinetics` | start again at a step (`read`, `markers`, `ik`, `qa`, `report`, `kinetics`, ...) |
| `S01_walk01 S01_walk02` | only these trials |

### 7. Look at the results

All results go into the folder the profile names as `root`:

| Where | What |
|---|---|
| `ik/report.md` | start here: one line per trial, PASS / WARN / FAIL and the reasons |
| `ik/<trial>.npz` | joint angles (`qpos`, `joint_names`), marker errors (`marker_err`) |
| `kin/<trial>.npz` | joint moments and forces, the ground reaction |

To watch a trial, export it and drop the files on the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/):

```bash
M2M_DATASET=my_study.toml python pipeline/export_viewer.py model viewer/              # once per model
M2M_DATASET=my_study.toml python pipeline/export_viewer.py clip S01_walk01 viewer/    # one trial
```

To use the markers in other software, write them with their gaps filled (from neighbouring markers where they are seen, else from the model):

```bash
M2M_DATASET=my_study.toml python pipeline/export_c3d.py filled_c3d/
```

## When something goes wrong

| You see | What to do |
|---|---|
| `x no trial files match ...` | the file pattern in the profile (`raw_trials`) does not find your files: compare it with `mocap2myo check`'s folder and fix the pattern |
| `marker set ...: 12 of 39 markers` | your labels differ from the marker set's: choose another set, or add your names as an alias set (`[aliases]` in the marker set's TOML) |
| `x pelvis markers missing` | the IK needs at least 3 of the 4 pelvis markers |
| `no height / mass for S03` | add the subject to the CSV, with the name the trials use |
| a trial is FAIL in the report | the report says which check failed: marker fit, a limb lost by the IK, standing posture, marker gaps, joint flips. The [documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/) explains each check |

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/pipeline-dark.svg">
  <img alt="Pipeline: raw recording → reader → markers → calibration on a static trial → IK, then reference and QA, inverse dynamics, and optionally OpenSim" src="docs/assets/pipeline-light.svg">
</picture>

| Step | Script | What it does |
|---|---|---|
| Read | `pipeline/read_raw.py` | `.tak`, C3D, TRC or `.b3d` → markers in metres, Z up; force plates; analog channels |
| Markers | `pipeline/prepare_markers.py` | labels onto the marker set, mislabelled samples dropped, pelvis gaps refilled |
| Calibration + IK | `pipeline/solve_ik.py` | where each marker sits on the body (static trial), then the pose per frame; short gaps (≤ 200 ms) by spline, longer ones left to the model |
| QA | `pipeline/qa.py`, `pipeline/report.py` | the grades |
| Joint loads | `pipeline/kinetics.py` | inverse dynamics at the subject's size and mass, with the force plates where the feet are on them |
| Muscles (optional) | `pipeline/muscles.py` | muscle forces by static optimisation |

```
mocap2myo/    the library: readers, IK, inverse dynamics, the command line, model profiles, marker sets
pipeline/     the steps above, validate/ (the comparisons with other tools and references)
datasets/     example dataset profiles (case study, two public gait datasets, AnDy, AddBiomechanics)
tasks/        task-specific steps, e.g. box_lift/ (the box, carry cycles, the box's load on the hands)
models/       fetch.py: the models other than MyoFullBody, at pinned versions
tests/        the tests (pytest)
docs/         the website, the documentation and the viewer
```

## Common settings

Settings are keys in the dataset profile, or environment variables with `M2M_` in front. All of them: [documentation, settings reference](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/#settings).

| Setting | Default | |
|---|---|---|
| `M2M_MODEL` | `myofullbody` | the model (`mocap2myo models` fetches the others) |
| `M2M_MARKER_SET` | `plug_in_gait` | the marker set |
| `M2M_TASK` | `box_lift` without a profile | the task package, or `none` (body motion and joint loads only) |
| `M2M_LOWPASS_HZ` | `6` | marker low-pass filter (Hz) |
| `M2M_MAX_GAP_S` | `0.2` | longest gap filled by spline (s) |
| `M2M_IK_DEVICE` | `cpu` | `gpu`: the IK on the GPU |

## Validation at a glance

| Check | Result |
|---|---|
| Marker fit, 22 lifting trials | median 9.2 mm; OpenSim IK on the same markers 19.8 mm |
| Quiet-standing knee flexion | 0.1°; OpenSim 10.6° |
| Low-back moment vs OpenSim Inverse Dynamics | r = 0.984 |
| Total ground force vs force plates | 1.8 % of body weight |
| Knee flexion vs Vicon Plug-in-Gait, 36 public walking trials | r = 0.996 |
| Knee flexion vs an Xsens suit worn at the same time, 13 people | r = 0.99 |

Method, caveats and every number: [documentation](https://fitz-yuan.github.io/mocap-to-myofullbody/documentation/).

## License

- Code: Apache-2.0 (`LICENSE`).
- MyoFullBody (`mocap2myo/assets/myofullbody/`): Apache-2.0 by the MuscleMimic team; its arm and hand parts derive from MoBL-ARMS, **released for non-commercial use only**. See its `NOTICE`, and `MODIFICATIONS.md` for the changes made here.
- The other models are not included and keep their own licences (`mocap2myo/profiles/sources.toml`). The viewer uses three.js (MIT).
