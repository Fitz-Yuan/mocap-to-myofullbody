(own-data)=
# Your own data

A dataset needs labelled markers, a known marker set and one static standing trial per subject. Its settings (file kind and naming, units, subjects, marker set, task) go into a small profile, so nothing in the code changes.

1. **Labelled marker trajectories**: OptiTrack `.tak` takes or Motive's CSV export ({doc}`optitrack`), Qualisys QTM's TSV export, C3D or TRC files. The labels the recording software assigned are enough; unlabelled point clouds are not. Force plates (C3D) and any frame rate are taken as they are.
2. **A known marker set**: Plug-in-Gait, the 52-marker ISB / Dumas set and Motive's Baseline (37), Baseline (41) and Baseline + 13 Additional Markers (50) ship with the code; another set is one TOML file in `mocap2myo/markersets/` (see {ref}`models and marker sets <models>`).
3. **Enough markers per segment** to fix its orientation (ideally three).
4. **A static standing trial per subject**, to anchor the joint angles ({ref}`static-trial calibration <static>`).
5. **Subject height** (for scaling) and mass (for the joint loads): a CSV, or the C3D parameters.

**Folder layout without a profile** (the case study's names), under `M2M_ROOT` (default `~/data/mocap2myo`):

```
$M2M_ROOT/
  raw_tak/S<ss>_<Solid|Spring>_NE.tak     trial takes (M2M_RAW_TRIALS, M2M_TRIAL_GLOB)
  raw_tak/static/S<ss>_Anatomy.tak        static standing trials (M2M_RAW_STATIC, M2M_STATIC_NAME)
  meta/anthropometrics.csv                Subject, Height (cm), Weight (kg) (M2M_ANTHRO_CSV, M2M_ANTHRO_COLUMNS)
  opensim_setup/                          only for --opensim: Model.osim, markers.xml, a scale setup
  raw_daq/                                case study only: force-plate recordings
```

**Dataset profiles.** A profile is a TOML file in `datasets/` (settings only, no data), loaded with `M2M_DATASET=datasets/<name>.toml`; its keys are the settings below without `M2M_`, and an environment variable still overrides any of them. Three ship with the code: `mmh_elevator` (the case study; the same as no profile), `ramp_vielemeyer2026` (level and ramp walking, Vicon C3D, Plug-in-Gait, 200 Hz, three inclined force plates) and `moissenet2019` (gait at five speeds, Qualisys C3D, the ISB / Dumas set, height and mass in the C3D). Each says at its top where its data can be downloaded. From the ramp profile:

```
[paths]
root = "~/data/ramp_vielemeyer2026"     # outputs; relative paths below are under it
raw_dir = "raw"                         # Ref_<n>/<condition>/*.c3d, static_c3d/Ref_<n>/*.c3d
[input]
input = "c3d"
up = "auto"                             # from head vs foot markers (Vicon: z)
points_exclude = '(PEL|HED|TRX|[LR](FE|TI|FO|TO|CL|HU|RA|HN))[OALP]'   # Nexus segment points
[naming]
raw_trials = "Ref_*/*/*.c3d"
trial_regex = 'Ref_\d+/(?P<cond>(level|ramp_\d+)_(up|down))/(?P<subject>Ref\d+)_\w*?shoes(?P<k>\d+)\.c3d'
trial_name = "{subject}_{cond}_{k}"     # Ref_1/level_up/Ref01_sss_le_shoes04.c3d -> Ref01_level_up_04
trial_glob = "Ref*"
raw_static = "static_c3d/*/*.c3d"
static_regex = 'static_c3d/Ref_\d+/(?P<subject>Ref\d+)_(sss_)?le_(sss_)?shoes(_static|01)\.c3d'
static_name = "{subject}_static"
[subjects]
anthro_csv = "anthropometrics.csv"      # Subject, Height (cm), Weight (kg): the article's Table 1
marker_set = "plug_in_gait"
[task]
task = "none"
```

**Tasks.** Everything about the lifting task (box template and pose, shelf, platform and floor, carry cycles, the reference, the dataset, the box's load on the hands and the viewer scene) is in `tasks/box_lift/`; another task is another package in `tasks/` with the same hooks (`tasks/box_lift/__init__.py`), chosen with `M2M_TASK`. With `M2M_TASK=none` the pipeline stops at the body motion: IK, QA, report and joint loads with the ground as the only external load; the viewer then reads the IK output (whole trials).

**Limitations to keep in mind.**

- Models are scaled uniformly by height, not per segment.
- Skin markers move on the soft tissue by 1–3 cm in large motions; residuals of about 1 cm are close to that floor.
- MyoFullBody has no knee axial rotation and no neck joint (the IK can add one, `M2M_NECK=1`, not validated); deep squats reach its 120° joint limits.
- The lifting task is written for MyoFullBody's bodies; other models run the case study with `M2M_TASK=none`.
