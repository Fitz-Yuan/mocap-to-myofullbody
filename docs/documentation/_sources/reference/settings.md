(settings)=
# Settings

Every `M2M_*` setting and `IK_WORKERS` is read in `pipeline/config.py`; `MYOFULLBODY_MODEL` and `M2M_MODELS_DIR` belong to the library, `PY` / `OSIM_PY` to `run_all.sh`. A setting comes from its environment variable, else from the dataset profile (key = the name without `M2M_`, lower case), else the default. Settings with a fixed set of values, and unknown profile keys, stop the run. `run_all.sh` turns the static-trial calibration on (`M2M_STATIC=1`); a stage run on its own does not.

```{table}
:class: refs

| Variable | Default | Meaning |
|---|---|---|
| `M2M_DATASET` | (none) | dataset profile (TOML), e.g. `datasets/ramp_vielemeyer2026.toml` |
| `M2M_ROOT` | `~/data/mocap2myo` | data folder (a relative path in a profile is relative to it) |
| `M2M_INPUT`, `M2M_RAW_DIR`, `M2M_CACHE_DIR` | `tak`, `raw_tak/`, `tak_cache/` | raw file kind (`tak`, `c3d`, `trc`, `b3d`, `csv`: Motive, `tsv`: Qualisys), their folder, `read_raw.py`'s output (`raw_cache/` for C3D / TRC) |
| `M2M_RAW_TRIALS`, `M2M_TRIAL_REGEX`, `M2M_TRIAL_NAME` | `S*_NE.tak`, (none), `{stem}` | trial files (glob under `M2M_RAW_DIR`, `**` allowed); a regex on the path relative to it whose named groups, with `{stem}` (the file name), fill the trial name |
| `M2M_RAW_STATIC`, `M2M_STATIC_REGEX` | `static/*_Anatomy.tak`, `static/(?P<subject>.+)_Anatomy\.tak` | static trial files, and the regex whose group `subject` names the subject |
| `M2M_UP`, `M2M_UNITS`, `M2M_RATE` | `y` (`.tak`) / `auto`, file's, `native` | the files' up axis (`auto`: from head vs foot markers; `x`/`y`/`z`, optionally `-`), length unit, frame rate (a number: keep every k-th frame) |
| `M2M_POINTS_EXCLUDE`, `M2M_ANALOG` | (none), `.*` | C3D: point labels that are not markers (besides the POINT type groups), analog channels to keep (EMG) |
| `M2M_FP_SIGN`, `M2M_FP_MIN_FZ` | `auto`, `20` | C3D force plates: sign of ezc3d's force (`auto`: the vertical force on a loaded plate points up), N below which there is no centre of pressure |
| `M2M_MODEL` | `myofullbody` | the model: a profile in `mocap2myo/profiles/` (see Models and marker sets) |
| `M2M_MARKER_SET`, `M2M_MARKER_ALIASES` | `plug_in_gait`, `auto` | the marker set the data uses (`mocap2myo/markersets/<name>.toml`); which of its alias label sets a recording may use (`auto`: the best-matching one, if it beats the set's own labels; `none`; or names) |
| `M2M_MODELS_DIR` | `~/data/mocap2myo/models` | where `models/fetch.py` puts the models that are not in the repository |
| `M2M_ANTHRO`, `M2M_ANTHRO_C3D_HEIGHT`, `M2M_ANTHRO_C3D_MASS` | `csv`, `SUBJECT:height`, `SUBJECT:weight` | where heights / masses come from: the CSV, or C3D parameters (`GROUP:NAME`), collected into `<cache>/anthropometrics.csv` |
| `M2M_NPZ_DIR`, `M2M_IK_DIR`, `M2M_REF_DIR` | `markers/`, `ik/`, `ref/` | stage folders under `M2M_ROOT` |
| `M2M_DATA`, `M2M_DATA_STEM` | `box_lift`, `box_lift` | dataset folder in `data/`, file stem `motion_<stem>[_ok]_100hz.npz` |
| `M2M_TASK` | `box_lift` | task package in `tasks/`; `none` = body motion only |
| `M2M_TRIAL_GLOB`, `M2M_SUBJECT_SEP`, `M2M_STATIC_NAME` | `S*_NE`, `_`, `{subject}_Anatomy` | trial names (how every stage finds the trials), subject = trial name up to the first separator, static trial name |
| `M2M_ANTHRO_CSV`, `M2M_ANTHRO_COLUMNS`, `M2M_ANTHRO_SUBJECT` | `meta/anthropometrics.csv`, `Subject,Height,Weight`, `S{:02d}` | subject heights (cm) and masses (kg); how a numeric subject cell maps to the subject in the file names (a cell like `S01` is used as it is) |
| `M2M_STATIC`, `M2M_STATIC_MARKERS`, `M2M_STATIC_SHARE`, `M2M_STATIC_W` | `0` (`run_all.sh`: `1`), `legs`, `0.5`, `1.0` | static-trial calibration: on/off, which markers it places (`legs`, `legs_only`, `all`), its share of the offset update, posture weight |
| `M2M_OUTLIER`, `M2M_ROBUST`, `M2M_LOWPASS_HZ`, `M2M_MTP_PRIOR`, `M2M_ROOT_INIT`, `M2M_PELVIS_GAP`, `M2M_ANCHOR_CENTROID`, `M2M_FOOT_CAP`, `M2M_FLAT_W`, `M2M_IK_ANKLE_SMOOTH` | see `config.py` | IK settings (`M2M_ROOT_INIT`: `pelvis` or `prev`; `M2M_PELVIS_GAP`: `solve` or `carry`; `M2M_LOWPASS_HZ` 6, 0 = unfiltered markers) |
| `M2M_NECK` | `0` | `1`: a 3-hinge neck in the IK model so the head follows its markers (MyoFullBody has none); the IK output keeps the model's qpos layout and adds `neck_qpos`; not validated |
| `M2M_IK_FRAMES` | (all) | `a:b`: IK on a frame range only |
| `M2M_IK_DEVICE` | `cpu` | `gpu`: the IK on the GPU (`mocap2myo/ik_gpu.py`, `pip install -e ".[gpu]"`), all trials in one batch |
| `M2M_SEX_COLUMN` | (none) | a column of the anthropometrics CSV with each subject's sex (`m` / `f`), for the segment-mass table |
| `M2M_GAP_FILL`, `M2M_MAX_GAP_S` | `spline`, `0.2` | short marker gaps: cubic spline (or `linear`) up to this many seconds ({doc}`../pipeline/gap-filling`) |
| `M2M_IK_SMOOTH`, `M2M_IK_ANKLE_SMOOTH` | `0.05`, `0.08` | frame-to-frame smoothing per frame at 100 Hz: most joints, and the ankle and subtalar joints |
| `M2M_IK_RECOVER` | `0` | `1`: a frame leaving 2+ markers more than 6 cm off is solved again from scratch (CPU; {ref}`recovery`) |
| `M2M_SCALE`, `M2M_SCALE_UNIFORM` | `uniform`, (none) | `segments`: each segment scaled to the subject from the static trial (one model per subject); groups kept uniform |
| `M2M_CONF_DIR`, `M2M_CONF_STEP_S` | `ik/confidence/`, `0.04` | per-joint confidence of the IK (`pipeline/confidence.py`) and its time step |
| `M2M_KIN_FILTER_PLATES` | `1` | force plates low-passed like the motion (Kristianslund et al. 2012) |
| `M2M_KIN_SPLIT` | `lever` | double support off the plates: `lever` rule, or `vpp` (contact detection and the virtual pivot point) |
| `M2M_KIN_BSIP` | `deleva` | segment masses: `deleva` (de Leva 1996), `dempster` (Dempster / Winter), `model` |
| `M2M_MUSCLE_DIR`, `M2M_MUSCLE_JOINTS`, `M2M_MUSCLE_RESERVE` | `kin/muscles/`, `auto`, `1.0` | muscle forces: output, the matched joints, the reserve torques' scale (N m) |
| `M2M_MUSCLE_WEIGHT`, `M2M_MUSCLE_FV`, `M2M_MUSCLE_PASSIVE` | `none`, `1`, `1` | the cost's weighting (`volume`), force–velocity, passive forces |
| `M2M_MUSCLE_STRENGTH`, `M2M_MUSCLE_SCALE` | `1.0`, `none` | a factor on every maximum isometric force; `mass`: scaled with body mass / model mass |
| `M2M_EMG_MAP` | (none) | `validate/emg_compare.py`: EMG channel → model muscles |
| `M2M_LOMO_EVERY_S` | `1.0` | `validate/lomo.py`: one frame per this many seconds |
| `M2M_KIN_DIR`, `M2M_KIN_LOWPASS_HZ` | `kin/`, `6` | inverse-dynamics output; low-pass (Hz) before differentiating positions (also the OpenSim ID's) |
| `M2M_CARRIED_COLUMN`, `M2M_CARRIED_BODY` | `Wgt+K5`, `torso` | a CSV column with the mass while recorded (equipment worn, here a metabolic system); the extra mass is lumped at that body (a body of the model, else that canonical segment's body); ignored when the CSV has no such column |
| `M2M_XSENS_TRIAL`, `M2M_XSENS_C3D` | `S11_Spring_NE`, `xsens_s11/<trial>.c3d` | the Xsens check (`validate/xsens.py`, `opensim_compare.py`) |
| `M2M_OSIM_SETUP_DIR`, `M2M_OSIM_SCALE_SETUP` | `opensim_setup/`, `Sub_09__Scalingsetup_09.xml` | the OpenSim inputs (below) |
| `IK_WORKERS` | 12 | parallel processes of every parallel stage (~1-2 GB each for the IK) |
| `MYOFULLBODY_MODEL` | `kneefix` | `original`: MyoFullBody with the left knee coupling as released ({doc}`license`) |
| `PY`, `OSIM_PY` | `python`, `$PY` | `run_all.sh`: the Python for the stages, and the one with `opensim` |
```
