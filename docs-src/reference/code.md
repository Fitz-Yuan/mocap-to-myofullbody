(code)=
# Source layout

```{table}
:class: refs

| File | What |
|---|---|
| `mocap2myo/tak.py` | Reads .tak files without Motive; exports TRC; compares against a TRC export |
| `mocap2myo/readers.py` | Reads one C3D, TRC, .tak or AddBiomechanics .b3d trial: markers (metres, Z-up), force plates as the ground reaction per marker frame, analog channels, the lab software's model outputs apart |
| `pipeline/config.py`, `datasets/*.toml` | Folders, file naming (trials, subjects, static trials, anthropometrics) and every setting / environment variable; a dataset's settings as a profile |
| `pipeline/read_raw.py` | Raw recordings (.tak, C3D, TRC, .b3d; trials and static trials) to the cache the cleaning step reads |
| `pipeline/prepare_markers.py` | Marker conditioning: label maps, head by position, mislabel rejection, pelvis refill (the task adds box and scene) |
| `mocap2myo/markerset.py`, `mocap2myo/markersets/*.toml` | Marker sets as data (Plug-in-Gait, ISB / Dumas): segment, landmark, offset and weight per marker, roles, label aliases |
| `mocap2myo/profile.py`, `mocap2myo/profiles/*.toml`, `mocap2myo/landmarks.py` | Model profiles (file, root, facing, IK joints, canonical segments, kinetics joints, feet and hands) and the anatomical landmarks computed from a model's bones |
| `models/`: `fetch.py`, `mass_to_mjcf.py`, `osim_to_mjcf.py`, `check.py` | The models that are not in the repository, fetched at pinned commits and built (MASS, MyoConverter); landmarks and counts of every profile |
| `mocap2myo/ik.py` | The IK solver: marker model, short-gap filling, per-frame least squares, tracking-loss recovery, the lost-limb check, subject calibration (with the static-trial option) |
| `mocap2myo/ik_gpu.py` | The same IK on the GPU (Warp kernels, chunked in time, all trials in one batch) |
| `mocap2myo/scaling.py` | Segment-by-segment scaling from the static trial (`M2M_SCALE=segments`) |
| `mocap2myo/cli.py` | The `mocap2myo` command: init, check, run, demo, models |
| `pipeline/check.py` | The pre-flight check of a dataset |
| `pipeline/confidence.py` | Per-joint confidence of the IK (a ±1 SD band in the viewer) |
| `pipeline/solve_ik.py` | Calibration and IK for all trials in parallel |
| `pipeline/qa.py` | QA of an IK run (with the task's checks) |
| `tasks/box_lift/`: `box_fit.py`, `build_reference.py`, `checks.py`, `build_dataset.py`, `motion_io.py`, `viewer_scene.py` | The lifting task: box and scene, reference, its QA checks, carry-cycle clips, motion-dataset writer, viewer objects |
| `pipeline/report.py`, `pipeline/validate/lomo.py`, `xsens.py`, `static_pose.py`, `pig_compare.py`, `addbio_compare.py` | Validation |
| `mocap2myo/muscles.py`, `pipeline/muscles.py`, `mocap2myo/emg.py`, `pipeline/validate/emg_compare.py`, `tasks/box_lift/emg.py` | Muscle forces by static optimisation, and the comparison with EMG |
| `pipeline/validate/opensim_ik.py`, `opensim_compare.py` | OpenSim baseline on the same markers, and the side-by-side scores |
| `mocap2myo/kinetics.py`, `pipeline/kinetics.py`, `tasks/box_lift/loads.py` | Inverse dynamics: subtree Newton–Euler at the subject's size and mass, ground reaction and foot split, the box's load on the hands |
| `pipeline/opensim_id.py`, `pipeline/validate/kinetics_compare.py`, `tasks/box_lift/force_plates.py` | OpenSim joint centres and Inverse Dynamics with the same loads; the comparison; the force-plate check |
| `pipeline/export_c3d.py` | Gap-filled markers as C3D (rigid body fill, else the model's marker) |
| `tests/`, `.github/workflows/ci.yml` | The tests (pytest) and the CI that runs them |
| `pipeline/export_viewer.py`, `pipeline/viewer_layers.py` | Model and clip files for the browser viewer; plots, force arrows and the OpenSim overlay in a clip |
| `mocap2myo/` | MyoFullBody model files (MuscleMimic; Apache-2.0, with a non-commercial restriction on the arm and hand parts derived from MoBL-ARMS: see the README and the model's NOTICE) and small kinematic helpers |
```
