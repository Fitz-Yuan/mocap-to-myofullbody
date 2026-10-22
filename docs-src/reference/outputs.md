# Files and formats

(stage-io)=
## Stage inputs and outputs

Run the scripts from the repository root; `run_all.sh` runs them in order, and each runs on its own (`python pipeline/<stage>.py --help`). The motion dataset of a task goes to `data/<M2M_DATA>/` of the repository.

```{table}
:class: refs

| Script | Input | Output |
|---|---|---|
| `pipeline/read_raw.py` | `raw_tak/*.tak` (`M2M_RAW_DIR`; C3D, TRC) | `tak_cache/` (`M2M_CACHE_DIR`) |
| `pipeline/prepare_markers.py` | `tak_cache/`, anthropometrics | `markers/` (`M2M_NPZ_DIR`; with force plates and analog channels from a C3D) |
| `pipeline/solve_ik.py` | `markers/` | `ik/` (`M2M_IK_DIR`: offsets, qpos, residuals, unsolved frames) |
| `tasks/box_lift/build_reference.py` | IK, `markers/` | `ref/` (`M2M_REF_DIR`) |
| `pipeline/qa.py` | IK, `markers/` | `ik/qa.json` |
| `tasks/box_lift/build_dataset.py` | IK, refs | `data/<M2M_DATA>/` |
| `pipeline/report.py` | all of the above | `ik/report.json`, `report.md` |
| `pipeline/kinetics.py` | IK, refs (the task's loads), anthropometrics | `kin/` (`M2M_KIN_DIR`: joint loads per trial, `summary.json`) |
| `pipeline/validate/opensim_ik.py`, `pipeline/opensim_id.py` (`--opensim`) | `markers/`, `kin/`, `opensim_setup/` | `opensim_ik/`, `opensim_id/` |
```

(viewer-files)=
## Viewer files

`mocap2myo export viewer OUT_DIR TRIAL ...` writes both; the script itself, `python pipeline/export_viewer.py model OUT_DIR` once per model, then `clip TRIAL OUT_DIR` per trial, has more options (`--clip N`: one carry cycle of the task; `--public NAME`: a de-identified clip; `--ghost-ref-dir`: a second run overlaid; `--help` lists the rest). Joint loads and the OpenSim overlay are included when `kin/` and `opensim_id/` exist. A clip carries body poses per frame, measured and model markers, joint-angle and joint-load curves, force arrows, the QA grades, and two task-neutral lists the viewer draws as given: objects (shape, size, pose) and state tracks (e.g. hand contact). The viewer reads the files in the browser; nothing is uploaded.

(motion-files)=
## Motion clips for training

`mocap2myo export motion OUT_DIR [trials] --fps 100` (`pipeline/export_motion.py`; `--stem`, `--min-s`) writes every trial's IK as clips at one rate. A trial is cut where the IK lost a limb for 0.2 s or longer (`mocap2myo.ik.limb_lost`) or did not solve a frame; stretches of at least `--min-s` seconds (default 1) are the clips. Each is resampled (joint angles and root position linearly, root orientation by slerp) and lowered so that the feet stand on z = 0 (the median floor height under the still feet). Positions are in the model's size, as the IK solves them.

| File | Arrays |
|---|---|
| `motion_<stem>_<fps>hz.npz` | `gts` [N, bodies, 3] and `gvs` the tracked bodies' positions and velocities (m, m/s), `root_pos`, `root_vel`, `root_quat` (w, x, y, z), per clip `motion_num_frames`, `length_starts`, `motion_fps`, `motion_dt`, `motion_lengths` (s), `clip_file`; `tracked_bodies` |
| `initial_poses_<stem>_<fps>hz.npz` | `qpos_joints` [N, nq − 7] the joint angles, `joint_names` |
| `index.json` | per clip: the trial, its start and end in the trial (s), the frames, the floor height, the trial's marker error and lost-limb time |

N is the sum of the clips' frames; clip k runs from `length_starts[k]` for `motion_num_frames[k]` frames.
