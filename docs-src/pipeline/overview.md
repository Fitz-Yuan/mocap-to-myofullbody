(pipeline)=
# Pipeline overview

```{raw} html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../../assets/pipeline-dark.svg">
  <img alt="Pipeline: raw recording → reader → markers → calibration on a static trial → IK, then QA, inverse dynamics, muscles, and optionally OpenSim" src="../../../assets/pipeline-light.svg" style="width:100%">
</picture>
```

`mocap2myo run` runs the steps below in order; each is also a script that runs on its own (`python pipeline/<step>.py --help`). The settings come from the dataset profile and `M2M_*` variables ({doc}`../reference/settings`); the folders from the profile's `root`.

| Step | Script | In | Out |
|---|---|---|---|
| {doc}`Read <reading>` | `pipeline/read_raw.py` | the recordings: `.tak`, C3D, TRC, `.b3d` | `tak_cache/` or `raw_cache/`: markers in metres, force plates, analog channels, heights and masses |
| {doc}`Markers <markers>` | `pipeline/prepare_markers.py` | the cache | `markers/`: the marker set's markers, Z up; mislabelled samples dropped; the pelvis made rigid |
| {doc}`Calibration and IK <ik>` | `pipeline/solve_ik.py` | `markers/` | `ik/<subject>_offsets.npz` (where each marker sits on the model), `ik/<trial>.npz` (the pose per frame, each marker's error) |
| Task reference (optional) | `tasks/<task>/build_reference.py` | `ik/`, the task's objects | `ref/`: e.g. the box, the grasp, the support phases ({doc}`box-lift`) |
| {doc}`QA <qa>` | `pipeline/qa.py`, `pipeline/report.py` | `ik/` | `ik/qa.json`, `ik/report.md`: PASS / WARN / FAIL per trial |
| {doc}`Joint loads <kinetics>` | `pipeline/kinetics.py` | `ik/`, force plates, the task's loads | `kin/<trial>.npz`: joint moments and forces, the ground reaction |
| {doc}`Muscles <muscles>` (optional) | `pipeline/muscles.py` | `ik/`, `kin/` | `kin/muscles/<trial>.npz`: activations and forces |
| OpenSim (optional) | `pipeline/validate/opensim_ik.py`, `pipeline/opensim_id.py` | the same markers and loads | the comparison ({doc}`../validation/opensim`) |

Exports for viewing and for other software: `pipeline/export_viewer.py` (viewer clips) and `pipeline/export_c3d.py` (gap-filled C3D).

## What is done where

- **No hand processing.** The readers take the raw files of the capture software; labelling must have been done (live labels are enough). Gaps, mislabels and an occluded pelvis are handled by fixed rules ({doc}`gap-filling`), and each rule leaves a record.
- **One model for every step.** The IK, the inverse dynamics and the muscle forces use the same MuJoCo model, scaled to the subject. Seven models are available ({doc}`models`).
- **Every trial graded.** The QA does not drop trials: it says which part of which trial to trust ({doc}`qa`). A task (such as box lifting) may drop clips by its own rules.
- **CPU or GPU.** Every step runs on the CPU, one trial per process (`--workers`); the IK also runs on an NVIDIA GPU, all trials at once ({doc}`ik`).
