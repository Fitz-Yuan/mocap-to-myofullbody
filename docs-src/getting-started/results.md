# Reading the results

All results go into the folder the dataset profile names as `root` (`M2M_ROOT`).

## Start with the report

`ik/report.md` has one line per trial: PASS, WARN or FAIL, the values behind the grade and the reasons. {doc}`../pipeline/qa` explains every check and its limits. A WARN is worth a look in the viewer; a FAIL says which part of the trial not to use (a lost limb, a bad marker fit, a crouched standing posture that points to the calibration).

## The files

| File | Contents |
|---|---|
| `ik/<trial>.npz` | `qpos` [frames, nq] the model's joint coordinates, `joint_names`, `marker_err` [frames, markers] each marker's distance to the model (m; NaN where missing), `rejected` (samples set aside as mislabelled), `rate`, `scale` |
| `ik/<subject>_offsets.npz` | the calibration: where each marker sits on the model |
| `kin/<trial>.npz` | joint moments and forces (`hip_r_ext`, `knee_l_ext`, `ankle_r_plantar`, `L5S1_ext`, `L5S1_compression`, ...; N m and N), the ground reaction and its centre of pressure, per foot `foot_force` and where it came from (`foot_source`: 1 force plate, 0 estimated), frames whose loads are not determined (`uncertain`) |
| `kin/muscles/<trial>.npz` | with `--muscles`: each muscle's activation and force, the reserve torques |

Load them with numpy:

```python
import numpy as np
K = np.load("ik/S01_walk01.npz")
names = list(K["joint_names"])
knee = K["qpos"][:, 7 + names.index("knee_angle_r") - 1]   # the root (a free joint) takes qpos[0:7]
```

{doc}`../reference/outputs` lists every array.

## Watch a trial

```bash
python pipeline/export_viewer.py model viewer/              # once per model
python pipeline/export_viewer.py clip S01_walk01 viewer/    # one trial
```

Drop the files on the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/). Use the same settings as for the run (`M2M_DATASET=...`).

## Gap-filled markers for other software

```bash
python pipeline/export_c3d.py filled_c3d/
```

writes each trial's markers as C3D with the gaps filled: from three neighbouring markers where they are seen (a rigid body fill), else from the subject's model in the IK pose. Residual 0 marks a measured sample, 1 a filled one. {doc}`../pipeline/gap-filling` explains the method and {doc}`../validation/gap-benchmark` its accuracy.
