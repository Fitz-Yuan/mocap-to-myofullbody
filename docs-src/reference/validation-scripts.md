# Validation scripts

(validation-scripts)=
## Validation scripts

```{table}
:class: refs

| Script | What it checks |
|---|---|
| `pipeline/validate/tak_reader.py DIR` | the take reader against Motive's TRC exports |
| `pipeline/validate/lomo.py` | leave-one-marker-out: solve without a marker, predict it (writes `ik/lomo.json`, which `report.py` then grades) |
| `pipeline/validate/xsens.py IK_DIR` | knee angles against a simultaneous Xsens recording (`M2M_XSENS_TRIAL`, `M2M_XSENS_C3D`) |
| `pipeline/validate/static_pose.py` | IK pose of the static trials and bone-length ratios against the model |
| `pipeline/validate/opensim_ik.py` | OpenSim Scale + Marker Placer + IK on the same markers |
| `pipeline/validate/opensim_compare.py ["LABEL=IK_DIR" ...]` | our IK runs and the OpenSim run scored with the same checks (the Xsens row only with `ezc3d` and the c3d) |
| `pipeline/validate/kinetics_compare.py` | our joint moments against OpenSim Inverse Dynamics with the same loads |
| `tasks/box_lift/force_plates.py` | the ground reaction against the lab's force plates (case study; plate delay, gain, total force, foot split, box mass) |
```

(opensim-setup)=
## OpenSim comparison: the files it needs

`pipeline/validate/opensim_ik.py` needs three files in `opensim_setup/` (`M2M_OSIM_SETUP_DIR`), not part of the repository, and `opensim` from conda (`conda install -c opensim-org opensim`; `OSIM_PY` if it is another Python):

- `Model.osim`: the Rajagopal et al. (2016) full-body model ([SimTK](https://simtk.org/projects/full_body)).
- `markers.xml`: an OpenSim MarkerSet for that model with the labels of the marker set (Plug-in-Gait: `LASI`, `RKNE`, `C7`, ...).
- A ScaleTool setup (`M2M_OSIM_SCALE_SETUP`): its measurement set defines the scaling and its Marker Placer task set the marker weights; markers it switches off are switched on at a low weight. The script sets the model and marker files, the subject's mass and height, and the static trial.
