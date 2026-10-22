(reading)=
# Further reading

1. `mocap2myo/tak.py`: the format notes at the top explain what a raw file contains.
2. `pipeline/prepare_markers.py`: read `convert()` from top to bottom; each helper is one bullet of stage 2 (the box and scene part is `tasks/box_lift/box_fit.py`).
3. `mocap2myo/markerset.py` and `mocap2myo/markersets/plug_in_gait.toml` (which segment and landmark each marker is on; the model side is `mocap2myo/profile.py`), then the core, `mocap2myo/ik.py`: `MarkerModel.solve` (one frame), `calibrate` (calibration) and `ik_trial` (a whole trial, including foot locking).
4. `pipeline/solve_ik.py`: runs the functions above in parallel with the current settings (`pipeline/config.py`).
5. `tasks/box_lift/build_reference.py`, `pipeline/qa.py`, `tasks/box_lift/build_dataset.py`: scene, checks, clips.
6. `pipeline/report.py`, `pipeline/validate/lomo.py`, `pipeline/validate/xsens.py`: the validation.
7. `pipeline/run_all.sh`: the script that chains it all.

For background, work through OpenSim's Scale and IK tutorial first; the principle is the same. Solver details: [scipy least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html). Model and forward kinematics: [MuJoCo documentation](https://mujoco.readthedocs.io/).
