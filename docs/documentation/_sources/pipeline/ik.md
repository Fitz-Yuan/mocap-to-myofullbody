(stage-ik)=
# Calibration and inverse kinematics

```{container} m2m-files
`pipeline/solve_ik.py` · `mocap2myo/ik.py` (the solver)
```

IK finds the joint angles that bring the model's virtual markers as close as possible to the measured ones. It is the same idea as OpenSim's IK, written directly on MuJoCo's MyoFullBody and solved with `scipy.optimize.least_squares`; joint limits are the bounds of the variables.

- **Scaling.** The model stays as it is; the markers are scaled by the ratio of model height to subject height.
- **Calibration, once per subject.** 40 frames with varied poses are taken from all of the subject's trials, plus standing frames, which must have flat feet. The solver alternates between solving IK with the current marker placements and updating where each marker sits on its segment from the solved poses, like OpenSim's Scale plus Marker Placer. A marker may move from its starting position on the segment by up to 6 cm, 10 cm for the foot markers (on thick-soled slides) and 15 cm for the wand, trunk and head markers, whose placement varies most.
- **Static trial (the default).** The subject's static standing trial joins the calibration: in its frames the hips, knees, ankles and subtalar joints are pulled toward 0°, and they carry half the weight of every update of the marker placements. They place only the pelvis and leg markers (`M2M_STATIC_MARKERS=legs`); the arm and trunk markers come from the motion trials. `run_all.sh` turns this on (`M2M_STATIC=1`); the default in `pipeline/config.py` is off, so a stage run on its own needs `M2M_STATIC=1`. Why it is needed: see {ref}`joint-angle validation <validation>` and {ref}`static-trial calibration <static>`.
- **Per-frame solve.** 37 degrees of freedom: pelvis 6, legs 14, lumbar 3, arms 14. Other joints follow the model's coupling constraints. Each frame starts from the previous one, and the pelvis starting pose comes from a rigid fit to the four pelvis markers.
- **Safeguards.** Markers are low-pass filtered at 6 Hz (the usual cut-off for whole-body motion; {doc}`../validation/parameters`). A robust loss weakens the pull of a marker that is far off. Feet may not go through the floor, a standing foot stays where it was planted, and joint speed is capped at 15 rad/s so the solution cannot flip between frames. Each frame is pulled toward the previous one: weight 0.05 per frame at 100 Hz for most joints (`M2M_IK_SMOOTH`), 0.08 for the ankle and subtalar joints (`M2M_IK_ANKLE_SMOOTH`), scaled with the square root of the frame rate so the lag is the same in seconds at any rate; the shoulder's elevation plane is held while the arm hangs (it is undefined there).
- **Tracking-loss recovery** (`M2M_IK_RECOVER`, on by default; CPU only for now). A frame that leaves two or more seen markers more than 10 cm off is solved again from scratch and the better fit kept ({ref}`recovery`).

```{container} m2m-io
**In** stage 2 markers

**Out** `ik/`: joint angles (qpos) per frame, each marker's residual, and the frames the IK could not solve
```

```{admonition} Fit (final run)
:class: tip

Body-marker residual per trial: median 7.7–14.3 mm, 95th percentile 25–48 mm (the IK's own measure, on the height-scaled, filtered markers). Against the raw markers in the subject's own size, as used for the OpenSim comparison: median 7.9–14.3 mm per trial (9.2 mm, median over trials); OpenSim IK on the same markers 14.6–27.5 mm (19.8 mm; S09: 16.0 and 14.9 mm).
```

```{admonition} Why the static trial
:class: danger

From the motion trials alone, the calibration can trade where markers sit on a segment against the pose itself: a knee marker a few centimetres further back fits as well as a knee bent 20°, which shifts the joint angles of every frame of that subject. The static trial removes that freedom; see {ref}`joint-angle validation <validation>` and {ref}`static-trial calibration <static>`.
```

```{admonition} On the GPU
:class: tip

(`--gpu`, `M2M_IK_DEVICE=gpu`, `pip install -e ".[gpu]"`; `mocap2myo/ik_gpu.py`). The same per-frame problem (markers, robust loss, toe prior, floor, stance anchors, smoothing, speed and joint limits), with the kinematics and marker Jacobians in Warp kernels (float64) and a projected Levenberg–Marquardt step per frame, for every model, marker set and frame rate. The frame-to-frame recursion runs in parallel over time: each trial is cut into 64-frame chunks that step through their own frames, and chunks whose start differs from where the previous chunk ended are run again until every chunk starts where the one before it ended. All trials of a dataset go into one batch (with segment scaling, one batch per subject). The subject calibration and each trial's first frame stay on the CPU. The 22 case-study trials (about 41 min of motion at 100 Hz) take 122 s in all (101 s on the GPU), against 45–50 min on 22 CPU cores; joint angles agree with the CPU's to 0.002–0.03° RMS per trial and the residuals to 0.01 mm.
```

```{admonition} Model limits
:class: warning

The knee has no axial rotation, so all twisting during turns goes to the hip. Deep squats go past the model's 120° joint limits. There is no neck joint, so head markers carry a low weight (an optional neck for the IK, `M2M_NECK=1`, lets the head follow its markers; it is not validated).
```
