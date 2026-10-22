(opensim)=
# Against OpenSim

The standard OpenSim workflow, run on exactly the markers this pipeline uses: the Rajagopal 2016 full-body model with the case study's Plug-in-Gait marker set, Scale and Marker Placer on the static trial with the subject's mass, then IK (`pipeline/validate/opensim_ik.py`; `run_all.sh --opensim`). Both are scored by the same code (`pipeline/validate/opensim_compare.py`).

**The current version** (static-trial calibration, the default settings) against OpenSim:

```{include} ../_generated/case_opensim_ik.md
```

**Calibration choices**, from an earlier version of the IK (before the spline gap filling and the 0.08 ankle smoothing): the calibration from the motion trials alone, with the static trial placing the pelvis and legs (the default), or the pelvis and legs only, against OpenSim.

```{table}
:class: tabnum

| Measure (22 trials) | motion trials | static, legs | static, legs only | OpenSim |
|---|---|---|---|---|
| Marker residual, median / 95th pct (mm) | 10.0 / 35.5 | 9.2 / 32.7 | 9.5 / 35.8 | 19.8 / 77.3 |
| Quiet-standing knee flexion, median (trials ≥ 10°) | 18.2° (16) | 0.1° (0) | 0.0° (0) | 10.6° (12) |
| Quiet-standing \|hip flexion\|, median (trials > 10°) | 13.2° (15) | 9.4° (10) | 8.8° (9) | 11.6° (13) |
| Frames with a joint faster than 15 rad/s, median | 0.16 % | 0.08 % | 0.10 % | 1.18 % |
| Knee vs Xsens (S11 Spring): offset R / L | -8.2° / -10.0° | +1.3° / -2.4° | +1.1° / -2.2° | -4.0° / -7.7° |
| Knee vs Xsens: RMS after the offset R / L | 3.2° / 2.8° | 3.8° / 3.0° | 4.1° / 3.5° | 3.9° / 3.6° |
```

OpenSim's residual is about twice ours for two reasons: the Rajagopal model has no scapula (the shoulder markers ride on the thorax, so arm elevation is fitted poorly when the box goes onto the shelf), and one scaling setup is used for every subject. Joint angles, ours minus OpenSim over all frames: knee -8.9° / -8.2°, hip +3.0° / +3.3°, ankle -0.5° / +0.2° (R / L). OpenSim's IK took 16 min of wall time for the 22 trials on the same machine; it is faster than ours, which solves 37 degrees of freedom with a robust loss and contact constraints.
