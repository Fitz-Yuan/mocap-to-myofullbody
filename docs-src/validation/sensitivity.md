# Sensitivity analysis

How much the results depend on each of the pipeline's settings, measured without any external reference: markers that were seen are hidden over stretches of 0.3 s and 1.0 s (four markers per stretch length and trial, the same stretches for every setting), the IK runs as usual, and the model's position of each hidden marker is compared with where it was measured (`data` folder script `gap_cv.py`; the case study on 30 s of six trials, the ramp data on 36 level-walking trials). A setting whose value sits on a flat part of this curve is not one the results hinge on.

```{include} ../_generated/sensitivity.md
```

- **Low-pass filter.** 4 Hz is worse, 6–12 Hz are the same: the conventional 6 Hz is on the flat part.
- **Smoothing.** Flat from 0.0125 to 0.1 (within 0.4 mm), worse at 0.2: the default 0.05 is in the middle.
- **Robust loss.** Flat on the public data; on the case study plain least squares (0) and a wide scale (5 cm) predict better than a narrow one (1.25 cm). The robust loss guards against mislabelled samples, which this test does not contain; it also made limb losses last longer, which the {ref}`tracking-loss recovery <recovery>` now ends.
- **Ankle smoothing.** On the ramp data flat from 0.02 to 0.6. On the case study 0.02–0.15 predict better than the earlier 0.3; 0.08 is the default for every dataset. It was chosen against Vicon Plug-in-Gait leave-one-subject-out on the ramp data (the subject left out of the choice: ankle r 0.81 / 0.85 / 0.63 with 0.08 vs 0.57 / 0.50 / 0.11 with 0.3).

The low-pass, robust and ankle sweeps ran on an earlier version of the IK (before the spline gap filling and the fixed smoothing weights); the smoothing sweep on the current one.
