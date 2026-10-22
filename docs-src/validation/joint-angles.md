# Joint angles: what the residual cannot show

(validation)=
## Joint-angle validation: errors the marker residual cannot show

The marker residual only says how tightly the model was fitted to the markers it saw. Three checks look further.

(posture)=
## Quiet-standing bias in knee and hip flexion

Quiet standing frames are found from the markers alone: pelvis and both ankles moving less than 5 cm/s for 0.5 s, pelvis within 3 cm of its upright height. A person standing quietly has the knee at about 0–5° and the hip near 0°. Before the calibration fix (motion trials only):

- Knee flexion in quiet standing: 0–24°. At least 10° in 16 of 22 trials, above 20° in 7 (S07, S08, S09, S10 Spring).
- Hip flexion in quiet standing: up to 28° from neutral. Above 10° in 15 trials, above 20° in 5 (S01, S02 Solid, S10).
- Asymmetry within a subject, e.g. S07: right knee about 0°, left 20° in the most extended frames.

Final run (static-trial calibration): knee flexion in quiet standing median 0.1°, at least 10° in no trial; |hip flexion| median 9.4°, above 10° in 10 trials (see the QA report below).

```{admonition} Reading
:class: danger

Neither the residual nor the leave-one-marker-out test catches this, and Xsens independently points the same way at the knee. The static standing trials settle the cause: with the motion-trial calibration, the IK puts subjects who stand still and upright at 13–26° of knee flexion or 14–21° of hip extension, while their bone-length ratios match the model within about 5 % (S02 and S08 have nearly the same ratios and 0° vs 24° of knee flexion). So it is not a segment-length problem. The calibration lets each marker move a few centimetres on its bone, and a knee marker a few centimetres further back fits as well as a knee bent 20°: from the motion trials alone the two cannot be told apart.
```

(static)=
## Static-trial calibration: separating marker placement from joint angle

Each subject was recorded standing still and upright for 5 s before the task (`raw_tak/static/S<ss>_Anatomy.tak`; S07 has none and uses quiet standing from its trials). Those frames join the calibration with the hips, knees, ankles and subtalar joints pulled toward 0°, and carry half the weight of every update of the marker placements (`mocap2myo.ik.calibrate(static=...)`). This is the role of the static trial in OpenSim's Scale and Marker Placer. It is switched on by `M2M_STATIC=1`, which `run_all.sh` sets; the default in `pipeline/config.py` is `0`, so a stage run on its own (for example `pipeline/solve_ik.py`) needs `M2M_STATIC=1` to reproduce these results.

```{table}
:class: tabnum

| Measure (22 trials) | Motion trials only | Static trial: all markers | Static trial: pelvis + legs (default) | Static trial: legs only |
|---|---|---|---|---|
| Quiet-standing knee flexion, median | 18.2° | 0.05° | 0.1° | 0.0° |
| Quiet-standing \|hip flexion\|, median | 13.2° | 6.9° | 9.4° | 8.8° |
| Knee offset against Xsens (S11 Spring), R / L | -8.2° / -10.0° | +0.7° / −2.4° | +1.3° / -2.4° | +1.1° / -2.2° |
| Knee RMS against Xsens after the offset, R / L | 3.2° / 2.8° | 4.0° / 2.9° | 3.8° / 3.0° | 4.1° / 3.5° |
| Marker residual median / 95th pct, raw markers in the subject's size (as for OpenSim) | 10.0 / 35.5 mm | not computed | 9.2 / 32.7 mm | 9.5 / 35.8 mm |
| Marker residual median / 95th pct, the IK's own measure (height-scaled, filtered markers; median over trials) | 9.2 / 31.1 mm | 9.8 / 39.0 mm | 8.5 / 29.9 mm | 8.8 / 31.1 mm |
| Report verdicts FAIL / WARN / PASS | 11 / 11 / 0 | 3 / 16 / 3 | 1 / 17 / 4 | 2 / 14 / 6 |
| Carry cycles passing QA | 73 | 64 | 63 | 66 |
```

All three static variants remove the knee offset. Placing every marker with the static trial cost upper-body fit: with the arms hanging loosely in the static pose, the hand markers were placed poorly. Letting the static trial place only the pelvis and leg markers recovers that (the lowest residual, by either measure), and is the default of `run_all.sh` (`M2M_STATIC=1`, `M2M_STATIC_MARKERS=legs`). Against Xsens the three static variants are about the same: offsets within 2.4°, RMS after the offset 3.8–4.1° on the right and 2.9–3.5° on the left, slightly above the 3.2 / 2.8° of the motion-trial calibration, whose offset is much larger. Placing only the legs (pelvis from the motion trials) keeps three more carry cycles, mostly in subjects whose pelvis markers are unreliable, at a slightly higher residual; it is an option. The raw-marker residual is against the unfiltered markers in the subject's own size, so it reads higher than the IK's own measure.
