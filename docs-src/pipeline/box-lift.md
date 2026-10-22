# Task: box lifting

(stage-ref)=
## Stage 4 · Reference construction: floor alignment, box pose, grasp and support labels

```{container} m2m-files
`tasks/box_lift/build_reference.py`
```

- The body is aligned with the floor (the elevator platform). Subjects wore thick-soled slides; with the static-trial calibration the barefoot model's soles sit between 1.3 cm below and 2.6 cm above the platform before this shift (median over trials −0.2 cm).
- Box size: width and height from the marker layout; length from the distance between the palms while carrying (0.428 m). Box mass 30 lb (13.6 kg).
- Shelf top height = height of the box bottom while the box rests on the shelf.
- Grasp label: the box is off the floor and shelf, and that palm is close enough to the box surface.
- Support state (floor, hands, shelf) is labelled, and each carry cycle is found: box lifted off the floor until it is set back down.

```{container} m2m-io
**Out** `ref/` (`M2M_REF_DIR`)
```

(stage-qa)=
## Stage 5 · QA gating and carry-cycle segmentation

```{container} m2m-files
`pipeline/qa.py` · `tasks/box_lift/build_dataset.py` · `pipeline/report.py`
```

- Each carry cycle becomes one clip at 100 Hz.
- Clips that fail the checks are dropped automatically: 95th-percentile marker residual above 6 cm, unsolved frames, sudden root jumps, pelvis markers far off, or the pelvis turning too fast (IK loses the pelvis when a deep squat goes past the model's limits).
- A manual review on top: after viewing the reference motions, subjects S10 and S11 and the 5th cycle of S09 Spring were removed.
- `report.py` grades every trial on gaps, fit, leave-one-marker-out prediction, joint flips and standing posture (PASS / WARN / FAIL with reasons). It does not drop anything by itself.

```{container} m2m-io
**Out** `data/<M2M_DATA>/`: final run 63 of 110 carry cycles pass, 25.9 min (S02–S09)
```
