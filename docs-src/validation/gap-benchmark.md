# Gap filling against established methods

The same test as {doc}`sensitivity`: measured markers hidden over 0.3 s and 1.0 s, then reconstructed. Here each established method fills the same hidden stretches from the same markers, through the authors' own code: Vicon Nexus' spline and rigid body fills as reimplemented in the `gapfill` package (Jung 2020), and the low-dimensional Kalman smoother of Burke & Lasenby 2016 (`mogapfill`). The error is the median distance between the reconstructed and the measured marker, in the subject's own size.

```{include} ../_generated/gap_benchmark.md
```

- **Rigid body fill** is the most accurate where it applies: three neighbouring markers must be seen throughout the gap and at both its ends, which held for 62 % (ramp) and 88 % (case study) of the hidden samples.
- **The spline** is good for short gaps and poor for long ones, as Howarth & Callaghan 2010 found; the pipeline uses it up to 200 ms.
- **The Kalman smoother** learns the motion from frames in which every marker is seen. A walking subject crosses the room, so the motion in the room's frame is dominated by the walk; with few complete frames (the ramp data with the windows hidden) it fails.
- **The model's marker** is available for every gap, and its error is close to the model's residual on markers that are seen (8–12 mm): the model's marker is not the skin marker, which moves on the soft tissue. For the IK this is the right quantity: a hidden marker drops out and the pose comes from the others, so what matters is the joint angles, not where the hidden marker is drawn.

The pipeline therefore keeps the model-based approach for the IK and, for gap-filled marker files written for other software, fills by rigid body where it can and by the model elsewhere ({ref}`c3d-export`).
