(xsens)=
# Against Xsens inertial capture

An Xsens MVN suit worn during the optical recording measures the same motion with different sensors (inertial units on the segments) and a different skeleton, so it shares none of the optical pipeline's error sources. `pipeline/validate/xsens.py` computes the same joint angles from MVN's exported anatomical points (knee and elbow flexion from the joint centres, trunk lean), resamples both to a common rate, aligns them in time (largest knee correlation within ±20 s) and compares correlation, mean offset and the RMS after removing it. Offsets come largely from the two skeletons' definitions (Xsens' elbow is about 20° straighter than the model's); the correlation and the RMS after the offset are the comparable part.

## 13 people, industrial tasks (AnDyDataset)

Maurice et al. 2019 (*Int J Robot Res*; CC BY 4.0): 13 participants doing industry-like tasks (carrying loads, screwing at different heights, walking), Qualisys with Plug-in-Gait markers (120 Hz, processed at 60 Hz) and an Xsens MVN Link suit (240 Hz) at the same time; no static trial, so the calibration uses quiet standing from the trials (`datasets/andydata.toml`). One trial per participant (60–120 s):

```{include} ../_generated/xsens_andy.md
```

Two participants stand out: P2274's right elbow (r 0.69, offset −44°) and P5319's knees (RMS after the offset 14–16°); their causes are not yet traced. Long MVN exports store their frame count as a real number, which the common C3D reader rejects; the comparison reads those files directly.

## The case study (S11)


In the case study the subjects also wore an Xsens suit. Only S11 Spring NE has a usable export (a c3d of MVN's 64 virtual anatomical points, 60 Hz; the other trials exist only as raw .mvn files, and S11 Solid's c3d is empty). The same geometric angles were computed on both sides and aligned by cross-correlation (lag 0.05 s). Script: `pipeline/validate/xsens.py`.

```{table}
:class: tabnum

| Angle | Correlation, before the fix / final | Mean offset (IK − Xsens), before the fix / final | RMS after removing the offset, before the fix / final |
|---|---|---|---|
| Right knee | 0.994 / 0.995 | −8.2° / +1.3° | 3.2° / 3.8° |
| Left knee | 0.995 / 0.996 | −10.0° / −2.4° | 2.8° / 3.0° |
```

"Before the fix": calibration from the motion trials alone; "final": with the static trial (pelvis + legs). Negative offset = the IK knee is more flexed (the included angle is smaller). The knee's shape over time matches almost exactly; before the fix the level was 8–10° apart, and the model's own geometry accounts for only 1.4° of it. Elbow and trunk were also computed but are not comparable: Xsens gives the acromion rather than the shoulder joint centre, and the model's elbow carries a 13° geometric offset. Xsens has its own errors of several degrees, so this is supporting evidence, not ground truth.


## With an exoskeleton (S11)

S11's six trials with two exoskeletons (Solid, Spring; three tasks each) also have Xsens exports: knee r 0.89–0.97, RMS after the offset 7–13°. They are not a clean check: the exoskeleton's belt hides the front pelvis markers for long stretches (the pelvis is then rebuilt from the static trial's template), and its metal and motors disturb the Xsens magnetometers (one export is degenerate, one knee r = 0.45).
