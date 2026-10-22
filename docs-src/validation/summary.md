# Validation summary

Every step was checked against something independent of it: another lab's processing of the same markers, a different measurement of the same motion, or a physical measurement. The marker residual alone is not enough: it says how tightly the model was fitted to the markers it saw, not whether the joint angles are right ({doc}`joint-angles`).

| Check | Reference | Result | Page |
|---|---|---|---|
| Joint angles, public walking data | Vicon Plug-in-Gait output in the same C3D files (36 trials, 3 subjects) | knee flexion r = 0.995–0.997, hip flexion 0.991–0.994 (RMS after the offset 1.6–2.5°) | {doc}`public-data` |
| Joint angles and moments, public data | AddBiomechanics' own IK and ID on the same markers (5 subjects; walking, sit-to-stand, stairs) | hip and knee flexion r = 0.995–1.000; hip, knee and low-back moments in sit-to-stand r = 0.99–1.00 | {doc}`addbiomechanics` |
| Joint angles, simultaneous inertial capture | Xsens MVN suit worn at the same time (13 people, industrial tasks) | knee flexion r = 0.99 (median), RMS after the offset 6–7° | {doc}`xsens` |
| Marker fit and standing posture | OpenSim (Rajagopal 2016; Scale, Marker Placer, IK) on the same markers (22 lifting trials) | residual 9.0 vs 19.8 mm; quiet-standing knee 0.1° vs 10.6° | {doc}`opensim` |
| Joint moments | OpenSim Inverse Dynamics with the same external loads | L5/S1 r = 0.984, shoulders and hips 0.97–0.99 | {doc}`inverse-dynamics` |
| Whole-body dynamics | force plates (22 lifting trials) | total vertical force within 1.9 % of body weight; the box's mass from the plates 13.2 kg (13.6 weighed) | {doc}`inverse-dynamics` |
| Marker prediction | each marker hidden in turn, predicted from the others | median 14.4 mm per trial (12.0–24.0) | {doc}`markers` |
| Gap filling | hidden stretches of measured trajectories, against established fills | rigid body fill 1–7 mm where it applies; spline only for short gaps | {doc}`gap-benchmark` |
| Settings | each setting varied around its value | conventional values sit on flat parts of the curves | {doc}`sensitivity` |

Where the numbers come from: the case study (22 box-lifting trials, OptiTrack, 11 subjects; {doc}`case-study`) and four public datasets, all CC BY 4.0: Vielemeyer et al. 2026 (Vicon, ramps), Moissenet & Schreiber 2019 (Qualisys, gait speeds), Maurice et al. 2019 (AnDyDataset, Qualisys with an Xsens suit), and the AddBiomechanics Dataset (Werling et al. 2024; its Falisse et al. 2017 part). The settings and their sources are on {doc}`parameters`.

## Known limits

- **Offsets are not errors by themselves.** Against Plug-in-Gait the hip flexion is about 10° lower and the elbow 6–12°; against Xsens the elbow is about 20° lower. These come from different definitions of the segments (Plug-in-Gait's pelvis from the ASIS and PSIS markers, Xsens' elbow from its own skeleton); the correlations and the RMS after the offset are the comparable part.
- **Quiet-standing hip flexion** of 9° (median over the case study's trials; up to 19°) is partly where the pelvis markers sit on the body and partly real posture; OpenSim gives 12°.
- **Knee moment** in the case study is lower than OpenSim's (95th percentile 52–56 vs 75–76 N m): the two models put the knee's centre of rotation differently.
- **Muscle forces** are an estimate: MyoFullBody's back extensors cannot supply the low-back moment of heavy lifting in about a fifth of the frames, which then need reserve torques ({doc}`../pipeline/muscles`).
- **Raw-take defects** of the case study (a wrong skeleton template, rotated head labels, labelling gaps) are listed on {doc}`case-study`.
