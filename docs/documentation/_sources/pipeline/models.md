(models)=
# Models and marker sets

The model and the marker set are both data, so the stages run with any pair of them. A **model profile** (`mocap2myo/profiles/<name>.toml`, chosen with `M2M_MODEL`) names the model file and how to fetch it, its root and neutral pose, the joints the IK solves, which bodies are the canonical segments (pelvis, torso, head, femur, tibia, foot, humerus, ...), the inverse-dynamics joints with the axes their moments are reported about, and the feet and hands. `mocap2myo/landmarks.py` computes some 70 anatomical landmarks from the model's own bones: joint centres and bony points such as the ASIS, the femoral epicondyles, C7 and the styloids.

```{table}
:class: tabnum

| Profile | Model | IK joints | Muscles | Markers used, Plug-in-Gait / ISB |
|---|---|---|---|---|
| `myofullbody` | MyoFullBody (MuscleMimic 1.0.5), in this repository; the default | 31 + root | 416 | 39 / 52 |
| `ms_human_700` | MS-Human-700, whole body | 37 + root | 700 | 39 / 52 |
| `ms_human_700_locomotion` | MS-Human-700 legs on a rigid upper body | 14 + root | 100 | 16 / 26 |
| `mass` | MASS (Lee et al. 2019), converted from its DART files (`models/mass_to_mjcf.py`) | 46 + root | 284 | 39 / 52 |
| `rajagopal2016` | Rajagopal et al. 2016 full body, converted with MyoConverter | 23 + root | 80 | 39 / 52 |
| `lai_uhlrich2022` | Lai–Uhlrich 2022 (OpenCap), converted with MyoConverter | 27 + root | 80 | 39 / 52 |
| `gait2354` | OpenSim Gait2354 (no arms), converted with MyoConverter | 17 + root | 54 | 27 / 38 |
| `myoleg` | MyoLeg (MyoSim 0.2.3): legs on a passive torso | 14 + root | 80 | 16 / 26 |
```

A **marker set** (`mocap2myo/markersets/<name>.toml`, chosen with `M2M_MARKER_SET` or the dataset profile's `marker_set`) describes each marker against the anatomy, not a model, so one file serves every model: the segment it moves with, the landmark under it, its offset from that landmark in the segment's anatomical axes (anterior, superior, lateral, mm), its weight and its calibration group (`loose`: may move up to 15 cm, `foot`, or the default 6 cm). Roles name the head, pelvis, heel, toe and ankle markers; alias sets map other labellings onto it and belong to their set.

- A marker on a segment or landmark the model lacks, or on a body no IK joint moves relative to the pelvis (MyoLeg's trunk, the arms of MS-Human-700 legs), is left out.
- A joint with no active marker beyond it (the neck of MASS and MS-Human-700 with the ISB set, which has no head markers) is held at the neutral pose instead of solved; the toes keep their prior.
- On MyoFullBody, Plug-in-Gait keeps the hand-tuned starting placements the case study was calibrated with (the set's `[override]`), so the case-study results are unchanged; on any other model, and for any other set, the placements come from the landmarks.
- The frame-to-frame smoothing of the IK is scaled with the square root of the frame rate (its lag in frames grows with the weight squared), so a 200 Hz recording is smoothed over the same time as a 100 Hz one.

```{admonition} Checked
:class: tip

MyoFullBody with Plug-in-Gait is bit-identical to the version before marker sets became data: conditioned markers of S01, IK on a slice of S01 with and without the static trial, the inverse dynamics, and the full trials S08 Solid (11,738 frames) and S11 Spring (9,771 frames), largest joint-angle difference 0.0. All profiles run end to end on one ramp trial (200 Hz) and one Moissenet trial (100 Hz); see {ref}`public-data validation <public>`.
```

```{admonition} Knee joint frame
:class: warning

MyoFullBody's and MyoLeg's knee rotates about the tibia's origin, 8.9 mm from the joint frame of the OpenSim knee it was converted from (up to 15 mm at the ankle in deep flexion). The opt-in profiles `myofullbody_kneeframe` and `myoleg_kneeframe` move it there. Against Xsens (S11 Spring) the knee offset moves by about 1° (right +1.3 → +0.4°, left −2.4 → −3.2°) and the RMS after the offset is unchanged (3.8 → 3.9°, 3.0 → 3.0°), so the default stays.
```

**Adding a model.** Write a profile (the comments in `mocap2myo/profile.py` list every table), give `models/fetch.py` a source in `mocap2myo/profiles/sources.toml` or point the profile at a file, then run `python models/check.py <name>`: it compiles the model, renders its landmarks, and checks the kinetics axes and the left / right symmetry. A trunk frame that is not y-up, x-anterior needs the L5/S1 `superior` / `anterior` keys. **Adding a marker set:** one TOML file as above; a landmark that is missing goes into `LANDMARKS_EXTRA` in `mocap2myo/landmarks.py`; check the static trial's per-marker residual.

The lifting task (`tasks/box_lift/`: box loads, hand contact, the reference) is written for MyoFullBody's bodies; other models run with `M2M_TASK=none`. Model files other than MyoFullBody are fetched at pinned commits into `M2M_MODELS_DIR` and keep their own licences (`mocap2myo/profiles/sources.toml`).
