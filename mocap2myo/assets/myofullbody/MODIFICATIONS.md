# Changes made to the MyoFullBody model files

These files are MyoFullBody by the MuscleMimic team (musclemimic_models 1.0.5), built upon
MyoSuite / MyoSim models, under Apache-2.0 (`LICENSE`) with the third-party terms in `NOTICE`.
The arm and hand parts derive from the MoBL-ARMS model, released for non-commercial use only
(see `NOTICE`).

Changed in this repository:

- `body/myofullbody_clean.xml`, `body/myofullbody_clean_kneefix.xml`: copies of the released
  `body/myofullbody.xml` without its scene include (floor and skybox), with a headlight and
  offscreen-size block.
- `leg/assets/myolegs_assets_kneefix.xml`: a copy of `leg/assets/myolegs_assets.xml` with the
  left knee's coupled rotation / translation fixed (two coefficient sets, which in the released
  file turned the left shank the wrong way with flexion). The `kneefix` body file (the default,
  `MYOFULLBODY_MODEL=kneefix`) includes it; `MYOFULLBODY_MODEL=original` uses the released one.
