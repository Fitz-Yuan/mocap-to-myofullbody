(terms)=
# Glossary

```{glossary}
Marker and label
  A reflective ball on the body. Its label tells IK which point on the model it belongs to; without that, IK cannot tell where to pull.

Gap
  Frames where a marker was hidden. Short gaps are interpolated, pelvis gaps are refilled as a rigid body, and IK handles the rest (see {ref}`gap filling <gaps>`).

Plug-in-Gait
  A common 39-marker full-body placement (LASI, RKNE, C7, ...). Used by the elevator study.

Kabsch fit
  Finds the rotation and translation that best map a set of points known in an object's frame onto their measured positions. Used for the box and the pelvis.

Forward kinematics
  From joint angles to the position of every segment and virtual marker. Computed by MuJoCo.

Least-squares IK
  The reverse: the joint angles that minimise the weighted sum of squared distances between virtual and measured markers.

Calibration
  Finding where each marker sits on its segment, since placement differs from person to person.

Residual
  How far a virtual marker still is from the measured one after the fit: a measure of how tight the fit is, not of whether the joint angles are right.

Model profile
  A small TOML file that tells the pipeline how a model is built: its file, root, neutral pose, IK joints, canonical segments, kinetics joints, feet and hands.

Marker set
  A TOML file that places each marker on the anatomy: segment, landmark, offset in the segment's anatomical axes, weight. The same file works on every model.

Leave one marker out
  Solve without one marker and predict where it should be. Tests whether the other markers pin that segment down.
```
