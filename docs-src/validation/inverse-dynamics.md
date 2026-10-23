(kin-validation)=
# Inverse dynamics against OpenSim and force plates

Two independent checks. First, OpenSim Inverse Dynamics on the OpenSim IK (another model, another IK), given the same external loads: the box on each hand at our palm points and the ground reaction under each foot at our centres of pressure (`pipeline/opensim_id.py`, `pipeline/validate/kinetics_compare.py`). Median over the 22 trials, frames with an undetermined hand load left out:

```{include} ../_generated/case_opensim_id.md
```

The trunk, shoulders, elbows and hips agree closely. Knees and ankles differ by 13–21 N m RMS: the same centre of pressure falls a centimetre or two differently under two skeletons whose joint centres and knee angles differ (OpenSim's knees are about 9° more flexed throughout), and the hip–knee–ankle lever arms change with it. Neither is a reference for the other there.

Second, the lab's force plates. The plates were stored as raw voltages without their calibration matrix, so only the vertical channels are used, and only checks that need no calibration: the gain comes from quiet standing without the box (body + K5 weight), and the comparison uses the frames where both feet are on the two AMTI plates in front of the shelf (`tasks/box_lift/force_plates.py`).

| Check (22 trials) | Result |
|---|---|
| Plate delay behind the motion capture | 100 ms (median; 60–130 ms by trial) |
| Vertical gain from quiet standing (body + K5 weight) | 292.6 N/V (median; mean 292.9), standard deviation 0.4 % across 22 trials of 11 subjects |
| Total vertical force, ours vs plates (both feet on the AMTI plates) | RMS 15.2 N = 1.8 % of body weight (median) |
| Dynamic part (force minus static weight), correlation | r = 0.88 (median) |
| Box mass from the plates (with box minus without, quiet standing) | 13.2 kg (median; 11.3–14.3 kg by trial) vs 13.6 kg nominal |
| Right-foot share of the vertical force, ours vs plates | RMS error 0.08, r = 0.88 (median) |

```{admonition} What the plates confirm
:class: tip

The same gain for every subject (calibration consistent), the box mass recovered from standing with and without it, and the predicted total force within a few percent of body weight including its dynamic part: the whole-body accelerations from the IK are right. The left / right split in double support is the weak point (an estimate, see above); there are no plates under the elevator platform, where the box is lifted from the floor.
```

```{admonition} Clock offset
:class: warning

The plate channels run about 100 ms behind the motion capture (60–130 ms by trial), seen both in the force traces and as plate onsets after the foot lands in the marker data. Any analog channel recorded through the same path (EMG) needs the same check before it is compared with the motion.
```
