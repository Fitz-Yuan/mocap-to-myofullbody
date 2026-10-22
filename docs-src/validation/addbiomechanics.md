# Against AddBiomechanics

[AddBiomechanics](https://addbiomechanics.org) (Werling et al. 2023, *PLOS ONE*) processes motion capture automatically with OpenSim models; its dataset (Werling et al. 2024) publishes, per subject, the markers and force plates together with its own scaled model, IK and inverse dynamics in one `.b3d` file. The pipeline reads the same file (`M2M_INPUT=b3d`), so both start from identical data, and `pipeline/validate/addbio_compare.py` compares the results frame by frame: the joint angles after AddBiomechanics' low-pass pass, and the joint torques of its dynamics pass on the frames it marks as having ground reaction forces.

Data: the Falisse et al. 2017 part, five subjects (three men, two women) with Plug-in-Gait markers (plus medial and cluster markers, left out), force plates, walking, sit-to-stand and stair ascent (`datasets/addbiomechanics_falisse2017.toml`). With `M2M_MODEL=rajagopal2016` the two pipelines use the same model and coordinates.

**Rajagopal 2016, the same model as AddBiomechanics**

```{include} ../_generated/addbio_rajagopal2016.md
```

**MyoFullBody**

```{include} ../_generated/addbio_myofullbody.md
```

- **Joint angles** agree closely: hip and knee flexion r = 0.995–1.000 with offsets of a few degrees (the calibrations place the markers differently).
- **Sit-to-stand moments** agree as well (hip, knee, low back r = 0.99–1.00). The ankle moment in sit-to-stand is small and noisy in both.
- **Walking moments** agree less: the leg moments r 0.79–0.94, the low-back moment (small in walking) 0.35–0.49. AddBiomechanics' dynamics pass adjusts the kinematics and the body's mass properties so that the motion is consistent with the measured forces; the pipeline keeps the kinematics as fitted, as classic inverse dynamics does. Part of the difference is this choice, part is how few walking frames have a valid ground reaction in the file (104 of 197 in the first trial).
- **Stair ascent** first showed the knee moment uncorrelated (r ≈ 0): the swinging foot passes over the stair's plate, and the plate went unused. The plate assignment now also checks the foot's height ({doc}`../pipeline/kinetics`): knee moment r 0.64 (Rajagopal) and 0.82 (MyoFullBody), hip 0.93 and 0.95.
