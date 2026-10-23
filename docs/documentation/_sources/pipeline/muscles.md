# Muscle forces

```{container} m2m-files
`pipeline/muscles.py` · `mocap2myo/muscles.py` (the optimisation) · `pipeline/validate/emg_compare.py`
```

An optional step (`mocap2myo run --muscles`): each frame's muscle activations by **static optimisation**, the least sum of squared activations whose moments match the net joint moments from the inverse dynamics (Crowninshield & Brand 1981), with the model's own muscle geometry and force properties.

- **Which moments.** The joints whose loads are determined (`M2M_MUSCLE_JOINTS=auto`): L5/S1, the shoulders and the elbows from the top down; the hips, knees and ankles too where the trial's ground reaction is measured by force plates.
- **Muscle model.** Each muscle's maximum isometric force, force–length and force–velocity curves and passive force, and its moment arms about every matched joint axis, all from the MuJoCo model in the frame's pose (moment arms checked against MuJoCo's own to 1e-9).
- **Reserve torques.** One per matched moment, with a high cost (`M2M_MUSCLE_RESERVE`, 1 N m): near zero where the muscles can supply the moment, the remainder where they saturate. They show where the model is too weak, rather than hiding it.
- **L5/S1 compression from the muscles**: the net force on the trunk above L5/S1 minus the pull of every muscle crossing it, the muscle-level counterpart of the single-equivalent-muscle estimate.
- About 0.2 ms per frame; a two-minute trial in about 15 s on one core.

```{container} m2m-io
**In** `ik/`, `kin/`

**Out** `kin/muscles/<trial>.npz` (`M2M_MUSCLE_DIR`): activation and force per muscle and frame, the reserve torques, the matched moments, the muscle-based L5/S1 compression
```

## What it shows on the case study

- **The back extensors are too weak for heavy lifting.** MyoFullBody's erector spinae at full activation give about 150 N m of extension, while the lifts need 187 N m at the 95th percentile and about 270 N m at their peak (medians over the trials). In the median trial 29 % of the frames need an L5/S1 reserve above 1 N m (14–42 % by trial). Scaling every muscle's strength with the subject's body mass (`M2M_MUSCLE_SCALE=mass`; Handsfield et al. 2014) does not help overall (median 32 %, 16–55 % by trial): it makes the heavier subjects stronger and the lighter ones weaker, so it stays off by default.
- **A passive flexor.** The rectus abdominis sits at 1.39 times its optimal fibre length in the model's neutral pose, so its passive force alone gives about 40 N m of trunk flexion.
- **L5/S1 compression from the muscles**: 95th percentile 4.7–5.1 kN by trial, against 4.0 kN from the single-muscle estimate (median; NIOSH action limit 3.4 kN).

## Against EMG

`pipeline/validate/emg_compare.py` compares the activations with the surface EMG recorded in the same trials (envelopes; time courses rather than absolute levels, since the maximum voluntary contractions were not reliable):

- The EMG leads the joint moments by 35–70 ms, as electromechanical delay predicts; it does not share the force plates' 100 ms lag ({doc}`../validation/inverse-dynamics`).
- Deltoid activation follows the EMG (r ≈ 0.65), biceps less (0.34–0.51), erector spinae weakly (0.34).
- The triceps does not (r ≈ 0): static optimisation adds no co-contraction beyond what the moments need, and the abdominals are never activated for the same reason.
