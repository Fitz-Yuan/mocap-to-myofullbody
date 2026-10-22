# Public data: Vicon Plug-in-Gait and every model

(public)=
## Validation on public data: Vicon Plug-in-Gait outputs, and every model on the same trials

Two public gait datasets (both CC BY 4.0) go through the same stages as the case study, read from their C3D files with their force plates. **Vielemeyer et al. 2026**: level and ramp walking, Vicon, Plug-in-Gait markers, 200 Hz, three inclined AMTI plates; its C3D files also carry Vicon's own Plug-in-Gait joint angles and moments, an independent reference. **Moissenet and Schreiber 2019**: gait at five speeds, Qualisys, the 52-marker ISB / Dumas set, 100 Hz, plates and EMG.

(pig)=
## Joint angles and moments against Vicon Plug-in-Gait (36 level-walking trials, 3 subjects)

MyoFullBody, static-trial calibration, the default settings, per-trial comparison with the Plug-in-Gait outputs in each C3D (`pipeline/validate/pig_compare.py`): correlation, mean difference (offset) and RMS difference after removing it, median over trials. Moments are compared only while that side's foot is on a plate, where Vicon includes the ground reaction.

```{include} ../_generated/ramp_pig.md
```

Knee and hip flexion follow Plug-in-Gait closely in shape. The hip flexion offset of about −10° is a difference of definition: Plug-in-Gait measures the thigh against its own pelvis frame from the ASIS and PSIS markers, the model against its pelvis bone. The ankle agrees least (r 0.80–0.84), partly because the two define the foot differently (Plug-in-Gait from the heel and toe markers). Its frame-to-frame smoothing matters here: the case study's 0.3 (`M2M_IK_ANKLE_SMOOTH`), set to steady the ankle in deep squats, lags the faster ankle of walking (r 0.38–0.51), so the gait profiles set 0.08; that also lowered the frames at the joint speed limit (0.11 → 0.05 %) and the marker residual (7.8 → 7.6 mm). Moments agree in shape (hip and ankle r 0.94–0.98, knee 0.81–0.93) with offsets of 8–18 N m, which come from where each model puts the joint centres relative to the plates' centre of pressure.

(allmodels)=
## Every model on one trial of each dataset

Ramp: trial Ref01 level up 04 (200 Hz). Moissenet: trial 2014001 C3 03 (100 Hz). Static-trial calibration (pelvis and legs) for every model, the default settings. Residual: median / 95th percentile distance between measured and model markers. Plates: RMS of the predicted total vertical force against the plates, in % of body weight. No unsolved frames and no joint flips on either trial.

```{include} ../_generated/models.md
```

The quiet-standing knee is 0° for every model on both datasets (Gait2354 −2° on one side): the static trial pins it. Models with fewer markers (legs only) fit the markers they have more tightly, which is not higher accuracy. The plate RMS stays within a few per cent of body weight for every model; the plates are low-passed like the motion, which removed most of the difference the ramp data showed before (7–9 % with unfiltered plates). IK takes 8–74 s per trial on one core, depending on the model.
