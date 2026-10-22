# Parameters and their sources

Where the field has an established value, the pipeline uses it. Where it does not, the value is marked as this pipeline's choice, with the reason and how much the results depend on it ({doc}`sensitivity`). No setting was tuned on the data it is validated on: the one setting chosen from data (the ankle smoothing) was chosen with a leave-one-subject-out procedure and checked on other datasets.

## Markers and gaps

| Setting | Value | Source |
|---|---|---|
| Marker low-pass filter | 6 Hz, 2nd-order Butterworth, zero phase, per gap-free stretch (`M2M_LOWPASS_HZ`) | the usual cut-off for whole-body motion (Winter 2009, *Biomechanics and Motor Control of Human Movement*); walking keeps 99 % of its signal power below about 7 Hz. Sensitivity: 4 Hz is worse, 6–12 Hz are the same |
| Short gaps | cubic spline, up to 200 ms, seen on both sides (`M2M_GAP_FILL`, `M2M_MAX_GAP_S`) | Howarth & Callaghan 2010 (*Comput Methods Biomech Biomed Engin*): splines for short gaps; longer gaps from the segment's other markers |
| Longer gaps | not interpolated; the model places the marker | model-based reconstruction (Lu & O'Connor 1999, *J Biomech*; Vicon Nexus' Kinematic Fill) |
| Exported fills | rigid body fill from 3 neighbouring markers, else the model's marker | Smolka & Lukasik 2016 (HSI); the order of Camargo et al. 2020 (*Comput Methods Biomech Biomed Engin*) |
| Mislabelled sample | 4 cm off its usual distance to 2 of its 3 steadiest partners | this pipeline's rule: skin markers move 1–3 cm on soft tissue (Leardini et al. 2005, *Gait Posture*), a wrong label tens of centimetres |
| Pelvis rigid refill | template from frames within 5 mm of the median distances; a marker 12 mm off is replaced | this pipeline's rule: the four pelvis markers sit on bony landmarks of one rigid bone |

## Inverse kinematics

| Setting | Value | Source |
|---|---|---|
| Calibration | static standing trial for the pelvis and legs, motion trials for the rest | as OpenSim's Scale and Marker Placer on a static trial (Delp et al. 2007; Hicks et al. 2015, *J Biomech Eng*); why the static trial is needed: {doc}`joint-angles` |
| Robust loss | Cauchy, scale 2.5 cm (`M2M_ROBUST`; 0 = plain least squares, as OpenSim) | this pipeline's choice, against mislabelled samples; plain least squares is OpenSim's. Sensitivity: flat on the public data, case study better with 0 or 5 cm than 1.25 cm |
| Frame-to-frame smoothing | 0.05 per frame at 100 Hz (`M2M_IK_SMOOTH`) | this pipeline's choice; the marker prediction error is flat from 0.0125 to 0.1 |
| Ankle and subtalar smoothing | 0.08 per frame at 100 Hz (`M2M_IK_ANKLE_SMOOTH`) | chosen leave-one-subject-out against Vicon Plug-in-Gait on the ramp data (0.08 chosen every time; ankle r on the subject left out 0.63–0.85 vs 0.11–0.57 with 0.3), confirmed on the case study (foot-marker residual 6.8 → 6.0 mm) |
| Joint speed limit | 15 rad/s | this pipeline's choice: above the joint speeds of the tasks analysed, below a fit jumping between two solutions |
| Tracking loss | 2 or more markers more than 6 cm off: solve the frame again from scratch | 1.5 times the largest marker error OpenSim's IK guidelines accept (2–4 cm) |

## Quality checks

| Setting | Value | Source |
|---|---|---|
| Per-frame RMS marker error | WARN above 2 cm | OpenSim's IK guidelines (RMS under 2 cm) |
| Per-frame largest marker error | WARN above 4 cm | OpenSim's IK guidelines (under 2–4 cm) |
| Other limits | {doc}`../pipeline/qa` | this pipeline's, each with its reason |

## Inverse dynamics and muscles

| Setting | Value | Source |
|---|---|---|
| Body positions low-pass | 6 Hz, as the markers (`M2M_KIN_LOWPASS_HZ`) | as above |
| Force plates | filtered at the same cut-off as the motion (`M2M_KIN_FILTER_PLATES`) | Kristianslund, Krosshaug & van den Bogert 2012 (*J Biomech*) |
| Plate contact | 20 N | a common force-plate threshold for contact and the centre of pressure |
| Segment masses | de Leva 1996 (*J Biomech*), by sex where known (`M2M_KIN_BSIP`); Dempster 1955 as Winter 2009 tabulates it, or the model's own, on request | de Leva 1996; Winter 2009 |
| Double-support split | lever rule (default) or contact detection plus the virtual pivot point (`M2M_KIN_SPLIT=vpp`) | Skals et al. 2017 (*Multibody Syst Dyn*); Gruben & Boehm 2012 (*J Biomech*) |
| L5/S1 compression, single muscle | 5 cm erector-spinae moment arm, 8 cm for the abdominals | the classic single-equivalent-muscle estimate; NIOSH action limit 3.4 kN (Waters et al. 1993, *Ergonomics*) |
| Muscle forces | least sum of squared activations, with reserve torques | static optimisation (Crowninshield & Brand 1981, *J Biomech*) |
| Muscle strength | the model's own; optionally scaled with body mass (`M2M_MUSCLE_SCALE=mass`) | Handsfield et al. 2014 (*J Biomech*): muscle volume grows with mass × height |
