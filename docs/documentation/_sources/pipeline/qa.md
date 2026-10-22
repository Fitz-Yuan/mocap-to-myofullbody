(report)=
# Quality checks and the report

Every trial is graded on each check below (`pipeline/qa.py` measures, `pipeline/report.py` grades). WARN and FAIL mark something to look at, not a failed trial: nothing is dropped here. A task's dataset builder may drop clips by its own rules ({doc}`box-lift`). The report is `ik/report.md` (a table) and `ik/report.json`.

| Check | What it measures | WARN / FAIL above | Why this limit |
|---|---|---|---|
| `resid_med_mm`, `resid_p95_mm` | body-marker residual over all samples: median, 95th percentile | 12 / 20 mm; 40 / 60 mm | skin markers move 1–3 cm on soft tissue; a median above 12 mm means the model or the calibration does not fit this subject |
| `frame_rms_med_mm` | the median frame's RMS marker error, as OpenSim's IK reports it | 20 / 30 mm | OpenSim's IK guideline: RMS under 2 cm |
| `frame_max_med_mm` | the median frame's largest marker error | 40 / 60 mm | OpenSim's IK guideline: largest error under 2–4 cm |
| `limb_lost_s` | the longest stretch in which the IK lost a limb: two or more of its markers more than 6 cm off, or set aside as mislabelled | 0.1 / 0.5 s | a lost arm moves 3–4 of 39 markers, which the residual percentiles do not show ({ref}`recovery`) |
| `flips` | share of frames with a joint faster than 15 rad/s | 0.3 % / 1 % | no human joint turns that fast; a fit jumping between two solutions does |
| `gap_max` | the worst placed body marker's share of missing frames | 20 % / 90 % | a segment held by few markers is held by its neighbours and the joint limits |
| `lomo_med_mm` | median over markers of the leave-one-marker-out prediction error (if run, {doc}`../validation/markers`) | 20 / 30 mm | how well the other markers place a hidden one |
| `stand_knee_deg`, `stand_hip_deg` | knee and hip flexion in quiet standing (pelvis and ankles still for 0.5 s, pelvis at its upright height) | 10 / 20° | quiet standing is near 0°; an offset here points to the calibration and shifts every frame of the subject ({doc}`../validation/joint-angles`) |

The per-trial grades of the case study are on {doc}`../validation/case-study`.
