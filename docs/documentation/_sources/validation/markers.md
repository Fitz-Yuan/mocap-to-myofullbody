(lomo)=
# Marker prediction: leave one marker out

Every body marker is left out in turn (one frame per second, all 22 trials), the frame is solved again, and the model's prediction of the missing marker is compared with where it was measured. Script: `pipeline/validate/lomo.py`. Final run (static-trial calibration); per marker, the median over trials:

- Prediction error per trial: median 12.0–24.0 mm (median 14.45), against a residual of 6.6–11.3 mm with all markers.
- Legs are pinned down well: ankle 8–10 mm, knee 11–13 mm, shank 7–9 mm.
- Shoulder 23–25 mm (residual 17 mm) and elbow about 19 mm (residual 12 mm): the upper arm is the loosest part of the fit.
- Finger 37–44 mm and toe 19–21 mm: each is the only marker that sets the wrist or toe angle, so leaving it out is expected to fail.
- This test cannot see a whole-subject offset: if the calibration bent the knees, the other markers still predict each other well.
