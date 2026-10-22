| Method | Ramp (Vicon): 0.3 s / 1.0 s | Case study (OptiTrack): 0.3 s / 1.0 s | Applies to |
|---|---|---|---|
| The model's marker in the IK pose (this pipeline) | 13.7 / 11.6 mm | 14.6 / 18.5 mm | every gap |
| Rigid body fill, 3 neighbouring markers (Nexus; Smolka & Lukasik 2016) | 2.3 / 6.9 mm | 1.0 / 3.7 mm | 62 % / 88 % of the samples |
| Cubic spline (Nexus spline fill) | 4.1 / 57.4 mm | 1.6 / 19.5 mm | 100 % / 100 % of the samples |
| Low-dimensional Kalman smoother (Burke & Lasenby 2016) | 50.8 / 64.0 mm | 5.9 / 8.5 mm | 100 % / 100 % of the samples |

Frames with every body marker (what the Kalman smoother learns from): ramp 23 %, case study 47 % (median over trials, with the windows hidden).
