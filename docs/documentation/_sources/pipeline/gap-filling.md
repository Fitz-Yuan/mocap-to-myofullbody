(gaps)=
# Gap filling

Markers drop out whenever the cameras cannot see them: an arm in front of the body, the box against the belly. In a manual workflow an operator fills every gap in Motive or Nexus before anything else. Here nothing is invented ahead of the IK: wrong samples are removed, the pelvis is rebuilt as a rigid body, short gaps are interpolated by spline, and longer gaps are left to the IK, where the skeleton itself determines where a hidden marker must be. For other software, the gaps can be filled afterwards and written out ({ref}`c3d-export`).

```{raw} html
<figure class="fig">
<svg aria-label="Left: a marker's coordinate over time with a gap; a straight line across the gap misses the bump the marker really made. Right: a leg with hip, knee and ankle; the knee marker is hidden but the rigid femur, fixed at the hip and seen through the thigh marker, places it." role="img" viewbox="0 0 760 300">
<text class="t" x="20" y="24">Interpolation: a line across the gap</text>
<path class="ax" d="M30 250 H360 M30 250 V50"></path>
<text x="300" y="270">time</text>
<text x="34" y="64">marker position</text>
<rect fill="var(--m2m-warn-soft)" height="194" stroke="none" width="58" x="172" y="56"></rect>
<text x="178" y="246">gap</text>
<circle class="dot" cx="40" cy="150.0" r="2.6"></circle><circle class="dot" cx="54" cy="148.1" r="2.6"></circle><circle class="dot" cx="68" cy="145.7" r="2.6"></circle><circle class="dot" cx="82" cy="143.0" r="2.6"></circle><circle class="dot" cx="96" cy="140.2" r="2.6"></circle><circle class="dot" cx="110" cy="137.2" r="2.6"></circle><circle class="dot" cx="124" cy="134.0" r="2.6"></circle><circle class="dot" cx="138" cy="130.8" r="2.6"></circle><circle class="dot" cx="152" cy="127.4" r="2.6"></circle><circle class="dot" cx="166" cy="124.0" r="2.6"></circle><circle class="dot" cx="236" cy="124.0" r="2.6"></circle><circle class="dot" cx="250" cy="124.9" r="2.6"></circle><circle class="dot" cx="264" cy="125.8" r="2.6"></circle><circle class="dot" cx="278" cy="126.7" r="2.6"></circle><circle class="dot" cx="292" cy="127.6" r="2.6"></circle><circle class="dot" cx="306" cy="128.4" r="2.6"></circle><circle class="dot" cx="320" cy="129.3" r="2.6"></circle><circle class="dot" cx="334" cy="130.2" r="2.6"></circle><circle class="dot" cx="348" cy="131.1" r="2.6"></circle><circle class="dot" cx="362" cy="132.0" r="2.6"></circle>
<path class="faint" d="M166 124 C 186 70, 214 64, 236 124"></path>
<text class="k" style="fill:var(--m2m-faint)" x="168" y="62">where it went</text>
<path class="acc dash" d="M166 124 L 236 124"></path>
<text class="a" x="150" y="146">straight line</text>
<path class="ax" d="M400 30 V280"></path>
<text class="t" x="430" y="24">Model: the bones place a hidden marker</text>
<path class="bone" d="M560 70 L 585 165"></path>
<path class="bone" d="M585 165 L 572 262"></path>
<path class="ln" d="M520 62 L 600 62 L 610 80 L 512 80 Z"></path>
<circle class="jt" cx="560" cy="70" r="7"></circle>
<circle class="jt" cx="585" cy="165" r="7"></circle>
<circle class="jt" cx="572" cy="262" r="7"></circle>
<circle class="mk" cx="516" cy="60" r="6"></circle><circle class="mk" cx="604" cy="60" r="6"></circle>
<circle class="mk" cx="590" cy="116" r="6"></circle>
<circle class="mkh" cx="604" cy="166" r="8"></circle>
<circle class="mk" cx="592" cy="214" r="6"></circle>
<circle class="mk" cx="586" cy="262" r="6"></circle>
<text class="k" x="620" y="64">pelvis markers: hip centre</text>
<text class="k" x="604" y="120">thigh marker, seen</text>
<text class="a" x="620" y="170">knee marker, hidden</text>
<text class="a" x="620" y="186">placed by the femur</text>
<text class="k" x="606" y="218">shank markers, seen</text>
<text x="440" y="122">femur: rigid,</text>
<text x="440" y="138">fixed length</text>
<path class="ln" d="M520 128 L 566 118"></path>
<text x="440" y="270">joints: hip, knee, ankle</text>
</svg>
<figcaption>Left: over a long gap a straight line (or a spline) assumes the marker kept moving smoothly, which it rarely does during a lift or a step. Right: the knee marker is hidden, but it rides on the femur, a rigid bone of fixed length whose upper end (the hip centre, from the pelvis markers) and middle (the thigh marker) are seen, and whose lower end is held by the shank markers through the knee joint. The femur's pose, and with it the hidden marker, is determined.</figcaption>
</figure>
```

1. **Mislabelled samples are removed first** (`pipeline/prepare_markers.py`, `reject_mislabels`). Each marker's partners are the three markers whose distance to it varies least over the trial (the same or the next segment). A sample whose distance to two or more of its partners is more than 4 cm off the usual is a wrong label (e.g. C7 jumping onto another marker) and becomes a gap, rather than pulling the model the wrong way.
2. **The pelvis is rebuilt as a rigid body** (`fix_pelvis`). A template of the four pelvis markers comes from the frames where their six distances stay within 5 mm of their medians; each frame is fitted to it (Kabsch). A missing pelvis marker is reconstructed from the other three, and one more than 12 mm off, whose removal makes the other three consistent, is replaced. The pelvis is the model's root, so its markers get the strictest treatment.
3. **Short gaps are interpolated** (`mocap2myo/ik.py`, `load_trial`): up to 200 ms (`M2M_MAX_GAP_S`), only where the marker is seen on both sides, by a cubic spline through the samples within 200 ms on either side (at least two on each; a straight line otherwise; `M2M_GAP_FILL`). This is the usual recommendation for short gaps (Howarth & Callaghan 2010); on gaps cut out of measured trajectories it was 2–8 times more accurate than a straight line ({doc}`../validation/gap-benchmark`). The 6 Hz low-pass filter runs over each gap-free stretch on its own, so it never smooths across a gap.
4. **Longer gaps are left to the IK.** Each frame's least squares uses the markers that are there; a missing marker simply drops out of the sum. Bones are rigid, joints have limits, and each frame starts from the previous one and is pulled gently toward it, so the remaining markers on a segment and on its neighbours fix its pose; the hidden marker is wherever the bone puts it. A marker that is present but far from anywhere the model can put it pulls less (a Cauchy loss with a 2.5 cm scale), so one bad sample cannot drag the pose.
5. **A limb the IK lost is found again** (`M2M_IK_RECOVER`, on by default, see below).

(recovery)=
## Tracking-loss recovery

Solving frame by frame from the previous frame has one failure: once a limb slides into a wrong pose, the next frames start there too. The robust loss then fades the very markers that are right (a marker 30 cm off weighs about 1 % of one that fits), and the smoothing pulls toward the wrong previous pose, so the limb can stay lost for seconds. In the case study this happened while the box was put down: the measured wrist and finger markers stayed on the box, the model's left hand was 28–48 cm away for 1.7 s (S01 Spring, 24.2–25.9 s).

The IK checks every frame after solving it (`M2M_IK_RECOVER`, on by default; on the CPU only for now: the GPU IK runs without it and says so): if two or more seen body markers are more than 10 cm from the model (`M2M_IK_RECOVER_DIST`; 2.5 times the largest marker error OpenSim's guidelines accept), it solves the frame again from scratch, from four starts (the tracked pose and three random poses within the joint limits) without the smoothing and without the joint-speed limit, keeps the floor and stance constraints, and takes the result if it fits the markers better (the sum of squared marker distances, each capped at 15 cm so a mislabelled marker cannot decide, at least 20 % lower). The frames after it then start from the recovered pose. When the restarts find nothing better, the same set of far markers is tried again only every 10th frame while it lasts: a marker that stays off (a mislabel the robust fit already sets aside) would otherwise cost the restarts in every frame.

Why 10 cm and not less: a recovery snaps the pose, a jump of one frame. Where markers fit only loosely, a lower trigger also fires on misfits that are not a lost limb, and the restarts then move the limb to another pose that fits a little better. On the 22 case-study trials a 6 cm trigger gave the same longest lost limb as 10 cm in 20 trials (in the other two, S10, 0.8 s shorter) but three times the one-frame pose jumps (92 against 31). On AddBiomechanics' walking trials, whose arm markers fit to 6–12 cm at the ends of the arm swing, it turned the forearm by 40–50° in single frames (24 jumps in 9 trials, 5 with 10 cm), and the elbow agreed less with AddBiomechanics' (r 0.93–0.96 against 0.96–0.99 without recovery); 10 cm keeps most of that agreement.

Separately from the IK, the QA marks every frame where two or more of a limb's markers are more than 6 cm off or were rejected (`limb_lost_s`, the longest such stretch per limb); a task dataset drops a clip that loses a limb for 0.2 s or more ({doc}`qa`).

(c3d-export)=
## Gap-filled markers for other software

The pipeline itself never writes filled markers back into the data. For software that needs complete trajectories (OpenSim, Visual3D, ...), `pipeline/export_c3d.py` writes each trial's markers as C3D with every gap of a solved frame filled, in the order Camargo et al. (2020) automate the manual tools:

1. **Rigid body fill** from the three markers nearest to the target (over the trial) that are seen throughout the gap and at both its ends: the target's position relative to them at each end, carried by each end's rigid fit (Kabsch) and blended linearly over the gap (Vicon Nexus' rigid body fill; Smolka & Lukasik 2016). Where it applies it is the most accurate fill in marker space ({doc}`../validation/gap-benchmark`).
2. **The model's marker** otherwise: the same marker on the subject's calibrated model in the IK pose (as Nexus' Kinematic Fill), shifted to meet the measurement at both ends of the gap (the measured-minus-model offsets before and after it, blended linearly), so the trajectory does not step at the gap's edges.

Measured samples are written unchanged (residual 0), filled ones with residual 1, and the parameter group `M2M` lists how many frames of each marker were filled.

```{admonition} Nothing filled is hidden
:class: tip

The marker files keep, per frame, which pelvis markers were rebuilt (`pelvis_fixed`) and which samples were rejected as mislabels (`mislabel_rejected`); the IK output keeps each marker's residual, the markers it set aside and the frames it could not solve; the QA report grades every trial on its worst marker gap.
```

**Why not interpolate everything.** A line or a spline across a gap is a guess about the path; over a fraction of a second it is a good one, over a second of lifting it is not. The model makes no guess about the path: it uses the fact that the marker rides on a rigid bone whose other markers and joints are seen. This is the principle of OpenSim's inverse kinematics and of model-based gap filling in Vicon Nexus (global optimisation, Lu and O'Connor 1999); what this pipeline adds is that it runs without an operator and records what it did.

|  | Manual gap filling (Motive, Nexus) | This pipeline |
|---|---|---|
| Who decides | An operator, gap by gap: spline, pattern fill from a nearby marker, rigid-body fill | Fixed rules, the same for every trial |
| Long gaps | Filled with a curve or a copied pattern | Not filled; the IK places the marker from the skeleton (exported C3D: filled from neighbouring markers, else the model) |
| Afterwards | Filled samples look like measured ones; the IK trusts them fully | Rebuilt and rejected samples are flagged; inferred positions are the model's, never mixed into the data |
| Reproducible | Depends on the operator | Yes |
| Effort, 22 trials | An estimated 15–40 person-hours with labelling and scaling | Machine time only |
| Physically consistent | Not necessarily: a filled point can break a bone's length | Yes: bone lengths and joint limits hold by construction |

**How well it works.**

- **A gap with a known answer**: {ref}`leave one marker out <lomo>` hides a marker that was seen, lets the model place it, and measures how far off it is. Median 14.5 mm per trial (12.3–24.0); ankle 8–10 mm, knee 11–13 mm, shank 7–9 mm, shoulder 23–25 mm, elbow about 19 mm. Skin markers themselves move 1–3 cm on the soft tissue. Where a segment has a single marker (finger, toe) it cannot be placed once hidden: 37–44 and 19–21 mm.
- **Against manual gap filling**: the same takes, processed by hand in Motive by a collaborating group (in S03 alone, 5,765 filled samples the raw take lacks), give the same fit as the raw takes with their gaps: marker residual 9.2 vs 9.5 mm, 73 vs 76 usable carry cycles, joint angles within 1° in five of the eight subjects compared. In the other three, a whole-trial shift in hip flexion came from the subject calibration, fixed since by the static trial ({ref}`input sensitivity <routes>`).
- **Where manual filling went wrong**: Motive interpolated long ASIS gaps while the box was against the belly, which left the pelvis cluster 30–40 mm off rigid; the rigid refill does not.
- **An independent check**: the knee angle against Xsens inertial capture, worn at the same time, r = 0.995 / 0.996 ({ref}`knee angle vs Xsens <xsens>`).

```{admonition} Limits
:class: warning

A segment that loses most of its markers for a long stretch is held only by its neighbours, the smoothing and its joint limits; the QA report flags such trials (S02 Solid: C7 missing in 98 % of frames, FAIL). Markers must be labelled: the labels the recording software assigned live are enough, but an unlabelled point cloud is not identified. If fewer than three pelvis markers are seen, the frame's pelvis comes from the remaining markers (`M2M_PELVIS_GAP`).
```
