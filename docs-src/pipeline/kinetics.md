(kinetics)=
# Inverse dynamics

Net joint forces and moments come from Newton–Euler sums over subtrees of MyoFullBody, scaled to the subject's size and body mass (plus the 1–3 kg COSMED K5 system the subjects wore in these trials, lumped at the torso). For a joint, the parent segment's force and moment on everything distal to it equal the rate of change of that subtree's momentum minus gravity and the external loads on it (`mocap2myo/kinetics.py`, `pipeline/kinetics.py`). Body positions are low-passed at 6 Hz before the double differentiation (`M2M_KIN_LOWPASS_HZ`), as the markers are for the IK. The floor height comes from the standing foot (a running low percentile of the sole points that are still), so a foot flat on the ground counts as in contact. Body mass comes from the static trial's force plates when they weigh within 80–125 % of the listed mass. Frames the IK did not solve (listed in its output) are interpolated and shaded like the undetermined ones.

| Output | How it is determined | Reliability |
|---|---|---|
| L5/S1 moment and force; shoulder and elbow moments | Top-down: the motion of the trunk, head and arms and the box's load on the hands. No ground reaction needed. | {bdg-success}`Determined` except while the box is half on a surface (shaded in the viewer) |
| Total ground reaction and its centre of pressure | The same sum over the whole body | {bdg-success}`Determined` (checked against the force plates below) |
| Hip, knee and ankle moments | Bottom-up from the ground reaction under each foot. In single support the foot carrying it is known; in double support it is shared between the feet (below). Where the recording has force plates, a foot on a plate gets the plate's force and centre of pressure, and the other foot the rest of the total | {bdg-primary}`Estimated` in double support off the plates (shaded in the viewer); {bdg-success}`Measured` on a plate |
| L5/S1 compression | Axial force plus the extension moment over a 5 cm erector-spinae moment arm (8 cm abdominals for flexion), the classic single-equivalent-muscle estimate; with {doc}`muscle forces <muscles>`, also from every muscle crossing L5/S1 | {bdg-primary}`Approximation` (single muscle); muscle-level with `--muscles` |

**The box's load on the hands.** While the box is off its support (support state "hands" in the reference) the hands carry *m*(*g* − *a*) through the box centre plus the couple −d*H*/d*t*, from the box pose fitted to its markers and its 13.6 kg mass. The total is what L5/S1 needs; for the shoulders and elbows it is shared between the grasping hands by a minimum-norm split, applied at the palm centres (`tasks/box_lift/loads.py`). Frames where the box is held but still within 2 cm of the floor or the shelf are flagged: there the surface carries an unknown part of it.

```{admonition} Model parameter fixed for the dynamics
:class: warning

MyoFullBody inherits from MyoArm eight massless shoulder "phantom" bodies (clavphant, scapphant, humphant, humphant1) with a placeholder inertia of 1 kg m² each, more than the whole trunk; the arm model's 1 g thorax keeps a thorax inertia too. Rotating with the shoulder joints, they added up to ~15 N m of false axial torque at L5/S1. The kinetics drops the inertia of every body under 10 g (their mass is kept). The model files are not changed.
```

| Joint loads (median over 22 trials) | Value |
|---|---|
| Ground force / weight while not carrying | 0.999 (range 0.996–1.002) |
| L5/S1 extension moment, 95th pct | 178 N m (trials 137–205) |
| L5/S1 extension moment, peak | 268 N m (trials 210–343) |
| L5/S1 compression (single-muscle estimate), 95th pct | 3.9 kN (trials 3.1–4.4; NIOSH action limit 3.4 kN) |
| Elbow flexion moment, 95th pct (R / L) | 30 / 30 N m |
| Shoulder flexion moment, 95th pct (R / L) | 31 / 30 N m |

## Force plates

- **Which foot is on a plate.** A plate carrying more than 20 N belongs to the foot whose centre is within 25 cm of its centre of pressure horizontally and whose lowest sole point is within 10 cm of the plate's height. The plate, not the kinematics, says which foot is down; the height test keeps a foot swinging over the plate, or standing on the step below it on a staircase, from making the plate ambiguous (stair ascent against AddBiomechanics: knee moment r 0.00 → 0.64–0.82, hip 0.70 → 0.93; {doc}`../validation/addbiomechanics`). If both feet qualify, the plate is shared and both stay estimated.
- **Filtered like the motion.** The plate forces and moments are low-passed at the same cut-off as the body positions (6 Hz) before they enter the equations of motion (`M2M_KIN_FILTER_PLATES`, default on): filtering only the kinematics leaves the two sides of the equations with different frequency content, which shows up as spikes in the joint moments at heel strike (Kristianslund, Krosshaug & van den Bogert 2012). The centre of pressure and free moment are solved again from the filtered force and moment about the plate centre, so nothing changes without the filter.
- **Plates without geometry.** AddBiomechanics files carry no plate corners; their plates are placed on the plane of their centre of pressure, and a plate whose centre of pressure lies at several heights (one channel for several stair steps) is split into one plate per height.

## Double support off the plates

How the total ground reaction is shared between two feet on the ground (`M2M_KIN_SPLIT`):

- `lever` (default): by the lever rule along the line between the two feet's contact centres, from where the total's centre of pressure lies.
- `vpp`: from where each foot's contact is (Skals et al. 2017 contact detection) and the virtual pivot point above the centre of mass the ground reaction lines cross in walking (Gruben & Boehm 2012); on the public walking data it lowered the hip moment error against the plates from 59 to 23 N m and 49 to 13 N m RMS, and on the case study the foot share error from 0.080 to 0.070.

## Segment masses

The model's own mass distribution, scaled to the subject's body mass, is the default (`M2M_KIN_BSIP=model`), as OpenSim's Scale tool keeps it. Two standard tables can replace it per segment group (head and neck, trunk with the pelvis, upper arm, forearm and hand, thigh, shank, foot), keeping the model's distribution, centres of mass and inertia shapes inside each group:

| Table | Thigh | Shank | Foot | Head and neck | Trunk | Source |
|---|---|---|---|---|---|---|
| MyoFullBody, as built | 10.0 % | 4.5 % | 1.4 % | 2.8 % | 56.3 % | the model |
| `dempster` | 10.0 % | 4.65 % | 1.45 % | 8.1 % | 49.7 % | Dempster 1955 as tabulated by Winter 2009; what Vicon Plug-in-Gait uses |
| `deleva` (male / female) | 14.16 / 14.78 % | 4.33 / 4.81 % | 1.37 / 1.29 % | 6.94 / 6.68 % | 43.46 / 42.57 % | de Leva 1996 (Zatsiorsky–Seluyanov, young adults); by sex with `M2M_SEX_COLUMN`, else the mean |

MyoFullBody's legs follow Dempster; its head is light and its trunk heavy against both tables. A head that is part of the torso body (the HAT of Rajagopal 2016, Gait2354, Lai–Uhlrich 2022) counts as trunk.
