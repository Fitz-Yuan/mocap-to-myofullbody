# Mocap to MSK

**Motion capture in, musculoskeletal motion and joint loads out.**

Mocap to MSK turns optical motion capture (OptiTrack `.tak` or Motive's CSV export, C3D, TRC, AddBiomechanics `.b3d`) into the motion and joint loads of a MuJoCo musculoskeletal model: MyoFullBody by default, or MS-Human-700, MASS, Rajagopal 2016, Lai–Uhlrich 2022, Gait2354 and MyoLeg. It reads the raw recordings itself, calibrates each subject, solves the inverse kinematics frame by frame, computes the joint loads by inverse dynamics (and, optionally, muscle forces), and grades every trial so you can see how far to trust it. Every step was checked against an independent reference: Vicon Plug-in-Gait, AddBiomechanics, OpenSim, Xsens inertial capture and force plates.

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} Get started
:link: getting-started/installation
:link-type: doc
Install, run the demo on a public walking trial, then your own data.
:::

:::{grid-item-card} How it works
:link: pipeline/overview
:link-type: doc
Each step, from the raw file to joint loads, and why it is done that way.
:::

:::{grid-item-card} How well it works
:link: validation/summary
:link-type: doc
Every check against an independent reference, with its numbers and limits.
:::

:::{grid-item-card} Reference
:link: reference/settings
:link-type: doc
All settings, the command line, the files it writes.
:::
::::

## Quick start

```bash
git clone https://github.com/Fitz-Yuan/mocap-to-myofullbody
cd mocap-to-myofullbody
python -m venv .venv && source .venv/bin/activate
pip install -e ".[c3d,tak]"
mocap2myo demo
```

The demo downloads one public walking trial (about 0.4 MB) and runs every step on it in about a minute. Then {doc}`process your own data <getting-started/your-data>`.

```{toctree}
:caption: Getting started
:maxdepth: 2
:hidden:

getting-started/installation
getting-started/demo
getting-started/your-data
getting-started/optitrack
getting-started/results
getting-started/troubleshooting
```

```{toctree}
:caption: How it works
:maxdepth: 2
:hidden:

pipeline/overview
pipeline/reading
pipeline/markers
pipeline/gap-filling
pipeline/ik
pipeline/qa
pipeline/kinetics
pipeline/muscles
pipeline/models
pipeline/box-lift
```

```{toctree}
:caption: Validation
:maxdepth: 2
:hidden:

validation/summary
validation/parameters
validation/sensitivity
validation/joint-angles
validation/public-data
validation/addbiomechanics
validation/xsens
validation/opensim
validation/inverse-dynamics
validation/markers
validation/gap-benchmark
validation/case-study
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

reference/cli
reference/settings
reference/outputs
reference/validation-scripts
reference/code
reference/glossary
reference/reading
reference/license
```
