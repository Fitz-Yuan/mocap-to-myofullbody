(stage-read)=
# Reading recordings

```{container} m2m-files
`mocap2myo/tak.py` · `pipeline/read_raw.py` · `pipeline/validate/tak_reader.py`
```

A .tak file is a Microsoft OLE compound file, which the open-source library olefile can open. The marker trajectories are in `Channels.dat`: one record of frame number plus x, y, z per marker per frame, with no record when the marker was not seen. Marker names, the skeleton or rigid body each marker belongs to, and each rigid body's marker layout are in `Nodes.dat`. We worked out this layout ourselves; there is no official specification and no existing open-source reader.

```{container} m2m-io
**In** .tak

**Out** `tak_cache/`: marker trajectories per trial (Motive world, Y up, metres)
```

```{admonition} Checked
:class: tip

All 11 processed takes in the dataset (S01, S03, S04, S09, S10, S11: anatomy, platform and S03's two NE trials) are identical to Motive's own TRC exports, largest difference 0.005 mm (float32 rounding). Raw takes match the TRC exports wherever both have a sample; they differ only where the lab edited the take in Motive before exporting.
```

```{admonition} Caution
:class: warning

Verified for this study's Motive version. For data from another Motive version, compare against one TRC export first: `python -m mocap2myo.tak IN.tak --compare EXPORT.trc`.
```

Other labs' data comes as C3D or TRC (`mocap2myo/readers.py`; C3D through ezc3d), described by a dataset profile (`datasets/*.toml`: file kind, which files are trials and static trials and how they are named, up axis, units, rate, where heights and masses come from, the task). The reader puts the markers in metres in a Z-up world, keeps the lab software's model outputs (joint angles, forces, segment points) apart from the markers, and turns each force plate into the ground reaction on the body per marker frame (the analog samples of a frame averaged): force, centre of pressure on the plate surface (also on inclined plates), free moment about the vertical, plate corners, in the markers' world frame. These go into the marker files for the later stages, with the other analog channels (EMG) by name. Every stage keeps the recording's frame rate.

```{admonition} Checked on public data
:class: tip

Vielemeyer et al. 2026 (Vicon, Plug-in-Gait, 200 Hz, three AMTI plates, subject Ref01): the 39 markers found by name, 76 segment points and 82 model outputs kept apart; body mass from the plates 67.0 kg (12 level trials, mean vertical force between two plate contacts corrected for the pelvis's vertical acceleration, range 66.8–67.2; the static trial 67.1 kg; the article lists 65 kg), the same on the inclined plates of the ramp trials; centre of pressure inside the plate and under the stance foot (from the heel marker to just past the toe marker) in every loaded frame. IK on one level trial (1.65 s): body-marker residual median 7.0 mm, 95th percentile 23 mm; static pose knees 0°, hips −2° (without the static trial: hips −11°, ankles 7–9°). Moissenet and Schreiber 2019 (Qualisys, 100 Hz, one subject, 25 trials): 52 markers, height and mass from the C3D, mass from single support on the plates 67.1 kg (C3D: 67.0). The TRC route reproduces the take route on a Motive TRC export (static trial, largest difference 0.005 mm).
```

## Motive CSV

A CSV export from Motive (`M2M_INPUT=csv`; format versions 1.1 to 1.25) is read from its header rows: the columns of type *Marker* are the measured markers, named without their skeleton's prefix (`Skeleton:LHeel` → `LHeel`); the frame rate and the length unit come from the first line; an empty cell is a gap. *Bone Marker* columns are where Motive's solved skeleton expects its markers, not measurements, and are not read; an export with only bones or rigid bodies stops with a message saying to export the markers. Motive's *Coordinate Space* option moves bones and rigid bodies, not markers, which are always in the world frame (Y up). In a take with several people, a name two skeletons share keeps its prefix, so their markers are never mixed, and `M2M_POINTS_EXCLUDE` leaves the others out ({doc}`../getting-started/optitrack`).

```{admonition} Checked
:class: tip

Exports from Motive 2 and 3 (format versions 1.21 to 1.25) by other labs read as described: a Baseline (37) take at 360 Hz, a Baseline + 13 (50) take, a Conventional Lower Body (16) take at 120 Hz; bones-only and rigid-body-only exports stop with the message.
```

## Qualisys TSV

A TSV export of 3D data from Qualisys Track Manager (`M2M_INPUT=tsv`) is read from its header lines: the marker names (`MARKER_NAMES`) and the rate (`FREQUENCY`); with or without the Frame and Time columns; a sample of 0, 0, 0, an empty cell or `NA` is a gap. QTM writes millimetres (`M2M_UNITS` overrides; `check` says when the markers' size does not fit the unit). QTM writes a trial's force plates, analog channels and rigid bodies into separate files next to it (`<trial>_f_1.tsv`, ...): these are recognised by their content and are not taken as trials. The force plates are not read from them; for force plates, export C3D from QTM.

```{admonition} Checked
:class: tip

Three public recordings exported by QTM both as TSV and as C3D (54, 62 and 23 markers, 100–250 Hz): the same markers, the same gaps, positions within 0.001 mm (the TSV's rounding).
```

## AddBiomechanics `.b3d`

An AddBiomechanics `.b3d` file holds one subject's trials (`M2M_INPUT=b3d`, `pip install nimblephysics`); each trial is read as `<file>.b3d#<trial>`, and the profile's `trial_regex` sees that path. The markers come as AddBiomechanics received them (names stripped of stray characters), in OpenSim's Y-up world (`up = "y"`); the force plates as force, centre of pressure and free moment. The file has no plate corners, so each plate gets a rectangle on the plane of its centre of pressure (for the filter, {doc}`kinetics`), and a plate channel whose centre of pressure lies at several heights (one channel for several stair steps) is split into one plate per height. The subject's height, mass and sex come from the file (`anthro = "c3d"` in the profile). Example: `datasets/addbiomechanics_falisse2017.toml`, and the comparison with AddBiomechanics' own results: {doc}`../validation/addbiomechanics`.

## Checking a dataset before running

`mocap2myo check` (`pipeline/check.py`) reads the profile and the first trial and lists the trials and static trials it finds, each subject's height and mass, the rate, units, up axis and force plates, and how many of the marker set's markers the labels match (directly or through an alias set), with what to set where something does not fit.
