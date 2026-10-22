(stage-markers)=
# Marker conditioning

```{container} m2m-files
`pipeline/prepare_markers.py` · `mocap2myo/markerset.py` · `tasks/box_lift/box_fit.py` (box, shelf, platform, floor)
```

Brings every trial to one format: the 39 Plug-in-Gait body markers, the box's 6 markers and pose, the shelf and elevator platform positions, and the floor height.

- Coordinates are turned to Z up.
- Mislabelled samples are dropped: a marker's distance to the steadiest markers on its own segment should stay nearly constant, and a sample that breaks this is treated as a wrong label.
- Missing pelvis markers are refilled from the other three, using the pelvis as a rigid body.
- Box: a template of its 6 markers is taken while the box rests in S03; each frame's box position and orientation come from a rigid (Kabsch) fit of that template.
- Shelf and floor: static markers, sorted by height.
- S01's labels are renamed (below), and the four head markers of every trial are assigned by their front/back and left/right position.

```{container} m2m-io
**In** stage 1 trajectories, subject height and weight (`meta/anthropometrics.csv`)

**Out** `markers/` (`M2M_NPZ_DIR`)
```

```{admonition} Checked
:class: tip

The raw-take markers were compared point by point with the collaborator's export. In 18 of 22 trials, 98–99.9% of the shared samples are identical; box pose, floor and shelf heights are identical in all 22. S06: the export was smoothed (3–5 mm per point, no time shift). S01: it was reconstructed again in Motive (differences below 1 mm).
```

## Other labels for the same markers

A recording whose labels differ from the marker set's names is mapped through an **alias set** (`[aliases.<name>]` in `mocap2myo/markersets/<set>.toml`, label → marker): the set whose names cover most of the trial's labels is used, if it covers more than the set's own names. An alias set may rename only a few markers; every other label keeps its own meaning (AnDyDataset's forearm wands, `LFRA` / `RFRA`, are Plug-in-Gait's `LFRM` / `RFRM`). The case study's S01, recorded with a Motive template of ISB names, maps 35 labels this way.
