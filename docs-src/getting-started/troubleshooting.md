# When something goes wrong

Run `mocap2myo check --dataset <profile>` first: it reads the settings and the first trial and says what does not fit, before any long step runs.

| You see | What to do |
|---|---|
| `x no trial files match ...` | the file pattern (`raw_trials`, a glob under `raw_dir`) does not find your files; compare it with the folder `check` prints |
| `! N files match the glob but not M2M_TRIAL_REGEX` | those files are skipped; widen `trial_regex` if they are trials |
| `marker set ...: 12 of 39 markers` | your labels differ from the marker set's: choose another set (`marker_set`), or add your names as an alias set (`[aliases.<name>]` in `mocap2myo/markersets/<set>.toml`; an alias set may rename only a few markers) |
| `x pelvis markers missing` | the IK needs at least three of the four pelvis markers |
| `x the markers span 0.0018 m` | the file's length unit is not what it says (an old export in metres, say): set `units = "m"` or `"mm"` in the profile |
| `x no height / mass for S03` | add the subject to the CSV, named as in the trial names (`anthro_subject` formats a numeric cell, e.g. `S{:02d}`) |
| `x M2M_INPUT=c3d needs the ezc3d package` | `pip install -e ".[c3d]"` (`.b3d`: `nimblephysics`, `.tak`: `olefile`) |
| `M2M_MODEL=...: ... (models/fetch.py ...)` | the model is not fetched yet: `mocap2myo models <name>` |
| `M2M_TASK=box_lift needs a model with hands` | the lifting task needs arms; set `task = "none"` in the profile |
| a trial is WARN or FAIL in the report | the report names the check; {doc}`../pipeline/qa` says what each one means and when to drop the trial |
| the IK is slow | `--workers` (one trial per process, 1–2 GB each) or `--gpu` |

Still stuck? [Open an issue](https://github.com/Fitz-Yuan/mocap-to-myofullbody/issues) with the output of `mocap2myo check`.
