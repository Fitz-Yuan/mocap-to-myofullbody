(optitrack)=
# From OptiTrack (Motive)

Recordings from an OptiTrack system go in as they are: the `.tak` takes themselves (Motive is not needed), or Motive's CSV or TRC export. Motive's own skeleton solve is not used: the musculoskeletal model is fitted to the measured markers.

## 1. When you record

- **Use one of Motive's skeleton marker sets.** These are built in:

  | Motive marker set | `marker_set` |
  |---|---|
  | Baseline (37) | `motive_baseline37` |
  | Baseline (41), with heel and toe-tip markers | `motive_baseline41` |
  | Baseline + 13 Additional Markers (50) | `motive_core50` |
  | Conventional Full Body (39), Conventional Lower Body (16): Plug-in-Gait names | `plug_in_gait` |

  `mocap2myo init` picks the set from the labels. Another set is one TOML file in `mocap2myo/markersets/` ({doc}`../pipeline/markers`).
- **Record a static trial per subject**: a few seconds standing still, as its own take. It anchors the joint angles; without it the calibration uses quiet standing in the trials ({ref}`static-trial calibration <static>`).
- **Write down each subject's height and mass.** Motive's files do not hold them.
- **Labels are what matter.** Gaps are fine: the pipeline fills short ones and lets the model place the rest, so Motive's gap filling is optional.

## 2. Get the files out of Motive

Either way works; pick one for a study.

**The takes (`.tak`).** Copy them from the Motive project's session folders. Set `input = "tak"` in the profile. The take reader was checked against Motive's own exports for the case study's Motive version ({doc}`../pipeline/reading`); with another version, compare one take with its TRC export first: `python -m mocap2myo.tak TAKE.tak --compare EXPORT.trc`.

**A CSV export.** In Motive: select the takes, *File → Export Tracking Data*, format CSV. In the export options:

- tick **Markers** (the measured markers: the only columns read);
- keep **Header information** (the frame rate and length unit come from it);
- the rest does not matter: units (mm or m), Rigid Bodies, Bones and Bone Markers (Bone Markers are where the solved skeleton expects its markers, not measurements, and are not read).

Set `input = "csv"` in the profile. A TRC export works the same way (`input = "trc"`).

## 3. Run it

From here it is the same as for any other data ({doc}`your-data`):

```bash
mocap2myo init my_study.toml        # finds the CSV / .tak files and the marker set
mocap2myo check --dataset my_study.toml
mocap2myo run --dataset my_study.toml
```

Motive writes Y up; the up axis is found from the labels (`up = "auto"`).

A lower-body set (Conventional Lower Body) has no trunk or arm markers: it needs a legs model. `mocap2myo init` sees that and writes `model = "myoleg"` near the top of the profile, above the first `[section]`; fetch the model once with `mocap2myo models myoleg` (it goes to `~/data/mocap2myo/models`, or `M2M_MODELS_DIR`).

## 4. Motion for training

To use the motion for imitation learning or animation:

```bash
mocap2myo export motion clips/ --dataset my_study.toml --fps 100
```

writes every trial's motion as clips at a fixed rate, with the feet on the floor (`z = 0`): each body's position and velocity, the root's position and orientation, and the joint angles ({doc}`../reference/outputs`). A trial is cut where the IK lost a limb for 0.2 s or longer, and clips shorter than 1 s are left out; `clips/index.json` lists where each clip comes from.

## When something goes wrong

| You see | What to do |
|---|---|
| `no measured markers (columns of type ['Bone'])` | the CSV holds only the skeleton's bones: export again with **Markers** ticked |
| `several people in this take (Alice, Bob)` | two skeletons in one take: leave all but one out with `points_exclude = 'Bob:.*'` in the profile; for the other person, a second profile |
| `marker set ...: 30 of 37 markers` | some markers were renamed in Motive: add the names as an alias set (`[aliases.<name>]` in the marker set's TOML) |
| `360 Hz: the IK solves every frame` | set `rate = 120` in the profile's `[input]`: every 3rd frame, three times faster, and still well above human movement's frequencies |
| `no static trial and no quiet standing` (calibration) | the person never stood still: the joint angles may be offset. Record a few seconds of standing still, as its own take or at the start of a take |
| `up axis: no head / upper-trunk or no foot marker labels` | set `up = "y"` in the profile (Motive's default) |
