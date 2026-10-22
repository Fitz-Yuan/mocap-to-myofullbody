# The case study: box lifting with OptiTrack

The pipeline was built on a box-lifting study: 11 subjects, each lifting a 13.6 kg box from the floor of an elevator platform onto a shelf and back, five carry cycles per trial, on two platform floors (Solid and Spring; trials `S<nn>_<floor>_NE`, without an exoskeleton), 22 trials of about 2 minutes. OptiTrack Motive at 100 Hz with Plug-in-Gait markers labelled live, the box, shelf, platform and floor tracked as rigid bodies, two AMTI force plates in front of the shelf, a static standing trial per subject; an Xsens suit for one trial ({doc}`xsens`). The recordings are the lab's own and are not published; one de-identified clip is in the [viewer](https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/). Profile: `datasets/mmh_elevator.toml`; the task: {doc}`../pipeline/box-lift`.

## Per-trial grades

The report of the current version ({doc}`../pipeline/qa` explains each check). Clips kept: the lifting dataset's own per-cycle checks, which are stricter.

```{include} ../_generated/case_report.md
```

**How to read it.** The most common flag is a **marker missing in many frames** (C7 in S02, the upper-arm wands in S06, a thigh wand in S08): a property of the recording, which the IK bridges with the other markers. The second is **hip flexion while standing** above 10° (S01–S05 and S10): the static trial pins the knees, but the hip angle also depends on where the pelvis markers sit on the body, and part of it may be real posture. S04 and S10 have the highest residuals (soft-tissue and marker placement); S02 Solid's only FAIL is C7 missing in 98 % of frames. **Cycles kept** is the dataset's own per-cycle check (95th-percentile marker residual, unsolved frames, root jumps, pelvis markers far off, pelvis turning speed, and the manual review that removed S10 and S11); it is stricter than the trial grade.

(sources)=
## Input data: raw Motive takes vs a hand-processed export

The pipeline reads the raw takes; no file anyone has processed by hand. An earlier version read an export of the same takes prepared by a collaborating group. That route is not part of this repository (its converter depends on the collaborator's files), but the two inputs were compared point by point, and the comparison is kept below as a check.

|  | Collaborator export (earlier version, not in this repository) | Raw take (this repository) |
|---|---|---|
| Input | Marker coordinates exported from Motive by a collaborating group, converted to xlsx | `raw_tak/*.tak` for each trial, `raw_tak/static/` for the static trials |
| Body labels | Edited and gap-filled by them in Motive | Labels Motive assigned live during recording |
| Box and scene markers | Labels lost in the export; recovered from geometry | Owning rigid body is in the file; still matched by geometry |
| Outputs | — | `tak_cache/`, then `markers/` (`M2M_NPZ_DIR`) |

(routes)=
## Input sensitivity: IK on the raw take vs the hand-processed export

Same IK code and settings, both with the calibration from the motion trials alone that was used before the static-trial fix; only the input differs. These are not the final results (those are in the sections below).

```{table}
:class: tabnum

| Measure (before the fix) | Raw take | Export | Notes |
|---|---|---|---|
| Marker residual, median of trial medians | 9.2 mm | 9.5 mm | range 7.5–12.3 vs 7.5–12.7 |
| Marker residual, 95th pct, median | 31.1 mm | 31.2 mm | range 25–49 vs 26–46 |
| Carry cycles kept | 73 (30.0 min) | 76 (31.1 min) | S01 6 vs 9, S06 8 vs 6, S07 4 vs 6; others equal |
| Joint angles, same frame | S02, S04, S05, S09, S11: median difference below 1° | | S01, S03, S07: hip flexion shifted 5–11° over the whole trial |
```

The S01 / S03 / S07 shifts appear although the two inputs share 98–99.9% of their samples. A small change in input moves the subject calibration and with it every joint angle of that subject: the same weakness the validation below points at, and the reason for the static-trial calibration. With it, the raw-take route keeps 63 carry cycles (25.9 min).

(problems)=
## Raw-take defects: wrong skeleton template, rotated head labels, labelling gaps

- **S01 was labelled with the wrong skeleton template.** Motive used a template with ISB landmark names (LIAS, LFAL, LHLE, ...) on the subject's Plug-in-Gait markers. An anatomy-based name map turns them back (`ISB_TO_PIG` in `mocap2myo/markerset.py`); the single elbow marker took either epicondyle name and is merged. The collaborators' export had renamed them the same way, which confirms the map.
- **Head labels rotated by 90°** in S01 and S10. The four head markers are now assigned by position in every trial.
- **S01 has no forearm markers** (LFRM, RFRM) in any source.
- **Large live-labelling gaps** in some markers: C7 in S02 (75–98% of frames missing) and S11 Solid (88%), RUPA in S06 (65–72%), RTHI in S08 (34–42%). The collaborators' export had these filled.
- **S01's right PSIS** is not rigid with the rest of the pelvis in either source (distance to the left PSIS varies by 35–42 mm; 2.5 mm in S03).
- **What the manual Motive processing adds** to a raw take is gap filling: in S03, the 5,765 samples the processed take has and the raw labels lack are in none of the unlabelled tracks. A few wrong labels were also removed (8 ASIS frames about 40 mm off).
- **S07 has no anatomy (static) trial** in the dataset.

(effort)=
## Run time and manual effort for 22 trials

Measured machine time for the 22 trials: reading the takes about 1 min, cleaning about 2 min, calibration plus IK about 50 min on 22 CPU cores, references, QA and clips about 1 min. Human work left: one marker-set configuration per new dataset, one reader check against a TRC export, and reviewing the report and the reference motions (in the viewer).

For comparison, a manual Motive plus OpenSim workflow is estimated (not measured) at roughly 15–40 person-hours for the same 22 trials: labelling and gap filling per trial, OpenSim scaling per subject, IK setup per trial. The automation relies on Motive's live labels; recordings without them need labelling in Motive or another source such as Xsens.
