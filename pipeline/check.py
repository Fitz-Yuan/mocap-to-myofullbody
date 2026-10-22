"""Pre-flight check of a dataset's settings: what the pipeline would read, before it runs.

  M2M_DATASET=datasets/<profile>.toml python pipeline/check.py      (mocap2myo check)

Lists the profile and its folders, the raw trial and static files it finds (per subject),
each subject's height and mass, the reader the input needs, and the first trial's content
(frames, rate, units, up axis, force plates) against the marker set: which of the set's
markers the trial has (its own labels or an alias set), which are missing, and labels it
does not know. Then the model. A problem that would stop a stage is marked "x" with what to
set; a doubtful point "!". Exit status 1 if there is a problem.
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402

problems, notes = [], []


def bad(msg):
  problems.append(msg)
  print(f"  x {msg}")


def warn(msg):
  notes.append(msg)
  print(f"  ! {msg}")


def ok(msg):
  print(f"  . {msg}")


def main():
  config.usage(__doc__)
  print(f"profile  {config.PROFILE or '(none: the case study defaults of pipeline/config.py)'}")
  print(f"root     {config.ROOT}")
  print("raw files")
  if not os.path.isdir(config.RAW_DIR):
    bad(f"no folder {config.RAW_DIR} (M2M_RAW_DIR / [paths] raw_dir under the root)")
  need = {"c3d": "ezc3d", "tak": "olefile", "b3d": "nimblephysics"}.get(config.INPUT)
  if need and importlib.util.find_spec(need) is None:
    bad(f"M2M_INPUT={config.INPUT} needs the {need} package (pip install {need})")
  trials, statics = {}, {}
  if not problems:
    trials, skipped = config.raw_trials()
    statics, _ = config.raw_statics()
    subj = {}
    for n in trials:
      subj.setdefault(config.subject_of(n), []).append(n)
    if not trials:
      bad(f"no trial files match {config.RAW_DIR}/{config.RAW_TRIALS} (M2M_RAW_TRIALS)")
    else:
      ok(f"{len(trials)} trials of {len(subj)} subjects, e.g. {list(trials)[:3]}")
    st_rel = {os.path.relpath(p, config.RAW_DIR).replace(os.sep, "/") for p in statics.values()}
    skipped = [x for x in skipped if x not in st_rel]  # the static trials are meant to be left out
    if skipped:
      warn(f"{len(skipped)} files match the glob but not M2M_TRIAL_REGEX, e.g. {skipped[:2]}")
    if config.STATIC:
      lack = sorted(set(subj) - set(statics))
      ok(f"static trials: {len(statics)}")
      if lack:
        warn(f"no static trial for {lack[:6]}: their calibration takes quiet standing from the trials")
    print("subjects")
    try:
      anth = config.anthropometrics() if os.path.exists(config.ANTHRO_CSV) else None
    except SystemExit as e:
      anth = None
      bad(str(e))
    if anth is None and config.ANTHRO == "csv":
      bad(f"no anthropometrics file {config.ANTHRO_CSV} (M2M_ANTHRO_CSV: subject, height cm, mass kg)")
    elif anth is None:
      ok(f"heights and masses from the files ({config.ANTHRO}); read_raw.py writes {config.ANTHRO_CSV}")
    else:
      miss = sorted(s for s in subj if s not in anth)
      if miss:
        bad(f"no height / mass for {miss[:6]} in {config.ANTHRO_CSV} (M2M_ANTHRO_SUBJECT names them)")
      else:
        ok(f"height and mass for all {len(subj)} subjects")
      odd = {s: v for s, v in anth.items() if not (120 < v["height_cm"] < 220 and 25 < v["mass_kg"] < 200)}
      if odd:
        warn(f"implausible values (height cm / mass kg): {dict(list(odd.items())[:4])}")
  if trials:
    print("first trial")
    name, path = next(iter(trials.items()))
    try:
      if config.INPUT == "tak":
        from mocap2myo import tak
        r = tak.read_tak(path)
        labels, rate, units, up, plates = list(r["names"]), r["rate"], "m", "y", None
      else:
        from mocap2myo import readers
        r = readers.read_trial(path, config.INPUT, up=config.UP, units=config.UNITS or None,
                               exclude=config.POINTS_EXCLUDE)
        labels, rate, units, up, plates = r["labels"], r["rate"], r["units"], r["up"], r["plates"]
        T = len(r["xyz"])
        ok(f"{name}: {T} frames at {rate:g} Hz ({T / rate:.1f} s), units {units}, up {up}")
      if plates is not None:
        ok(f"{plates['force'].shape[1]} force plates")
      check_markers(labels)
    except Exception as e:  # noqa: BLE001  (any reader error is a finding here)
      bad(f"{path}: {type(e).__name__}: {e}")
  print("model")
  try:
    p = config.model_profile()
    p.compile()
    ok(f"{p.name}: {p.title}")
  except (SystemExit, Exception) as e:  # noqa: BLE001
    bad(f"M2M_MODEL={config.MODEL}: {e} (models/fetch.py fetches the models not in the repository)")
  if config.TASK != "none":
    try:
      config.task()
      ok(f"task {config.TASK}")
    except SystemExit as e:
      bad(str(e))
  print(f"\n{len(problems)} problems, {len(notes)} notes")
  if problems:
    sys.exit(1)


def check_markers(labels):
  from mocap2myo import readers
  ms = config.marker_set()
  labels = [l for l in labels if l and not l.startswith(("*", "Unlabeled"))]
  own = [l for l in labels if l in ms.markers or l in ms.HEAD_LABELS]
  best, bname = None, None
  nbest = len(own)
  for an, a in config.marker_aliases(ms).items():  # as prepare_markers.choose_alias
    k = sum(l in a or l in ms.markers for l in labels)
    if k > nbest:
      best, bname, nbest = a, an, k
  mapped = {(best.get(l, l) if best else l) for l in labels}
  have = [l for l in ms.LABELS if l in mapped]
  miss = [l for l in ms.LABELS if l not in mapped]
  unknown = [l for l in labels if (best.get(l, l) if best else l) not in (*ms.LABELS, "HEAD")]
  frac = len(have) / len(ms.LABELS)
  msg = (f"marker set {ms.name}{f' (alias set {bname})' if bname else ''}: {len(have)} of "
         f"{len(ms.LABELS)} markers; missing {miss[:10]}{'...' if len(miss) > 10 else ''}")
  legs = {"pelvis", "femur_l", "femur_r", "tibia_l", "tibia_r", "foot_l", "foot_r", "toes_l", "toes_r"}
  leg_labels = [l for l in ms.LABELS if ms.segment(l) in legs]
  legs_only = frac < 0.8 and all(l in mapped for l in leg_labels if ms.markers[l].group != "loose") \
    and not any(l in mapped for l in ms.LABELS if ms.segment(l) not in legs and not ms.HEAD[ms.LABELS.index(l)])
  if legs_only:
    warn(msg + ": a lower-body recording; use a legs model (M2M_MODEL=myoleg or ms_human_700_locomotion), "
         "else the trunk and arms only follow the model's defaults")
  else:
    (ok if frac >= 0.8 else warn if frac >= 0.5 else bad)(msg + ("" if frac >= 0.5 else
                                                                " (another M2M_MARKER_SET? mocap2myo/markersets)"))
  if unknown:
    ok(f"{len(unknown)} labels not in the set (left out): {unknown[:8]}{'...' if len(unknown) > 8 else ''}")
  shared = sorted({l.rsplit(":", 1)[0] for l in labels if ":" in l and readers.strip_prefix(l) in ms.LABELS})
  if len(shared) > 1:
    bad(f"several people in this take ({', '.join(shared)}): their markers share names; leave all but one "
        f"out, e.g. points_exclude = '{'|'.join(x + ':.*' for x in shared[1:])}' in the profile (M2M_POINTS_EXCLUDE)")
  pel = [l for l in ms.PELVIS_LABELS if l not in mapped]
  if len(pel) > 1:
    bad(f"pelvis markers missing: {pel} (the IK needs at least 3 of {ms.PELVIS_LABELS})")


if __name__ == "__main__":
  main()
