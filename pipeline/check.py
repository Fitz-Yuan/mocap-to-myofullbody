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
import warnings

import numpy as np

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
    anth, err = None, False
    try:
      anth = config.anthropometrics() if os.path.exists(config.ANTHRO_CSV) else None
    except SystemExit as e:
      err = True
      bad(str(e))
    if err:
      pass
    elif anth is None and config.ANTHRO == "csv":
      bad(f"no anthropometrics file {config.ANTHRO_CSV} (M2M_ANTHRO_CSV: subject, height cm, mass kg)")
    elif anth is None:
      ok(f"heights and masses from the files ({config.ANTHRO}); read_raw.py writes {config.ANTHRO_CSV}")
    else:
      miss = sorted(s for s in subj if s not in anth)
      if miss:
        bad(f"no height (cm) / mass (kg) for {miss[:6]} in {config.ANTHRO_CSV}")
      else:
        ok(f"height and mass for all {len(subj)} subjects")
      metres = {s: v["height_cm"] for s, v in anth.items() if v["height_cm"] < 3}
      if metres:
        bad(f"heights in metres? {dict(list(metres.items())[:4])}: {config.ANTHRO_CSV} needs centimetres")
      odd = {s: v for s, v in anth.items() if s not in metres and not (120 < v["height_cm"] < 220 and 25 < v["mass_kg"] < 200)}
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
        if rate > 150 and config.RATE is None:
          k = max(1, round(rate / 110))
          warn(f"{rate:g} Hz: the IK solves every frame, about 0.1 s each on one core (~{T * 0.1 / 60:.0f} min "
               f"for this trial); rate = {rate / k:g} in the profile's [input] keeps every {k}th frame")
        X = r["xyz"][:, [not l.startswith(("*", "Unlabeled")) for l in labels]]  # the labelled markers
        with np.errstate(all="ignore"), warnings.catch_warnings():
          warnings.simplefilter("ignore", RuntimeWarning)
          size = float(np.nanmedian(np.nanmax(np.nanmax(X, 1) - np.nanmin(X, 1), 1)))
        if not 0.25 < size < 4.0:  # a person spans about 0.3-2.5 m
          bad(f"the markers span {size:.3g} m: the length unit is probably not {units!r}; "
              "set units = 'mm' or 'm' in the profile (M2M_UNITS)")
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
    bad(f"M2M_MODEL={config.MODEL}: {e} (fetch it once: mocap2myo models {config.MODEL})")
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
  from mocap2myo import markerset, readers
  ms = config.marker_set()
  labels = [l for l in labels if l and not l.startswith(("*", "Unlabeled"))]
  c = markerset.coverage(ms, labels, config.marker_aliases(ms))
  have, miss, unknown, bname = c["have"], c["missing"], c["unknown"], c["alias"]
  mapped = set(have)
  frac = len(have) / len(ms.LABELS)
  msg = (f"marker set {ms.name}{f' (alias set {bname})' if bname else ''}: {len(have)} of "
         f"{len(ms.LABELS)} markers; missing {miss[:10]}{'...' if len(miss) > 10 else ''}")
  if c["legs_only"]:
    try:
      legs_model = config.model_profile().segment_body("humerus_r") is None
    except (SystemExit, Exception):  # noqa: BLE001  (the model is checked below)
      legs_model = False
    if legs_model:
      ok(msg + f": a lower-body recording, on a legs model ({config.MODEL})")
    else:
      warn(msg + ': a lower-body recording; use a legs model (model = "myoleg" in the profile; fetch it '
           "once: mocap2myo models myoleg), else the trunk and arms only follow the model's defaults")
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
