"""mocap2myo: motion capture to musculoskeletal models, from the command line.

  mocap2myo init [PROFILE.toml]          write a dataset profile from a few questions
  mocap2myo check [--dataset P]          what the pipeline would read, before it runs
  mocap2myo run [--dataset P] [...]      the stages, raw files to joint loads
  mocap2myo export motion|c3d|viewer OUT motion clips for training, gap-filled C3D, viewer clips
  mocap2myo video VIDEO ... --subject S  motion from video (GVHMR, installed separately) into a study
  mocap2myo demo [DIR]                   one public walking trial end to end (~0.5 MB download)
  mocap2myo models [PROFILE ...]         fetch and build the models not in the repository

Settings come from the dataset profile (--dataset or M2M_DATASET) and M2M_* variables
(pipeline/config.py). The stages are the repository's scripts (pipeline/, tasks/), so the
command runs from a clone: git clone ...; pip install -e ".[c3d,tak]".
"""

import argparse
import io
import os
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGES = ["read", "markers", "ik", "reference", "qa", "dataset", "report", "kinetics", "muscles",
          "opensim"]
TAKES_TRIALS = {"read", "markers", "ik", "reference", "qa", "kinetics", "muscles"}


def _repo():
  if not (REPO / "pipeline" / "config.py").exists():
    raise SystemExit("mocap2myo: the pipeline's stages are not next to the package; install "
                     "from a clone of the repository (pip install -e .)")
  return REPO


def _env(args):
  env = dict(os.environ)
  env.setdefault("OMP_NUM_THREADS", "1")
  if getattr(args, "dataset", None):
    env["M2M_DATASET"] = str(Path(args.dataset).resolve())
  if getattr(args, "root", None):
    env["M2M_ROOT"] = str(Path(args.root).resolve())
  if getattr(args, "gpu", False):
    env["M2M_IK_DEVICE"] = "gpu"
  if getattr(args, "workers", None):
    env["IK_WORKERS"] = str(args.workers)
  if not env.get("M2M_DATASET"):  # the case study's defaults (run_all.sh)
    env.setdefault("M2M_STATIC", "1")
    env.setdefault("M2M_STATIC_MARKERS", "legs")
  return env


def _py(script, env, argv=(), python=None):
  cmd = [python or sys.executable, "-u", str(_repo() / script), *argv]
  r = subprocess.run(cmd, env=env, cwd=_repo())
  if r.returncode:
    raise SystemExit(f"mocap2myo: {script} failed (exit {r.returncode})")


def _task(env):
  out = subprocess.run([sys.executable, "-c", "from pipeline import config; config.task(); print(config.TASK)"],
                       env=env, cwd=_repo(), capture_output=True, text=True)
  if out.returncode:
    raise SystemExit(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "mocap2myo: no task")
  return out.stdout.strip().splitlines()[-1]


def _solved(env):
  """The trials the IK has solved (M2M_IK_DIR)."""
  out = subprocess.run([sys.executable, "-c", "from pipeline import config; print(chr(10).join(config.trials(config.IK_DIR)))"],
                       env=env, cwd=_repo(), capture_output=True, text=True)
  if out.returncode:
    raise SystemExit(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "mocap2myo: no settings")
  return [t for t in out.stdout.splitlines() if t.strip()]


def cmd_check(args):
  r = subprocess.run([sys.executable, "-u", str(_repo() / "pipeline/check.py")], env=_env(args), cwd=_repo())
  if r.returncode:  # the problems are listed above
    raise SystemExit(1)


def cmd_run(args):
  env = _env(args)
  task = _task(env)
  first = STAGES.index(args.start) if args.start else 0
  only = set(args.only.split(",")) if args.only else None
  trials = list(args.trials or [])
  plan = []
  for st in STAGES[first:]:
    if only is not None and st not in only:
      continue
    if st == "muscles" and not args.muscles or st == "opensim" and not args.opensim:
      continue
    if st in ("reference", "dataset"):
      script = f"tasks/{task}/build_{'reference' if st == 'reference' else 'dataset'}.py"
      if not (_repo() / script).exists():
        continue
    plan.append(st)
  print(f"mocap2myo run: {' -> '.join(plan)} (task {task})", flush=True)
  scripts = {"read": "pipeline/read_raw.py", "markers": "pipeline/prepare_markers.py",
             "ik": "pipeline/solve_ik.py", "reference": f"tasks/{task}/build_reference.py",
             "qa": "pipeline/qa.py", "dataset": f"tasks/{task}/build_dataset.py",
             "report": "pipeline/report.py", "kinetics": "pipeline/kinetics.py",
             "muscles": "pipeline/muscles.py"}
  for st in plan:
    print(f"\n== {st}", flush=True)
    if st == "opensim":
      osim = args.osim_python or sys.executable
      _py("pipeline/validate/opensim_ik.py", env, python=osim)
      _py("pipeline/opensim_id.py", env, python=osim)
      _py("pipeline/validate/opensim_compare.py", env)
      _py("pipeline/validate/kinetics_compare.py", env)
      continue
    _py(scripts[st], env, trials if st in TAKES_TRIALS else ())
  print("\nmocap2myo run: done", flush=True)


# --- demo: one public walking trial ------------------------------------------------------

DEMO_ZIP = "https://ndownloader.figshare.com/files/22295019"  # Moissenet & Schreiber 2019, CC BY 4.0
DEMO_DIR_IN_ZIP = "A multimodal dataset of human gait at different walking speeds"
DEMO_FILES = ("2014001/2014001_ST.c3d", "2014001/2014001_C3_03.c3d")


class _HttpRange(io.RawIOBase):
  """A remote file read by HTTP range requests (a few members of a large zip)."""

  def __init__(self, url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
      self.url, self.size = r.geturl(), int(r.headers["Content-Range"].rsplit("/", 1)[1])
    self.pos = 0

  def readable(self):
    return True

  def seekable(self):
    return True

  def tell(self):
    return self.pos

  def seek(self, off, whence=0):
    self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
    return self.pos

  def readinto(self, b):
    if self.pos >= self.size:
      return 0
    end = min(self.pos + len(b), self.size) - 1
    req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
      data = r.read()
    b[:len(data)] = data
    self.pos += len(data)
    return len(data)


def cmd_demo(args):
  root = Path(args.dir).resolve()
  raw = root / "raw" / DEMO_DIR_IN_ZIP
  todo = [f for f in DEMO_FILES if not (raw / f).exists()]
  if todo:
    print(f"downloading {len(todo)} files of Moissenet & Schreiber 2019 (CC BY 4.0, figshare 7734767)")
    z = zipfile.ZipFile(io.BufferedReader(_HttpRange(DEMO_ZIP), 1 << 20))
    for f in todo:
      (raw / f).parent.mkdir(parents=True, exist_ok=True)
      (raw / f).write_bytes(z.read(f"{DEMO_DIR_IN_ZIP}/{f}"))
      print(f"  {f} ({(raw / f).stat().st_size // 1024} kB)")
  args.dataset, args.root = str(_repo() / "datasets" / "moissenet2019.toml"), str(root)
  args.start = args.only = None
  args.trials, args.muscles, args.opensim, args.osim_python = [], False, False, None
  args.gpu, args.workers = False, None
  cmd_run(args)
  env = _env(args)
  out = root / "viewer"
  _py("pipeline/export_viewer.py", env, ["clip", "2014001_C3_03", str(out)])
  print(f"\nThe result: {root}/ik (joint angles), {root}/kin (joint loads), {out} (a clip: drop the "
        ".elvw file on https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/ or on a local "
        "docs/viewer/index.html)")


def cmd_video(args):
  sys.path.insert(0, str(_repo()))
  from mocap2myo import video
  if args.result and len(args.video) != 1:
    raise SystemExit("--result: one video at a time")
  if not args.height:
    print("! no --height: the size comes from the video, which fixes it only roughly", flush=True)
  prof = None
  for v in args.video:
    prof = video.add_video(v, args.subject, args.mass, args.out, height_cm=args.height, gvhmr=args.gvhmr,
                           python=args.gvhmr_python, static_cam=args.static_cam, result=args.result,
                           floor=args.floor, lock=not args.no_foot_lock)
  print(f"\nnext: mocap2myo check --dataset {prof}\n      mocap2myo run --dataset {prof}")


def cmd_export(args):
  env = _env(args)
  out = str(Path(args.out).resolve())  # the scripts run in the repository's folder
  solved = _solved(env)
  if not solved:
    raise SystemExit("no trials solved yet: mocap2myo run first")
  bad = [t for t in args.trials if t not in solved]
  if bad:
    raise SystemExit(f"{bad}: not solved (trials: {solved[:8]}{' ...' if len(solved) > 8 else ''})")
  if args.what == "motion":
    _py("pipeline/export_motion.py", env, [out, "--fps", str(args.fps), "--min-s", str(args.min_s), *args.trials])
  elif args.what == "c3d":
    _py("pipeline/export_c3d.py", env, [out, *args.trials])
  else:  # viewer: the model once, then one clip per trial
    trials = args.trials or solved
    _py("pipeline/export_viewer.py", env, ["model", out])
    for t in trials:
      _py("pipeline/export_viewer.py", env, ["clip", t, out])
    print(f"\nin {out}: drop a trial's .elvw on the viewer (https://fitz-yuan.github.io/mocap-to-myofullbody/viewer/); "
          "it brings the eight built-in models itself (another model: drop its *_model.elvw too)")


def cmd_models(args):
  _py("models/fetch.py", dict(os.environ), args.rest)


# --- init: a dataset profile from questions ------------------------------------------------

def _ask(q, default=""):
  try:
    a = input(f"{q}{f' [{default}]' if default else ''}: ").strip()
  except EOFError:
    raise SystemExit("\nmocap2myo init asks a few questions: run it in a terminal, or copy a profile "
                     "from datasets/ and edit it") from None
  return a or default


def cmd_init(args):
  sys.path.insert(0, str(_repo()))
  from mocap2myo import markerset, readers
  out = Path(args.profile or "my_dataset.toml")
  root = Path(_ask("Folder of the recordings (trial files, possibly in one folder per subject)", ".")).resolve()
  kinds = {}
  def found(k):  # QTM: only the 3D files are trials (plates, analog: files beside them)
    return [f for f in root.rglob(f"*.{k}") if k != "tsv" or readers.qtm_tsv_content(str(f)) == "3D"]

  for k in ("c3d", "trc", "tak", "b3d", "csv", "tsv"):
    n = len(found(k))
    if n:
      kinds[k] = n
  if not kinds:
    raise SystemExit(f"no .c3d / .trc / .tak / .b3d / .csv / .tsv files under {root}")
  kind = max(kinds, key=kinds.get)
  print(f"  found {kinds}; input: {kind}")
  files = sorted(found(kind))
  rel = [f.relative_to(root).as_posix() for f in files]
  by_folder = all("/" in r for r in rel)
  word = _ask("A word in the static (standing) trial file names, or 'none'", "static").lower()
  static = [r for r in rel if word != "none" and word in r.lower()]
  trials = [r for r in rel if r not in static]
  if static:  # the word as the files write it (globs and regexes are case-sensitive)
    word = re.search(re.escape(word), static[0], re.I).group(0)
  sample = files[rel.index(trials[0])]
  print(f"  {len(trials)} trial files, {len(static)} static; reading {trials[0]}")
  r = readers.read_trial(str(sample), kind, up="auto",
                         anthro=("SUBJECT:height", "SUBJECT:weight") if kind == "c3d" else None)
  cover = {name: markerset.coverage(markerset.load(name), r["labels"]) for name in markerset.names()}
  mset = max(cover, key=lambda k: len(cover[k]["have"]) / len(markerset.load(k).LABELS))
  c, n_set = cover[mset], len(markerset.load(mset).LABELS)
  alias_note = f" (names of its alias set {c['alias']})" if c["alias"] else ""
  print(f"  {r['rate']:g} Hz, units {r['units']}, up {r['up']}; marker set {mset}: {len(c['have'])} of "
        f"{n_set} markers{alias_note}")
  model = ""
  if c["legs_only"]:
    print('  a lower-body recording: the profile uses a legs model, model = "myoleg" '
          "(fetch it once: mocap2myo models myoleg)")
    model = 'model = "myoleg"                 # a lower-body recording (mocap2myo models myoleg)\n'
  elif len(c["have"]) < 0.5 * n_set:
    print("  ! few of the set's markers are found: if your marker set is not built in, "
          "mocap2myo/markersets/ describes how to add it")
  if by_folder:
    print("  the files are in one folder per subject: subject = folder name")
    seps = {Path(t).stem[len(Path(t).parent.name):][:1] for t in trials}
    if all(Path(t).stem.startswith(Path(t).parent.name) for t in trials) and len(seps) == 1 and seps <= {"_", "-", "."}:
      sep = seps.pop()  # S01/S01_walk01.c3d: the trial is S01_walk01
      trial_regex = (rf"(?P<subject>[^/]+)/(?P=subject){re.escape(sep)}(?!.*{word})[^/]+\.{kind}" if static
                     else rf"(?P<subject>[^/]+)/(?P=subject){re.escape(sep)}[^/]+\.{kind}")
      trial_name = "{stem}"
    else:  # P01/walk01.c3d: the trial is P01-walk01
      sep = "-"
      trial_regex = rf"(?P<subject>[^/]+)/(?!.*{word})(?P<trial>[^/]+)\.{kind}" if static else \
        rf"(?P<subject>[^/]+)/(?P<trial>[^/]+)\.{kind}"
      trial_name = "{subject}-{trial}"
    static_regex = rf"(?P<subject>[^/]+)/[^/]*{word}[^/]*\.{kind}"
  else:
    sep = _ask("Subject = the file name up to which character", "_")
    stems = [Path(t).stem for t in trials[:5]]
    print(f"  subjects of {stems}: {[s.split(sep)[0] for s in stems]}")
    trial_regex = rf"(?!.*{word}).*\.{kind}" if static else ""
    trial_name = "{stem}"
    static_regex = rf"(?P<subject>[^{sep}/]+){sep}?[^/]*{word}[^/]*\.{kind}"
  anth = r.get("anthro") or {}
  if anth.get("height_m") and anth.get("mass_kg"):
    anthro = 'anthro = "c3d"                   # SUBJECT:height / SUBJECT:weight of the C3D files'
  else:
    base = root.parent if by_folder else root
    csv = _ask(f"CSV with each subject's height (cm) and mass (kg), relative to {base}", "anthropometrics.csv")
    cols = "Subject,Height,Weight"
    anthro = f'anthro = "csv"\nanthro_csv = "{csv}"                # columns Subject, Height (cm), Weight (kg)\nanthro_columns = "{cols}"'
    path = base / csv
    if not path.exists():
      subjects = sorted({Path(t).parent.name for t in trials} if by_folder else
                        {Path(t).stem.split(sep)[0] for t in trials})
      path.write_text(cols + "\n" + "".join(f"{x},,\n" for x in subjects))
      print(f"  wrote {path}: fill in each subject's height (cm) and mass (kg)")
  raw_static = (("*/" if by_folder else "") + f"*{word}*.{kind}") if static else f"none/*.{kind}"
  if not static:
    static_regex = r"none/(?P<subject>[^/]+)\." + kind
  text = f'''# Dataset profile written by "mocap2myo init"; see datasets/*.toml for documented examples.
name = "{out.stem}"

[paths]
root = "{root.parent.as_posix() if by_folder else root.as_posix()}"
raw_dir = "{root.name if by_folder else "."}"

[input]
input = "{kind}"
up = "auto"
units = ""
rate = "native"

[naming]
raw_trials = "{'*/' if by_folder else ''}*.{kind}"
trial_regex = '{trial_regex}'
trial_name = "{trial_name}"
trial_glob = "*"
subject_sep = "{sep}"
raw_static = "{raw_static}"
static_regex = '{static_regex}'
static_name = "{{subject}}{sep}static"

{model}
[subjects]
{anthro}
marker_set = "{mset}"

[task]
task = "none"

[ik]
static = true                      # the static trial; without one, quiet standing in the trials
static_markers = "legs"
'''
  out.write_text(text)
  print(f"\nwrote {out}\nnext: mocap2myo check --dataset {out}")


def main(argv=None):
  head, _, rest = __doc__.partition("\n\n")
  p = argparse.ArgumentParser(prog="mocap2myo", description=head,
                              formatter_class=argparse.RawDescriptionHelpFormatter, epilog=rest)
  sub = p.add_subparsers(dest="cmd", required=True)
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--dataset", help="dataset profile (datasets/*.toml); default M2M_DATASET")
  common.add_argument("--root", help="the data folder (M2M_ROOT); default the profile's")
  sub.add_parser("check", parents=[common], help="pre-flight check").set_defaults(fn=cmd_check)
  r = sub.add_parser("run", parents=[common], help="run the stages")
  r.add_argument("trials", nargs="*", help="only these trials (default all)")
  r.add_argument("--from", dest="start", choices=STAGES, help="start at this stage")
  r.add_argument("--only", help="only these stages, comma separated")
  r.add_argument("--gpu", action="store_true", help="IK on the GPU (M2M_IK_DEVICE=gpu)")
  r.add_argument("--workers", type=int, help="parallel processes (IK_WORKERS, default 12)")
  r.add_argument("--muscles", action="store_true", help="also muscle forces (static optimisation)")
  r.add_argument("--opensim", action="store_true", help="also the OpenSim comparison (needs opensim)")
  r.add_argument("--osim-python", help="a Python with the opensim package")
  r.set_defaults(fn=cmd_run)
  d = sub.add_parser("demo", help="one public walking trial end to end")
  d.add_argument("dir", nargs="?", default="mocap2myo_demo")
  d.set_defaults(fn=cmd_demo)
  i = sub.add_parser("init", help="write a dataset profile")
  i.add_argument("profile", nargs="?")
  i.set_defaults(fn=cmd_init)
  e = sub.add_parser("export", parents=[common], help="motion clips for training, gap-filled C3D, viewer clips")
  e.add_argument("what", choices=["motion", "c3d", "viewer"])
  e.add_argument("out", help="output folder")
  e.add_argument("trials", nargs="*", help="only these trials (default: every solved trial)")
  e.add_argument("--fps", type=float, default=100.0, help="motion: the clips' rate")
  e.add_argument("--min-s", type=float, default=1.0, help="motion: the shortest clip kept (s)")
  e.set_defaults(fn=cmd_export)
  v = sub.add_parser("video", help="motion from video (GVHMR, installed separately) into a study folder")
  v.add_argument("video", nargs="+", help="video file(s) of one person")
  v.add_argument("--subject", required=True, help="the person's name or code (letters, digits, '-')")
  v.add_argument("--mass", type=float, required=True, help="body mass (kg)")
  v.add_argument("--height", type=float, help="body height (cm); without it, the video's estimate")
  v.add_argument("--out", required=True, help="the study folder (created; more videos and people can be added)")
  v.add_argument("--gvhmr", default=os.environ.get("M2M_GVHMR"), help="GVHMR's folder (default M2M_GVHMR)")
  v.add_argument("--gvhmr-python", default=os.environ.get("M2M_GVHMR_PYTHON", "python"),
                 help="the Python of GVHMR's environment (default M2M_GVHMR_PYTHON, else python)")
  v.add_argument("--static-cam", action="store_true", help="the camera did not move (tripod)")
  v.add_argument("--floor", choices=["flat", "as-is"], default="flat",
                 help="flat: level ground, the reconstruction's slow vertical drift removed (default); "
                      "as-is: its heights kept (stairs, slopes)")
  v.add_argument("--no-foot-lock", action="store_true", help="keep the reconstruction's foot sliding")
  v.add_argument("--result", help="an existing GVHMR hmr4d_results.pt for the video (GVHMR is not run again)")
  v.set_defaults(fn=cmd_video)
  m = sub.add_parser("models", help="fetch / build the other models (models/fetch.py)")
  m.add_argument("rest", nargs=argparse.REMAINDER)
  m.set_defaults(fn=cmd_models)
  a = p.parse_args(argv)
  a.fn(a)


if __name__ == "__main__":
  main()
