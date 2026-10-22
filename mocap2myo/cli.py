"""mocap2myo: motion capture to musculoskeletal models, from the command line.

  mocap2myo init [PROFILE.toml]          write a dataset profile from a few questions
  mocap2myo check [--dataset P]          what the pipeline would read, before it runs
  mocap2myo run [--dataset P] [...]      the stages, raw files to joint loads
  mocap2myo export motion|c3d|viewer OUT motion clips for training, gap-filled C3D, viewer clips
  mocap2myo demo [DIR]                   one public walking trial end to end (~0.5 MB download)
  mocap2myo models [PROFILE ...]         fetch and build the models not in the repository

Settings come from the dataset profile (--dataset or M2M_DATASET) and M2M_* variables
(pipeline/config.py). The stages are the repository's scripts (pipeline/, tasks/), so the
command runs from a clone: git clone ...; pip install -e ".[c3d,tak]".
"""

import argparse
import io
import os
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


def cmd_check(args):
  _py("pipeline/check.py", _env(args))


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


def cmd_export(args):
  env = _env(args)
  if args.what == "motion":
    _py("pipeline/export_motion.py", env, [args.out, "--fps", str(args.fps), *args.trials])
  elif args.what == "c3d":
    _py("pipeline/export_c3d.py", env, [args.out, *args.trials])
  else:  # viewer: the model once, then one clip per trial
    _py("pipeline/export_viewer.py", env, ["model", args.out])
    for t in args.trials:
      _py("pipeline/export_viewer.py", env, ["clip", t, args.out])


def cmd_models(args):
  _py("models/fetch.py", dict(os.environ), args.rest)


# --- init: a dataset profile from questions ------------------------------------------------

def _ask(q, default=""):
  a = input(f"{q}{f' [{default}]' if default else ''}: ").strip()
  return a or default


def cmd_init(args):
  sys.path.insert(0, str(_repo()))
  from mocap2myo import markerset, readers
  out = Path(args.profile or "my_dataset.toml")
  root = Path(_ask("Folder of the recordings (trial files, possibly in one folder per subject)", ".")).resolve()
  kinds = {}
  for k in ("c3d", "trc", "tak", "b3d", "csv"):
    n = len(list(root.rglob(f"*.{k}")))
    if n:
      kinds[k] = n
  if not kinds:
    raise SystemExit(f"no .c3d / .trc / .tak / .b3d / .csv files under {root}")
  kind = max(kinds, key=kinds.get)
  print(f"  found {kinds}; input: {kind}")
  files = sorted(root.rglob(f"*.{kind}"))
  rel = [f.relative_to(root).as_posix() for f in files]
  by_folder = all("/" in r for r in rel)
  word = _ask("A word in the static (standing) trial file names, or 'none'", "static").lower()
  static = [r for r in rel if word != "none" and word in r.lower()]
  trials = [r for r in rel if r not in static]
  if static:  # the word as the files write it (globs and regexes are case-sensitive)
    import re
    word = re.search(re.escape(word), static[0], re.I).group(0)
  sample = files[rel.index(trials[0])]
  print(f"  {len(trials)} trial files, {len(static)} static; reading {trials[0]}")
  r = readers.read_trial(str(sample), kind, up="auto",
                         anthro=("SUBJECT:height", "SUBJECT:weight") if kind == "c3d" else None)
  labels = set(r["labels"])
  cover = {}
  for name in markerset.names():
    ms = markerset.load(name)
    al = [ms.markers.keys()] + [set(a) for a in ms.ALIASES.values()]
    cover[name] = max(len(labels & set(a)) for a in al) / max(1, len(ms.LABELS))
  mset = max(cover, key=cover.get)
  print(f"  {r['rate']:g} Hz, units {r['units']}, up {r['up']}; marker set {mset} "
        f"({100 * cover[mset]:.0f} % of its markers by name)")
  if cover[mset] < 0.5:
    print("  ! few markers match a known set: mocap2myo/markersets/ describes how to add yours")
  if by_folder:
    print("  the files are in one folder per subject: subject = folder name")
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
    csv = _ask("CSV with each subject's height (cm) and mass (kg)", "anthropometrics.csv")
    cols = _ask("Its columns: subject, height, mass", "Subject,Height,Weight")
    anthro = f'anthro = "csv"\nanthro_csv = "{csv}"\nanthro_columns = "{cols}"'
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

[subjects]
{anthro}
marker_set = "{mset}"

[task]
task = "none"

[ik]
static = true
static_markers = "legs"
'''
  out.write_text(text)
  print(f"\nwrote {out}\nnext: mocap2myo check --dataset {out}")


def main(argv=None):
  p = argparse.ArgumentParser(prog="mocap2myo", description=__doc__.split("\n\n")[0],
                              formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
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
  e.add_argument("trials", nargs="*", help="only these trials (viewer: the trials to export)")
  e.add_argument("--fps", type=float, default=100.0, help="motion: the clips' rate")
  e.set_defaults(fn=cmd_export)
  m = sub.add_parser("models", help="fetch / build the other models (models/fetch.py)")
  m.add_argument("rest", nargs=argparse.REMAINDER)
  m.set_defaults(fn=cmd_models)
  a = p.parse_args(argv)
  a.fn(a)


if __name__ == "__main__":
  main()
