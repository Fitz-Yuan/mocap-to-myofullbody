"""Fetch the models that are not in this repository, and build their MuJoCo files.

  python models/fetch.py [PROFILE ...] [--myoconverter-python PATH] [--no-build] [--force]

Into M2M_MODELS_DIR (default ~/data/mocap2myo/models):
  <source>/src/          sparse git checkout of an upstream repository at its pinned commit
                         (mocap2myo/profiles/sources.toml: URL, commit, licence, paths)
  <source>/SOURCE.json   what was fetched (URL, commit, licence, paths, date)
  <profile>/             a built model (when the profile has a [build] step), BUILD.json
                         (inputs, command, tool versions, date)

Build kinds ([build] kind in the profile):
  myo_sim       compose a MyoSim model with the repository's own builder
                (myo_sim.build.compose.build_spec + write_spec_xml), e.g. "myolegs";
  mass          models/mass_to_mjcf.py: MASS's DART skeleton + muscle file -> MJCF;
  myoconverter  models/osim_to_mjcf.py: an OpenSim model -> MJCF with MyoConverter. Needs
                a Python with MyoConverter's dependencies (opensim, mujoco 2.3.7, ...;
                see the fetched myoconverter/src/conda_env.yml):
                --myoconverter-python or M2M_MYOCONVERTER_PY.

Profiles without [build] use a fetched file directly (MS-Human-700, the pre-converted
Gait2354). Code from the fetched repositories runs only in the build steps that need it
(myo_sim's composer, MyoConverter).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mocap2myo import profile as prof  # noqa: E402

HERE = Path(__file__).resolve().parent


def run(cmd, **kw):
  print("  $", " ".join(str(c) for c in cmd))
  subprocess.run([str(c) for c in cmd], check=True, **kw)


def git_head(d: Path) -> str | None:
  try:
    return subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
  except (subprocess.CalledProcessError, FileNotFoundError):
    return None


def fetch_source(key: str, src: dict, force=False) -> Path:
  root = prof.MODELS_DIR / key
  dest = root / "src"
  if dest.exists() and git_head(dest) == src["commit"] and not force:
    print(f"{key}: at {src['commit'][:12]} already")
    return dest
  print(f"{key}: fetching {src['repo']} @ {src['commit'][:12]}")
  root.mkdir(parents=True, exist_ok=True)
  if not dest.exists():
    run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", src["repo"], dest])
  run(["git", "-C", dest, "sparse-checkout", "init", "--no-cone"])
  pats = ["/" + p.lstrip("/") for p in src["paths"]]
  run(["git", "-C", dest, "sparse-checkout", "set", "--no-cone", *pats])
  run(["git", "-C", dest, "fetch", "--quiet", "--filter=blob:none", "origin", src["commit"]])
  run(["git", "-C", dest, "checkout", "--quiet", "--detach", src["commit"]])
  info = dict(key=key, repo=src["repo"], commit=git_head(dest), license=src["license"], paths=src["paths"],
              fetched=datetime.datetime.now().isoformat(timespec="seconds"))
  (root / "SOURCE.json").write_text(json.dumps(info, indent=1) + "\n")
  return dest


def build(p: prof.Profile, myoconverter_py: str | None, force=False):
  b = p.data.get("build")
  if not b:
    return
  out = p.xml
  if out.exists() and not force:
    print(f"{p.name}: {out} exists (--force to rebuild)")
    return
  out.parent.mkdir(parents=True, exist_ok=True)
  kind = b["kind"]
  info = dict(profile=p.name, kind=kind, output=str(out), date=datetime.datetime.now().isoformat(timespec="seconds"),
              inputs={k: v for k, v in b.items() if k != "kind"})
  if kind == "myo_sim":
    src = prof.MODELS_DIR / b["source"] / "src"
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from myo_sim.build.compose import build_spec, write_spec_xml; "
            "write_spec_xml(build_spec(sys.argv[2]), Path(sys.argv[3])); "
            "import mujoco; print('mujoco', mujoco.__version__)")
    cmd = [sys.executable, "-I", "-c", code, src, b["model"], out]
  elif kind == "mass":
    src = prof.MODELS_DIR / b["source"] / "src"
    cmd = [sys.executable, HERE / "mass_to_mjcf.py", "--check", src / "data" / "human.xml",
           src / "data" / "muscle284.xml", src / "data" / "OBJ", out]
  elif kind == "myoconverter":
    py = myoconverter_py or os.environ.get("M2M_MYOCONVERTER_PY")
    if not py:
      raise SystemExit(f"{p.name}: needs --myoconverter-python (a Python with MyoConverter's dependencies)")
    conv = prof.MODELS_DIR / "myoconverter" / "src"
    osim = prof.resolve_path(b["osim"])
    geom = prof.resolve_path(b["geometry"])
    cmd = [py, "-I", HERE / "osim_to_mjcf.py", "--myoconverter", conv, "--steps", ",".join(map(str, b.get("steps", [1]))),
           osim, geom, out]
  else:
    raise SystemExit(f"{p.name}: unknown build kind {kind!r}")
  info["command"] = [str(c) for c in cmd]
  print(f"{p.name}: building ({kind})")
  run(cmd)
  if not out.exists():
    raise SystemExit(f"{p.name}: build did not write {out}")
  (out.parent / f"BUILD_{p.name}.json").write_text(json.dumps(info, indent=1) + "\n")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("names", nargs="*", help="profiles (default: all that need fetching)")
  ap.add_argument("--myoconverter-python", help="Python with MyoConverter's dependencies")
  ap.add_argument("--no-build", action="store_true")
  ap.add_argument("--force", action="store_true", help="fetch / build again")
  a = ap.parse_args()
  srcs = prof.sources()
  names = a.names or [n for n in prof.profile_names() if prof.load(n).data.get("sources")]
  print(f"models folder: {prof.MODELS_DIR}")
  for n in names:
    p = prof.load(n)
    for key in p.data.get("sources", []):
      fetch_source(key, srcs[key], a.force)
    if not a.no_build:
      build(p, a.myoconverter_python, a.force)
    print(f"{n}: {p.xml} {'ok' if p.xml.exists() else 'MISSING'}")


if __name__ == "__main__":
  main()
