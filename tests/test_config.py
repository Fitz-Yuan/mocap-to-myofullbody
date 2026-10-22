"""Every dataset profile in datasets/ loads; the readers' kind detection."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mocap2myo import readers

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("profile", sorted((REPO / "datasets").glob("*.toml")), ids=lambda p: p.stem)
def test_profile_loads(profile, tmp_path):
  env = dict(os.environ, M2M_DATASET=str(profile), M2M_ROOT=str(tmp_path))
  code = "from pipeline import config; print(config.INPUT, config.MARKER_SET, config.TASK)"
  r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
  assert r.returncode == 0, r.stderr


def test_kind_of():
  assert readers.kind_of("a/b.C3D") == "c3d"
  assert readers.kind_of("s/s.b3d#Gait_1_segment_0") == "b3d"
  with pytest.raises(ValueError):
    readers.kind_of("x.mat")


def test_cli_help():
  r = subprocess.run([sys.executable, "-m", "mocap2myo.cli", "--help"], cwd=REPO, capture_output=True, text=True)
  assert r.returncode == 0 and "mocap2myo" in r.stdout
