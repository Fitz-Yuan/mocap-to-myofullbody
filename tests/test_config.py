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


def test_anthropometrics_csv_headers(tmp_path):
  """Headers with units or another case are read; the sex column (segment masses) is kept."""
  (tmp_path / "a.csv").write_text("subject,Height (cm),Mass [kg],Sex\nS01,175,70,F\nS02,,,M\n")
  env = dict(os.environ, M2M_DATASET=str(REPO / "datasets" / "moissenet2019.toml"), M2M_ROOT=str(tmp_path),
             M2M_ANTHRO="csv", M2M_ANTHRO_CSV=str(tmp_path / "a.csv"), M2M_SEX_COLUMN="Sex")
  code = "from pipeline import config; print(config.anthropometrics(), config.sexes())"
  r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
  assert r.returncode == 0, r.stderr
  assert "{'S01': {'height_cm': 175.0, 'mass_kg': 70.0}}" in r.stdout  # S02's blank cells: not given yet
  assert "'S01': 'f'" in r.stdout and "'S02': 'm'" in r.stdout
