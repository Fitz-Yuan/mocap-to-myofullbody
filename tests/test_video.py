"""Video input: the post-processing of GVHMR's virtual markers (no GVHMR needed)."""
import re

import numpy as np

from mocap2myo import gvhmr_markers, markerset, video


def _npz(tmp_path, T=300, drift=0.0, skate=0.0):
  """A person standing still (Y up, as GVHMR writes) whose reconstruction drifts upwards and
  skates forwards at constant speed: neither is real."""
  names = list(gvhmr_markers.MARKERS)
  rng = np.random.default_rng(0)
  S = rng.uniform(-0.2, 0.2, (len(names), 3))
  S[:, 1] = rng.uniform(0.1, 1.5, len(names))  # heights (Y up)
  for n in ("l_calc", "r_calc", "l_toe", "r_toe"):
    S[names.index(n), 1] = 0.04
  t = np.arange(T) / video.FPS
  X = np.repeat(S[None], T, 0)
  X[..., 1] += drift * t[:, None] / t[-1]
  X[..., 2] += skate * t[:, None]  # +z in GVHMR's world = -y after the turn to Z up
  low = np.stack([np.zeros(T), drift * t / t[-1], np.zeros(T)], 1)
  p = tmp_path / "m.npz"
  np.savez(p, names=np.array(names), markers=X, lowest=low, static=S, static_lowest=np.zeros(3),
           stature=1.5, betas=np.zeros(10))
  return str(p), names, S


def test_names_match_the_marker_set():
  assert list(gvhmr_markers.MARKERS) == markerset.load("smpl_opencap").LABELS


def test_scale_floor_and_static(tmp_path):
  p, names, S = _npz(tmp_path, drift=0.2)
  n, X, St, info = video.to_trials(p, height_cm=180, lock=False)
  assert n == names and info["scale"] == 1.2 and info["height_m"] == 1.8
  heel = X[:, names.index("l_calc"), 2]
  assert np.allclose(heel, 0.04 * 1.2, atol=0.01)  # the drift is gone, the heel stays on the floor
  assert np.allclose(St[names.index("l_calc"), 2], 0.04 * 1.2)
  pel = St[[names.index(k) for k in ("l_ASIS", "r_ASIS", "l_PSIS", "r_PSIS")], :2].mean(0)
  assert np.allclose(pel, 0, atol=1e-9)


def test_foot_lock_stops_a_skating_stance(tmp_path):
  p, names, _ = _npz(tmp_path, skate=0.3)
  _, X, _, info = video.to_trials(p, lock=True)
  assert info["slide_before_m_s"] > 0.25 and info["slide_after_m_s"] < 0.02
  mid = X[30:-30, names.index("l_calc"), :2]
  assert np.ptp(mid, 0).max() < 0.02  # the planted heel stays put


def test_profile_finds_trials_not_the_static():
  prof = video.PROFILE.format(name="v", root="/x")
  rx = re.search(r"trial_regex = '(.+)'", prof).group(1)
  srx = re.search(r"static_regex = '(.+)'", prof).group(1)
  assert re.fullmatch(rx, "P01/P01_walk-2.trc")
  assert not re.fullmatch(rx, "P01/P01_static.trc")
  assert re.fullmatch(srx, "P01/P01_static.trc").group("subject") == "P01"
