"""Short-gap filling of mocap2myo.ik.load_trial."""
from types import SimpleNamespace

import numpy as np
import pytest

from mocap2myo import ik


def _trial(tmp_path, X, rate=100.0):
  p = tmp_path / "t.npz"
  np.savez(p, markers=X, marker_labels=np.array(["A"]), marker_rate=rate, height_m=1.75)
  mm = SimpleNamespace(ms=SimpleNamespace(LABELS=["A"]), model_height=1.75, scaling=None)
  return p, mm


def _cubic(T, rate=100.0):
  t = np.arange(T) / rate
  return np.stack([0.3 * t ** 3 - t ** 2 + 0.5 * t, 0.1 * t ** 2, 0.2 + 0.0 * t], 1)[:, None]


@pytest.mark.parametrize("method", ["spline", "linear"])
def test_short_gap_filled(tmp_path, method):
  X = _cubic(300)
  G = X.copy()
  G[100:115] = np.nan  # 150 ms
  P, rate, s = ik.load_trial(*_gap_args(tmp_path, G), gap_fill=method)
  assert s == 1.0 and rate == 100.0
  assert np.isfinite(P[100:115]).all()
  err = np.abs(P[100:115] - X[100:115]).max()
  if method == "spline":
    assert err < 1e-9  # a cubic spline reproduces a cubic
  else:
    assert err > 1e-5  # a line does not


def _gap_args(tmp_path, X):
  p, mm = _trial(tmp_path, X)
  return str(p), 0.2, None, mm


def test_long_and_edge_gaps_stay_missing(tmp_path):
  X = _cubic(300)
  X[50:80] = np.nan    # 300 ms > 200 ms
  X[:5] = np.nan       # not seen before the gap
  X[-5:] = np.nan      # not seen after
  P, _, _ = ik.load_trial(*_gap_args(tmp_path, X))
  assert np.isnan(P[50:80]).all() and np.isnan(P[:5]).all() and np.isnan(P[-5:]).all()


def test_spline_falls_back_to_linear_with_one_sample(tmp_path):
  X = _cubic(300)
  X[:99] = np.nan
  X[100:110] = np.nan  # one sample before the gap: linear
  P, _, _ = ik.load_trial(*_gap_args(tmp_path, X))
  lin = X[99] + (X[110] - X[99]) * (np.arange(1, 11) / 11)[:, None, None]
  assert np.allclose(P[100:110], lin)


def test_unknown_method(tmp_path):
  with pytest.raises(ValueError):
    ik.load_trial(*_gap_args(tmp_path, _cubic(50)), gap_fill="akima")
