"""Surface EMG against muscle activations: envelopes at the motion-capture rate, the
recording delay, within-trial normalisation, lagged correlation, on / off timing and
cycle averages (pipeline/validate/emg_compare.py).

Envelope: band-pass (default 20-450 Hz, 4th-order Butterworth, zero phase), full-wave
rectification, low-pass (default 6 Hz, zero phase: the cut-off the joint loads use before
differentiating), then averaged over the analog samples of each motion-capture frame.
Amplitudes are compared within a trial only (each signal over its own percentiles), not
as a percentage of maximal voluntary contractions: MVC trials are often not maximal for
every channel.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def envelope(X, fs, band=(20.0, 450.0), lp_hz=6.0):
  """Linear envelope [S, k] of raw EMG X [S, k] sampled at fs (Hz). The band's upper edge is
  kept below 0.95 x Nyquist."""
  X = np.asarray(X, float)
  hi = min(band[1], 0.95 * fs / 2)
  b, a = butter(4, [band[0] / (fs / 2), hi / (fs / 2)], "band")
  Y = np.abs(filtfilt(b, a, X - X.mean(0), axis=0))
  b, a = butter(4, lp_hz / (fs / 2))
  return filtfilt(b, a, Y, axis=0)


def per_frame(env, frame, n=None):
  """Mean of the samples of each frame: env [S, k], frame [S] (the frame index of each
  sample) -> [n, k]."""
  frame = np.asarray(frame, int)
  n = int(frame.max()) + 1 if n is None else n
  keep = (frame >= 0) & (frame < n)
  cnt = np.bincount(frame[keep], minlength=n)
  out = np.stack([np.bincount(frame[keep], weights=env[keep, k], minlength=n)
                  for k in range(env.shape[1])], 1)
  out = out / np.maximum(cnt, 1)[:, None]
  out[cnt == 0] = np.nan
  return out


def by_ratio(env, ratio, n):
  """Frames of a channel block sampled at `ratio` x the frame rate (an integer; C3D analog
  channels): the mean of the samples of each frame -> [n, k]."""
  S = min(len(env), n * ratio)
  return per_frame(env[:S], np.arange(S) // ratio, n)


def shift(x, lag):
  """x [T, ...] advanced by `lag` frames (x recorded `lag` frames late: out[t] = x[t + lag]);
  NaN where it has no sample."""
  out = np.full_like(np.asarray(x, float), np.nan)
  if lag >= 0:
    out[:len(x) - lag] = x[lag:]
  else:
    out[-lag:] = x[:lag]
  return out


def scale01(x, mask=None, lo=2.0, hi=98.0):
  """x over its own lo / hi percentiles (within-trial normalisation; over `mask` frames)."""
  v = x[mask] if mask is not None else x
  v = v[np.isfinite(v)]
  a, b = np.percentile(v, [lo, hi])
  return (x - a) / max(b - a, 1e-12)


def corr(x, y, mask=None):
  """Pearson r over the frames where both are finite (and mask)."""
  ok = np.isfinite(x) & np.isfinite(y)
  if mask is not None:
    ok &= mask
  if ok.sum() < 10 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
    return np.nan
  return float(np.corrcoef(x[ok], y[ok])[0, 1])


def lagged_corr(x, y, lags, mask=None):
  """r(x(t), y(t + L)) for each lag L in frames: L > 0 when x leads y. mask: frames of x."""
  out = []
  for L in lags:
    ys = shift(y, L)
    out.append(corr(x, ys, mask))
  return np.array(out)


def on_windows(p, thr=0.25):
  """On / off of a cycle profile p [n] (normalised 0..1 over its own range): on where p is
  above thr; returns the on mask."""
  q = (p - np.nanmin(p)) / max(np.nanmax(p) - np.nanmin(p), 1e-12)
  return q > thr


def phase_profile(x, events, n_per=50):
  """Time-normalised profile of x [T] (or [T, k]) between consecutive event frames:
  each interval resampled to n_per points (linear), concatenated -> [n_per * (len(events)
  - 1)] (or [.., k])."""
  x = np.asarray(x, float)
  out = []
  t = np.arange(len(x))
  for a, b in zip(events[:-1], events[1:]):
    s = np.linspace(a, b, n_per, endpoint=False)
    if x.ndim == 1:
      out.append(np.interp(s, t, x))
    else:
      out.append(np.stack([np.interp(s, t, x[:, k]) for k in range(x.shape[1])], 1))
  return np.concatenate(out, 0)


def overlap(on_a, on_b):
  """Agreement of two on / off masks: (fraction of samples that agree, Jaccard index of the
  on samples)."""
  agree = float(np.mean(on_a == on_b))
  u = np.sum(on_a | on_b)
  return agree, float(np.sum(on_a & on_b) / u) if u else np.nan


def half_rise(x):
  """Index where x [n] first passes half its range on the way up to its maximum (from the
  lowest point before the maximum), or None if it does not rise."""
  x = np.asarray(x, float)
  if not np.isfinite(x).all() or np.ptp(x) <= 0:
    return None
  top = int(np.argmax(x))
  low = int(np.argmin(x[:top + 1]))
  if top == low:
    return None
  half = 0.5 * (x[low] + x[top])
  return low + int(np.argmax(x[low:top + 1] > half))
