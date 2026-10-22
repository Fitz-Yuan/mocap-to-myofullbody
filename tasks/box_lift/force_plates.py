"""Check the inverse dynamics against the lab's force plates (case study only).

The plates were recorded as raw voltages (NI DAQ through Motive, 1 kHz, one file per
take: raw_daq/Subject<NN>_<floor>_NE_NIDAQ*.csv, see daq_file; channels from the
study's ReadMe: Bertec in the walkway, AMTI left / right in front of the shelf). There is
no calibration matrix, so only the vertical channels are used, and checks that do not
need one:
  1. sync: the plates lag the motion capture (best correlation of measured vs predicted
     total vertical force; also seen as plate onsets ~0.1 s after the foot lands),
  2. gain: from quiet standing on both AMTI plates without the box (= body + equipment
     weight; the same N/V for every subject if the calibration holds),
  3. total vertical force: plates (AMTI L + R, both feet on them) vs our prediction,
     outside the frames where the box is changing hands with a surface (+-0.5 s),
  4. left / right share of that force vs our estimate (M2M_KIN_SPLIT),
  5. the box mass: quiet standing with the box minus without it.
The DAQ samples are averaged per motion-capture frame, and the windows below are in
frames of the case study's 100 Hz capture (checked: RATE).
Writes <M2M_KIN_DIR>/force_plates.json and .md.
  python tasks/box_lift/force_plates.py [trial ...]
"""

import glob
import json
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402

DAQ_DIR = os.path.join(config.ROOT, "raw_daq")
RATE = 100.0  # Hz, motion capture (frame-based windows and the 6 Hz filter below)
PLATES = {"bertec": ["ai17", "ai19", "ai7", "ai30", "ai40", "ai15"],  # Fx Fy Fz Mx My Mz
          "amti_l": ["ai1", "ai2", "ai3", "ai4", "ai5", "ai6"],
          "amti_r": ["ai9", "ai10", "ai11", "ai12", "ai13", "ai14"]}


def daq_file(trial):
  """The take's NI DAQ file (S01_Solid_NE -> Subject01_Solid_NE_NIDAQ<device>.csv), or
  None."""
  subj, floor = trial.split("_")[:2]
  hits = sorted(glob.glob(os.path.join(DAQ_DIR, f"Subject{int(subj[1:]):02d}_{floor}_NE_NIDAQ*.csv")))
  if len(hits) > 1:
    raise SystemExit(f"{trial}: several DAQ files {hits}")
  return hits[0] if hits else None


def read_daq(path):
  """Channel voltages averaged per motion-capture frame {channel: [T]}."""
  with open(path) as f:
    for i, line in enumerate(f):
      if line.startswith("MocapFrame"):
        cols = [c.strip().replace("Dev3/", "") for c in line.strip().split(",")]
        break
  A = np.loadtxt(path, delimiter=",", skiprows=i + 1, dtype=np.float32)
  fr = A[:, 0].astype(int)
  cnt = np.bincount(fr)
  return {c: np.bincount(fr, weights=A[:, k], minlength=len(cnt)) / np.maximum(cnt, 1)
          for k, c in enumerate(cols) if k >= 3}


def check(trial):
  path = daq_file(trial)
  if path is None:
    raise SystemExit(f"{trial}: no NI DAQ file in {DAQ_DIR}")
  D = read_daq(path)
  K = np.load(os.path.join(config.KIN_DIR, f"{trial}.npz"))
  ref = np.load(os.path.join(config.REF_DIR, f"{trial}.npz"))
  if float(K["rate"]) != RATE:
    raise SystemExit(f"{trial}: motion capture at {float(K['rate']):g} Hz; force_plates.py "
                     f"is written for {RATE:g} Hz (frame-based windows)")
  pred = K["grf"][:, 2]
  n = min(len(D["ai3"]), len(pred))
  pred = pred[:n]
  b, a = butter(4, 6 / 50)
  V = {p: filtfilt(b, a, D[ch[2]][:n] - np.median(D[ch[2]][:n])) for p, ch in PLATES.items()}
  both = (V["amti_l"] > 0.3) & (V["amti_r"] > 0.3) & (np.abs(V["bertec"]) < 0.2)
  both = np.convolve(both, np.ones(41), "same") >= 41  # 0.2 s in from stepping on / off
  W = (float(K["body_kg"]) + float(K["extra_kg"])) * 9.81
  held, unc = K["held"][:n], K["uncertain"][:n]
  state = ref["support"][:n].astype(int) * 4 + (ref["grasp"][:n] > 0.5) @ np.array([1, 2])
  calm = np.convolve(np.r_[False, np.diff(state) != 0], np.ones(101), "same") == 0
  ok = both & ~unc & calm
  idx = np.where(ok)[0]
  idx = idx[(idx > 30) & (idx < n - 30)]
  if len(idx) < 200:
    return None
  Vs = V["amti_l"] + V["amti_r"]
  lags = list(range(-25, 26))
  lag = lags[int(np.argmax([np.corrcoef(pred[idx], Vs[idx + L])[0, 1] for L in lags]))]
  shift = lambda x: np.r_[x[lag:], np.full(lag, np.nan)] if lag >= 0 else np.r_[np.full(-lag, np.nan), x[:lag]]
  Vl, Vr = shift(V["amti_l"]), shift(V["amti_r"])
  Vs = Vl + Vr
  static = W + held * float(K["box_kg"]) * 9.81
  quiet = ok & (np.abs(pred - static) < 0.02 * W)
  qn, qh = quiet & ~held, quiet & held
  if qn.sum() < 50:
    return None
  gain = W / np.nanmedian(Vs[qn])
  meas = Vs * gain
  m = ok & ~np.isnan(meas)
  e = pred[m] - meas[m]
  share_meas, share_pred = (Vr / Vs)[m], K["foot_share"][:n, 1][m]
  return dict(lag_ms=10 * lag, gain_N_per_V=float(gain), frames=int(m.sum()),
              fz_rms_N=float(np.sqrt(np.mean(e**2))), fz_rms_pct_weight=float(100 * np.sqrt(np.mean(e**2)) / W),
              fz_r_dynamic=float(np.corrcoef(meas[m] - static[m], pred[m] - static[m])[0, 1]),
              box_kg_plates=float((np.nanmedian(meas[qh]) - W) / 9.81) if qh.sum() > 30 else None,
              share_r_rms=float(np.sqrt(np.mean((share_pred - share_meas) ** 2))),
              share_r_corr=float(np.corrcoef(share_pred, share_meas)[0, 1]))


def main():
  config.usage(__doc__)
  trials = sys.argv[1:] or [t for t in config.trials(config.KIN_DIR) if daq_file(t)]
  res = {}
  for t in trials:
    r = check(t)
    res[t] = r
    print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in (r or {}).items()} or "not enough frames on the plates", flush=True)
  good = {t: r for t, r in res.items() if r}
  med = lambda k: float(np.median([r[k] for r in good.values() if r[k] is not None]))
  summ = {k: med(k) for k in ("lag_ms", "gain_N_per_V", "fz_rms_N", "fz_rms_pct_weight", "fz_r_dynamic",
                              "box_kg_plates", "share_r_rms", "share_r_corr")}
  summ["trials"] = len(good)
  json.dump(dict(per_trial=res, median=summ), open(os.path.join(config.KIN_DIR, "force_plates.json"), "w"), indent=1)
  lines = [f"Force plates vs inverse dynamics, {len(good)} trials (median over trials)", "",
           f"- plates lag the motion capture by {summ['lag_ms']:.0f} ms",
           f"- vertical gain {summ['gain_N_per_V']:.1f} N/V (from quiet standing, body + equipment weight)",
           f"- total vertical force, prediction vs plates: RMS {summ['fz_rms_N']:.1f} N ({summ['fz_rms_pct_weight']:.1f} % of body weight), "
           f"r = {summ['fz_r_dynamic']:.2f} for the part that is not body weight",
           f"- right-foot share of the vertical force: RMS error {summ['share_r_rms']:.3f}, r = {summ['share_r_corr']:.2f}",
           f"- box mass from the plates: {summ['box_kg_plates']:.1f} kg"]
  open(os.path.join(config.KIN_DIR, "force_plates.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
