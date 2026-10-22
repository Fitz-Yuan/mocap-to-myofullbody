"""Muscle activations of the static optimisation (pipeline/muscles.py) against surface EMG.

EMG: the task's reader if it has one (tasks/<task>/emg.py: read(trial, n), MUSCLES channel ->
regular expression on the model's muscle names, and optionally REFERENCE channel -> kinetics
key, BASELINE channel -> (kinetics key, sign) and cycles(trial)); else the C3D analog channels
kept in the raw cache (<M2M_CACHE_DIR>/<trial>.npz) with M2M_EMG_MAP
("channel=muscle+muscle[@[-]kinetics key],...": the muscles, and the baseline moment) and the gait
cycle of the C3D's events (Foot Strike1 -> Foot Strike2 of the side the channels are on,
"R_" / "L_", in 1 % steps; the foot off is reported; needs ezc3d). Envelopes: mocap2myo/emg.py (20-450 Hz, rectified, 6 Hz),
averaged per motion-capture frame. The model side of a channel: the F0-weighted mean
activation of its muscles that the optimisation uses (none: the channel has no counterpart).

EMG clock. With a REFERENCE (case study), the take's EMG is shifted as a whole: per take, the
lag at which each reference channel best follows its net joint moment (within +-1.5 s, kept
when r > 0.3) gives a median; every take is shifted so that median equals the median over all
takes. That removes recording offsets between takes (a channel's own electromechanical delay,
common to all takes, stays in the comparison). Without one, the EMG is taken as recorded.

Per trial and channel, over the frames where the joints the channel's model muscles cross
are determined (no such matched component uncertain):
  r0          correlation of the EMG envelope with the model activation
  r_net       with a baseline (channel -> kinetics key, sign): correlation of the EMG with
              the positive part of that net moment, the baseline the model should beat
  lead_ms, r  the lag at which they agree best (+-0.5 s; > 0: the EMG leads), r there
  shape_r     correlation of the cycle-averaged profiles (each cycle time-normalised between
              its events, then averaged over the trial's cycles); shape_r_net the same for
              the baseline
  on_agree, on_jaccard   agreement of the profiles' on phases (above 25 % of each profile's
              range): fraction of the cycle, and overlap of the on parts
  rise_ms     cycles with more than two events (box_lift): within 1 s of each cycle's second
              event (the lift-off), when the EMG passes half its rise minus when the model
              activation does (< 0: the EMG rises first), median over the cycles
and the profiles themselves (profiles.npz) for figures.

Writes <out>/emg_compare.json, emg_compare.md, profiles.npz (default out: <M2M_MUSCLE_DIR>).
  python pipeline/validate/emg_compare.py [--muscle-dir DIR] [--out DIR] [trial ...]
"""

import importlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402
from mocap2myo import emg  # noqa: E402

N_PER = 50  # profile samples per phase between two events
LAG_REF = 150  # frames: lag search for the clock alignment
LAG_CMP = 50  # frames: lag search for the EMG lead
RISE_S = 1.0  # s each side of a cycle's second event: the window of the rise timing


def source():
  """The task's EMG module, or None (C3D analog channels)."""
  t = config.TASK
  if t not in ("", "none") and os.path.exists(os.path.join(config.REPO, "tasks", t, "emg.py")):
    return importlib.import_module(f"tasks.{t}.emg")
  return None


def c3d_map():
  """M2M_EMG_MAP as ({channel: regular expression on the muscle names}, {channel: (kinetics
  key, sign)} of the entries with a baseline, "channel=muscle+muscle@[-]key")."""
  if not config.EMG_MAP:
    raise SystemExit("no task EMG reader and no M2M_EMG_MAP (channel=muscle+muscle,...)")
  out, base = {}, {}
  for item in config.EMG_MAP.split(","):
    ch, _, mus = item.partition("=")
    mus, _, key = mus.partition("@")
    if not mus:
      raise SystemExit(f"M2M_EMG_MAP: {item!r} is not channel=muscle+muscle[@[-]kinetics key]")
    out[ch.strip()] = "(" + "|".join(re.escape(x.strip()) for x in mus.split("+")) + ")"
    if key.strip():
      key = key.strip()
      base[ch.strip()] = (key.lstrip("-"), -1 if key.startswith("-") else 1)
  return out, base


def c3d_emg(trial, n, rate):
  """(labels, envelopes [n, k]) from the raw cache's analog channels."""
  D = np.load(os.path.join(config.CACHE_DIR, f"{trial}.npz"), allow_pickle=True)
  if "analog" not in D.files:
    return None
  fs = float(D["analog_rate"])
  ratio = fs / rate
  if abs(ratio - round(ratio)) > 1e-6:
    raise SystemExit(f"{trial}: analog rate {fs:g} Hz is not a multiple of {rate:g} Hz")
  env = emg.envelope(D["analog"], fs)
  return [str(x) for x in D["analog_labels"]], emg.by_ratio(env, int(round(ratio)), n)


def c3d_cycle(trial, rate, side):
  """([[foot strike 1, foot strike 2]], foot off in % of that cycle) of `side` (Right / Left)
  from the C3D events, or ([], None)."""
  import ezc3d
  D = np.load(os.path.join(config.CACHE_DIR, f"{trial}.npz"), allow_pickle=True)
  c = ezc3d.c3d(os.path.join(config.RAW_DIR, str(D["source_file"])))
  ev = c["parameters"].get("EVENT", {})
  if "LABELS" not in ev:
    return [], None
  lab, ctx = ev["LABELS"]["value"], ev["CONTEXTS"]["value"]
  tm = np.asarray(ev["TIMES"]["value"], float)[1]
  start = (c["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0] - 1) / rate
  t = {}
  for lb, cx, tt in zip(lab, ctx, tm):
    if cx == side:
      t.setdefault(lb.replace(" ", ""), []).append(int(round((tt - start) * rate)))
  if not ("FootStrike1" in t and "FootStrike2" in t):
    return [], None
  a, b = t["FootStrike1"][0], t["FootStrike2"][0]
  off = [x for x in t.get("FootOff", []) if a < x < b]
  return [[a, b]], (100.0 * (off[0] - a) / (b - a) if off else None)


def model_side(S, pattern):
  """F0-weighted mean activation [T] of the used muscles matching pattern, and their names."""
  names = [str(x) for x in S["muscles"]]
  k = np.array([bool(re.fullmatch(pattern, n)) for n in names]) & S["used"]
  if not k.any():
    return None, []
  w = S["F0"][k].astype(float)
  return (S["act"][:, k].astype(float) * w).sum(1) / w.sum(), [n for n, kk in zip(names, k) if kk]


def take_lag(S, kn, E, chans, ref):
  """Median over the reference channels of the lag (frames) at which the EMG best follows the
  channel's net moment, r(moment(t), EMG(t + L)); None if no channel reaches r > 0.3."""
  ok = ~kn["uncertain"]
  lags = np.arange(-LAG_REF, LAG_REF + 1)
  out = []
  for ch, key in ref.items():
    if ch not in chans:
      continue
    r = emg.lagged_corr(kn[key], E[:, chans.index(ch)], lags, ok)
    if np.nanmax(r) > 0.3:
      out.append(lags[int(np.nanargmax(r))])
  return float(np.median(out)) if out else None


def compare(trial, S, E, chans, cmap, cycles, shift, kn=None, base=None):
  """Metrics and profiles of one trial (E: envelopes as recorded, shifted here); base:
  {channel: (kinetics key, sign)} for the net-moment baseline."""
  E = emg.shift(E, int(round(shift)))
  res, prof = {}, {}
  lags = np.arange(-LAG_CMP, LAG_CMP + 1)
  for ch in chans:
    if ch not in cmap:
      continue
    e = E[:, chans.index(ch)]
    a, used = (None, []) if cmap[ch] is None else model_side(S, cmap[ch])
    row = dict(muscles=used)
    # the frames where every joint these muscles cross is determined
    k = np.isin(S["muscles"], used)
    ok = ~S["uncertain"][:, S["crosses"][k].any(0)].any(1) if k.any() else ~kn["uncertain"].astype(bool)
    n_per = N_PER if not cycles or len(cycles[0]) > 2 else 100  # one phase: 1 % steps
    pe = [emg.phase_profile(np.nan_to_num(e, nan=np.nanmedian(e)), ev, n_per) for ev in cycles]
    net = None
    if base and ch in base:  # the net moment in the muscle's direction, as a baseline
      key, sg = base[ch]
      net = np.maximum(sg * kn[key], 0.0)
      row["r_net"] = emg.corr(net, e, ok)
    if a is not None:
      row.update(model_p95=float(np.percentile(a[ok], 95)), model_active=float(np.mean(a[ok] > 0.02)))
      row["r0"] = emg.corr(a, e, ok)
      r = emg.lagged_corr(a, e, lags, ok)  # r(act(t), EMG(t + L)): L < 0 = EMG leads
      if np.isfinite(r).any():
        i = int(np.nanargmax(r))
        row.update(lead_ms=float(-lags[i] * 1000 / float(S["rate"])), r_best=float(r[i]))
      pa = [emg.phase_profile(a, ev, n_per) for ev in cycles]
      if cycles and len(cycles[0]) > 2:  # rise around each cycle's second event (box_lift: lift-off)
        w, dt = int(round(RISE_S * S["rate"])), []
        for ev in cycles:
          sl = slice(max(0, ev[1] - w), min(len(a), ev[1] + w))
          te, ta = emg.half_rise(e[sl]), emg.half_rise(a[sl])
          if te is not None and ta is not None:
            dt.append(te - ta)
        if dt:
          row["rise_ms"] = float(np.median(dt) * 1000 / S["rate"])
    if cycles:
      pe = np.mean(pe, 0)
      prof[f"{trial}|{ch}|emg"] = pe
      if net is not None:
        pn = np.mean([emg.phase_profile(net, ev, n_per) for ev in cycles], 0)
        prof[f"{trial}|{ch}|net"] = pn
        row["shape_r_net"] = emg.corr(pn, pe)
      if a is not None:
        pa = np.mean(pa, 0)
        prof[f"{trial}|{ch}|model"] = pa
        row["shape_r"] = emg.corr(pa, pe)
        row["on_agree"], row["on_jaccard"] = emg.overlap(emg.on_windows(pa), emg.on_windows(pe))
    res[ch] = row
  return res, prof


def main():
  config.usage(__doc__)
  args = sys.argv[1:]
  opt = {}
  for k in ("--muscle-dir", "--out"):
    if k in args:
      i = args.index(k)
      opt[k] = args[i + 1]
      del args[i:i + 2]
  mdir = opt.get("--muscle-dir", config.MUSCLE_DIR)
  out = opt.get("--out", mdir)
  os.makedirs(out, exist_ok=True)
  trials = args or config.trials(mdir)
  src = source()
  cmap, base = (src.MUSCLES, getattr(src, "BASELINE", None)) if src is not None else c3d_map()
  ref = getattr(src, "REFERENCE", None) if src is not None else None
  data = {}
  for trial in trials:
    S = dict(np.load(os.path.join(mdir, f"{trial}.npz"), allow_pickle=True))
    kn = np.load(os.path.join(config.KIN_DIR, f"{trial}.npz"))
    T = len(S["act"])
    rate = float(kn["rate"])
    S["rate"] = rate
    if src is not None:
      got = src.read(trial, T)
      if got is None:
        print(trial, "no EMG", flush=True)
        continue
      chans, E = got[0], got[1]
      cycles, off = (src.cycles(trial) if hasattr(src, "cycles") else []), None
    else:
      got = c3d_emg(trial, T, rate)
      if got is None:
        print(trial, "no analog channels", flush=True)
        continue
      chans, E = got
      side = "Right" if all(c.startswith("R_") for c in cmap) else "Left"
      cycles, off = c3d_cycle(trial, rate, side)
    lag = take_lag(S, kn, E, chans, ref) if ref else 0.0
    data[trial] = (S, E, chans, cycles, lag, off)
    print(trial, f"{len(cycles)} cycles", f"take lag {lag * 1000 / rate:.0f} ms" if lag is not None else "take lag: none", flush=True)
  lags = [v[4] for v in data.values() if v[4] is not None]
  common = float(np.median(lags)) if ref and lags else 0.0
  results, profiles, shifts = {}, {}, {}
  for trial, (S, E, chans, cycles, lag, off) in data.items():
    s = (lag - common) if lag is not None else 0.0
    shifts[trial] = s * 1000 / S["rate"]
    results[trial], p = compare(trial, S, E, chans, cmap, cycles, s,
                                np.load(os.path.join(config.KIN_DIR, f"{trial}.npz")), base)
    if off is not None:
      results[trial]["foot_off_pct"] = off
    profiles.update(p)
  # summary per channel: medians over trials
  summ = {}
  for ch in cmap:
    rows = [r[ch] for r in results.values() if ch in r and isinstance(r[ch], dict)]
    if not rows:
      continue
    def med(k):
      v = np.array([r[k] for r in rows if k in r], float)
      v = v[np.isfinite(v)]
      return round(float(np.median(v)), 3) if len(v) else None
    summ[ch] = dict(trials=len(rows), muscles=rows[0]["muscles"], **{k: med(k) for k in (
      "model_p95", "model_active", "r0", "r_net", "r_best", "lead_ms", "shape_r", "shape_r_net", "on_agree",
      "on_jaccard", "rise_ms")})
  rate = next(iter(data.values()))[0]["rate"] if data else 100.0
  rep = dict(muscle_dir=mdir, clock=dict(reference=bool(ref), common_lag_ms=common * 1000 / rate if ref else None,
                                         shift_ms=shifts), channels=summ, per_trial=results)
  json.dump(rep, open(os.path.join(out, "emg_compare.json"), "w"), indent=1)
  np.savez_compressed(os.path.join(out, "profiles.npz"), **profiles)
  lines = [f"EMG vs static optimisation, {len(results)} trials (median over trials)", "",
           "| Channel | Model muscles | model p95 / active | r | r net moment | best r | EMG lead (ms) | profile r | "
           "profile r net moment | on/off agreement | rise EMG - model (ms) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
  f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
  for ch, s in summ.items():
    mus = ", ".join(s["muscles"][:4]) + (f" (+{len(s['muscles']) - 4})" if len(s["muscles"]) > 4 else "") or "none"
    lines.append(f"| {ch} | {mus} | {f(s['model_p95'])} / {f(s['model_active'])} | {f(s['r0'])} | {f(s['r_net'])} | {f(s['r_best'])} | "
                 f"{f(s['lead_ms'], 0)} | {f(s['shape_r'])} | {f(s['shape_r_net'])} | {f(s['on_agree'])} | "
                 f"{f(s['rise_ms'], 0)} |")
  open(os.path.join(out, "emg_compare.md"), "w").write("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
