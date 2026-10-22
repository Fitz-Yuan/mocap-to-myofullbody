"""Surface EMG of the case study (Delsys Trigno through Motive, 1 kHz, 16 channels), for
pipeline/validate/emg_compare.py.

Files: raw_daq/Subject<NN>_<floor>_NE_Trigno_3502.csv, one per take, each sample tagged
with its motion-capture frame (MocapFrame). The channel names in the files' header are a
lower-limb template left in the export; the true order is the study's ReadMe (CHANNELS),
which the MVC trials confirm (each channel peaks in its own muscle's MVC). The files are
read as they are.

The analog channels run behind the motion capture (the force plates by 60-130 ms,
tasks/box_lift/force_plates.py); the delay of a take is estimated in emg_compare.py.
"""

import glob
import os

import numpy as np

from mocap2myo import emg
from pipeline import config

DAQ_DIR = os.path.join(config.ROOT, "raw_daq")
CHANNELS = ["ESL_R", "ESL_L", "RA_R", "RA_L", "EO_R", "EO_L", "TRAP_R", "TRAP_L", "DELT_R", "DELT_L",
            "BIC_R", "BIC_L", "TRI_R", "TRI_L", "FCR_R", "FCR_L"]
# channel -> MyoFullBody muscles (regular expressions on the actuator names); the group's
# activation is the F0-weighted mean over the muscles the optimisation uses
ARM = {"DELT": "DELT1", "BIC": "BIClong|BICshort", "TRI": "TRIlong|TRIlat|TRImed", "FCR": "FCR"}
MUSCLES = {
  **{f"ESL_{s.upper()}": rf"(IL|LTpT|LTpL)_.*_{s}" for s in "rl"},   # lumbar erector spinae
  **{f"RA_{s.upper()}": f"rect_abd_{s}" for s in "rl"},
  **{f"EO_{s.upper()}": rf"EO\d_{s}" for s in "rl"},
  **{f"TRAP_{s.upper()}": None for s in "rl"},   # no trapezius in MyoFullBody
  **{f"{k}_R": f"({v})" for k, v in ARM.items()},
  **{f"{k}_L": "(" + "|".join(f"{x}_left" for x in v.split("|")) + ")" for k, v in ARM.items()},
}


def trigno_file(trial):
  """The take's Trigno file (S01_Solid_NE -> Subject01_Solid_NE_Trigno_3502.csv), or None."""
  subj, floor = trial.split("_")[:2]
  hits = sorted(glob.glob(os.path.join(DAQ_DIR, f"Subject{int(subj[1:]):02d}_{floor}_NE_Trigno*.csv")))
  if len(hits) > 1:
    raise SystemExit(f"{trial}: several Trigno files {hits}")
  return hits[0] if hits else None


def read(trial, n, band=(20.0, 450.0), lp_hz=6.0):
  """(CHANNELS, envelopes [n, 16] per motion-capture frame, sample rate) of a take, or None
  without a file. Not delay-corrected."""
  path = trigno_file(trial)
  if path is None:
    return None
  with open(path) as f:
    head = {}
    for i, line in enumerate(f):
      if line.startswith("MocapFrame"):
        break
      parts = line.strip().split(",")
      if len(parts) > 1:
        head[parts[0]] = parts[1]
  fs = float(head.get("Capture Frame Rate", 1000.0))
  A = np.loadtxt(path, delimiter=",", skiprows=i + 1, dtype=np.float64)
  if A.shape[1] != 3 + len(CHANNELS):
    raise SystemExit(f"{path}: {A.shape[1] - 3} channels, expected {len(CHANNELS)}")
  env = emg.envelope(A[:, 3:], fs, band, lp_hz)
  return list(CHANNELS), emg.per_frame(env, A[:, 0].astype(int), n), fs


# channels whose take-wise lag against a net joint moment sets the take's EMG clock
# (pipeline/validate/emg_compare.py): the lumbar erector spinae, biceps and anterior deltoid
REFERENCE = {"ESL_R": "L5S1_ext", "ESL_L": "L5S1_ext", "BIC_R": "elbow_r_flex", "BIC_L": "elbow_l_flex",
             "DELT_R": "shoulder_r_flex", "DELT_L": "shoulder_l_flex"}
# baseline for each channel: the net joint moment in the direction the muscle acts
# (kinetics key, sign), its positive part; the trapezius, with no model counterpart, against
# the shoulder flexion moment
BASELINE = {**{f"ESL_{s}": ("L5S1_ext", 1) for s in "RL"}, **{f"RA_{s}": ("L5S1_ext", -1) for s in "RL"},
            **{f"EO_{s}": ("L5S1_ext", -1) for s in "RL"},
            **{f"{k}_{s}": (f"{j}_{s.lower()}_{c}", sg) for s in "RL"
               for k, j, c, sg in (("TRAP", "shoulder", "flex", 1), ("DELT", "shoulder", "flex", 1),
                                   ("BIC", "elbow", "flex", 1), ("TRI", "elbow", "flex", -1),
                                   ("FCR", "elbow", "flex", 1))}}
PHASES = ("bend and grasp", "lift and carry to the shelf", "on the shelf", "carry back and set down",
          "release and stand up")


def cycles(trial, margin_s=1.0):
  """Events of each carry cycle (frames): 1 s before the lift-off from the floor, lift-off,
  the box placed on the shelf, taken off it, set down on the floor, 1 s after (PHASES lie
  between them). Cycles without a shelf phase are left out."""
  ref = np.load(os.path.join(config.REF_DIR, f"{trial}.npz"))
  sup, rate = ref["support"], float(ref["rate"])
  m = int(round(margin_s * rate))
  out = []
  for a, b in ref["cycles"]:
    on = np.where(sup[a:b] == 2)[0]
    if len(on) == 0 or a - m < 0 or b + m >= len(sup):
      continue
    out.append([a - m, a, a + on[0], a + on[-1] + 1, b, b + m])
  return out
