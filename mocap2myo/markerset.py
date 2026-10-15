"""Marker sets: which MyoFullBody body each optical marker moves with.

A marker set is a dict with these keys:
  name        what it is called
  markers     label -> (body, weight): the MyoFullBody body the marker sits on and
              its weight in the IK. The label order is the marker order of every
              marker array in the pipeline ([T, n, 3], "marker_labels" in the npz).
  anat_left   anatomical guess of each left-side marker (world, model at qpos0 with
              the root at (0, 0, 0.99) and identity rotation: it faces -y, left = +x),
              keyed by the label without its side letter; the right side mirrors x.
              Calibration refines them.
  anat_mid    the same for the midline markers, keyed by the full label
  head        head marker -> (front, left): +1 / -1 for front / back and left /
              right of the head centre. The model has no neck joint, so head
              markers get a low weight and are never rejected as mislabels; their
              labels are also checked (or assigned) by position.
  pelvis      the four pelvis markers, left / right ASIS, left / right PSIS (root fit,
              rigid refill)
  sole        (side, "heel" | "toe") -> label: foot markers for stance detection
  ankles      ankle markers (quiet-standing detection)
  aliases     other naming schemes for the same physical markers, name ->
              {take label: marker label}; "HEAD" = one of the head markers, assigned
              by position
  prefer      when two take labels map onto one marker (both seen in a frame), the
              one in this list wins

Only Plug-in-Gait is defined. For another marker set, write a dict with the same
keys and point MARKER_SET at it: the module-level names below (MARKERS, LABELS,
HEAD, ...) are the selected set's and are what the solver and the stages use.
"""

import numpy as np

# marker -> (body, weight). Head is rigid on the thorax in MyoFullBody (no neck
# joint), so head markers get a low weight; thigh/shank wands and the
# asymmetric RBAK marker are soft-tissue/placement noisy -> medium weight.
_PIG_MARKERS = {
  "LFHD": ("head", 0.3), "RFHD": ("head", 0.3), "LBHD": ("head", 0.3),
  "RBHD": ("head", 0.3),
  "C7": ("thorax", 1.0), "T10": ("thorax", 1.0), "CLAV": ("thorax", 1.0),
  "STRN": ("thorax", 1.0), "RBAK": ("thorax", 0.3),
  "LSHO": ("scapula_l", 1.0), "LUPA": ("humerus_l", 0.5),
  "LELB": ("humerus_l", 1.0), "LFRM": ("radius_l", 0.5),
  "LWRA": ("radius_l", 1.0), "LWRB": ("ulna_l", 1.0), "LFIN": ("thirdmc_l", 1.0),
  "RSHO": ("scapula_r", 1.0), "RUPA": ("humerus_r", 0.5),
  "RELB": ("humerus_r", 1.0), "RFRM": ("radius_r", 0.5),
  "RWRA": ("radius_r", 1.0), "RWRB": ("ulna_r", 1.0), "RFIN": ("thirdmc_r", 1.0),
  "LASI": ("pelvis", 1.0), "RASI": ("pelvis", 1.0), "LPSI": ("pelvis", 1.0),
  "RPSI": ("pelvis", 1.0),
  "LTHI": ("femur_l", 0.5), "LKNE": ("femur_l", 1.0), "LTIB": ("tibia_l", 0.5),
  "LANK": ("tibia_l", 1.0), "LHEE": ("calcn_l", 1.0), "LTOE": ("calcn_l", 1.0),
  "RTHI": ("femur_r", 0.5), "RKNE": ("femur_r", 1.0), "RTIB": ("tibia_r", 0.5),
  "RANK": ("tibia_r", 1.0), "RHEE": ("calcn_r", 1.0), "RTOE": ("calcn_r", 1.0),
}
# Anatomical guesses (world, model at qpos0, root at (0,0,0.99) identity: faces
# -y, left = +x) for the LEFT-side / midline markers; right side mirrors x.
# Built from the model's joint centres (hip 0.077,0.056,0.911; knee z .507;
# ankle z .107; shoulder .17,.10,1.464; elbow .158,.093,1.174; wrist .203,.075,
# .92) plus typical skin-marker offsets. Calibration refines them.
_PIG_ANAT_L = {
  "FHD": (0.06, 0.02, 1.68), "BHD": (0.06, 0.19, 1.66),
  "SHO": (0.17, 0.10, 1.51), "UPA": (0.23, 0.10, 1.32), "ELB": (0.193, 0.093, 1.17),
  "FRM": (0.21, 0.09, 1.04), "WRA": (0.205, 0.045, 0.925), "WRB": (0.205, 0.105, 0.925),
  "FIN": (0.215, 0.065, 0.81),
  "ASI": (0.125, -0.014, 0.98), "PSI": (0.05, 0.146, 0.985),
  "THI": (0.147, 0.056, 0.71), "KNE": (0.126, 0.056, 0.507), "TIB": (0.136, 0.06, 0.31),
  "ANK": (0.116, 0.066, 0.10), "HEE": (0.084, 0.135, 0.045), "TOE": (0.08, -0.055, 0.035),
}
_PIG_ANAT_MID = {"C7": (0, 0.17, 1.47), "T10": (0, 0.16, 1.23), "CLAV": (0, 0.04, 1.44),
                 "STRN": (0, 0.0, 1.26), "RBAK": (-0.08, 0.18, 1.36)}

# Plug-in-Gait markers labelled live with a Motive skeleton template with ISB
# landmark names (in the case study: one subject). Same physical markers, other
# names; the template's single elbow marker took either epicondyle name (the static
# trial also has a real medial one: the lateral HLE wins, see "prefer", as
# Plug-in-Gait's ELB is the lateral epicondyle), and its head names were rotated on
# that subject (assigned by position instead). LHGT (left counterpart of RBAK) and
# TV2 are not in the model's marker set.
ISB_TO_PIG = {
  "CV7": "C7", "SJN": "CLAV", "SXS": "STRN", "TV7": "T10", "RHGT": "RBAK",
  "LCAJ": "LSHO", "RCAJ": "RSHO", "LUA": "LUPA", "RUA": "RUPA",
  "LHLE": "LELB", "LHME": "LELB", "RHLE": "RELB", "RHME": "RELB",
  "LRSP": "LWRA", "RRSP": "RWRA", "LUSP": "LWRB", "RUSP": "RWRB",
  "LHM2": "LFIN", "RHM2": "RFIN", "LIAS": "LASI", "RIAS": "RASI",
  "LIPS": "LPSI", "RIPS": "RPSI", "LTH": "LTHI", "RTH": "RTHI",
  "LFLE": "LKNE", "RFLE": "RKNE", "LTTC": "LTIB", "RTTC": "RTIB",
  "LFAL": "LANK", "RFAL": "RANK", "LFCC": "LHEE", "RFCC": "RHEE",
  "LDP1": "LTOE", "RDP1": "RTOE",
  "LAH": "HEAD", "RAH": "HEAD", "LPH": "HEAD", "RPH": "HEAD"}

PLUG_IN_GAIT = dict(
  name="Plug-in-Gait",
  markers=_PIG_MARKERS,
  anat_left=_PIG_ANAT_L,
  anat_mid=_PIG_ANAT_MID,
  head={"LFHD": (1, 1), "RFHD": (1, -1), "LBHD": (-1, 1), "RBHD": (-1, -1)},
  pelvis=("LASI", "RASI", "LPSI", "RPSI"),
  sole={("l", "heel"): "LHEE", ("l", "toe"): "LTOE",
        ("r", "heel"): "RHEE", ("r", "toe"): "RTOE"},
  ankles=("LANK", "RANK"),
  aliases={"ISB": ISB_TO_PIG},
  prefer=("LHLE", "RHLE"),
)

MARKER_SET = PLUG_IN_GAIT

MARKERS = MARKER_SET["markers"]
LABELS = list(MARKERS)
HEAD_LABELS = tuple(MARKER_SET["head"])  # head markers, in the order of MARKER_SET["head"]
HEAD = np.array([l in MARKER_SET["head"] for l in LABELS])
PELVIS_LABELS = tuple(MARKER_SET["pelvis"])
PELVIS = [LABELS.index(l) for l in PELVIS_LABELS]
PELVIS_MASK = np.array([l in PELVIS_LABELS for l in LABELS])
SOLE = MARKER_SET["sole"]
ANKLES = tuple(MARKER_SET["ankles"])
ALIASES = MARKER_SET["aliases"]
PREFER = tuple(MARKER_SET["prefer"])
_ANAT_L = MARKER_SET["anat_left"]
_ANAT_MID = MARKER_SET["anat_mid"]


def anat_world(label):
  if label in _ANAT_MID:
    return np.array(_ANAT_MID[label], float)
  side, key = label[0], label[1:]
  x, y, z = _ANAT_L[key]
  return np.array([x if side == "L" else -x, y, z], float)
