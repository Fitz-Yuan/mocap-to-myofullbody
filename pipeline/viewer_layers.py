"""Viewer layers for a clip: plot menu (angles, joint loads, ground reaction), force arrows
and the OpenSim comparison (stick figure, markers, dashed curves). Positions go into the
clip's frame: the reference's model space (subject x scale) raised by the reference's
human shift.

layers(m, Q, idx, scale, human_dz, q16, ps, kin=None, osim=None, osim_markers=None, profile=None,
       conf=None)
  m: MuJoCo model (joint names), Q: reference qpos for the clip frames [n, nq],
  profile: the model profile (its [angles]; default MyoFullBody),
  idx: clip frame indices into the trial, scale: subject scale of the IK, human_dz: the
  reference's vertical shift of the body, q16 / ps: export_viewer's int16 position
  encoder and its metres per unit, kin / osim: npz of our kinetics and of the OpenSim
  stage (optional), osim_markers: OpenSim's model markers [T, n, 3] (optional), conf: npz of
  pipeline/confidence.py (optional: each joint-angle series gets a band of +-1 SD, the
  per-frame SD of its joint).
  Returns (arrays, meta) to merge into the clip. Without task loads in the kinetics
  (M2M_TASK=none: no hand_force) the hand-load plot and arrows are left out.
"""

import numpy as np

ANGLES = [  # id, title, [(our anatomical angle, OpenSim coordinate or None)], note
  # (the model profile's [angles]: angle -> joint and sign; MyoFullBody: knee_flex_r = knee_angle_r)
  ("ang_knee", "Knee flexion", [("knee_flex_r", "knee_angle_r"), ("knee_flex_l", "knee_angle_l")], ""),
  ("ang_hipf", "Hip flexion", [("hip_flex_r", "hip_flexion_r"), ("hip_flex_l", "hip_flexion_l")], ""),
  ("ang_hipa", "Hip adduction", [("hip_add_r", "hip_adduction_r"), ("hip_add_l", "hip_adduction_l")], ""),
  ("ang_ankle", "Ankle dorsiflexion", [("ankle_dorsi_r", "ankle_angle_r"), ("ankle_dorsi_l", "ankle_angle_l")], ""),
  ("ang_lumbar", "Lumbar extension", [("lumbar_ext", "lumbar_extension")],
   "Negative = flexion (bending forward). MyoFullBody spreads it over five lumbar joints; OpenSim's Rajagopal model has one."),
  ("ang_shoulder", "Shoulder elevation", [("shoulder_elv_r", None), ("shoulder_elv_l", None)],
   "Elevation angle of the humerus (no OpenSim equivalent: Rajagopal uses flexion / adduction / rotation)."),
  ("ang_elbow", "Elbow flexion", [("elbow_flex_r", "elbow_flex_r"), ("elbow_flex_l", "elbow_flex_l")], ""),
]
UNCERTAIN = ("Grey: the carried load is touching a surface while held (or the IK left the frame "
             "unsolved), so the share it puts on the hands is not known")
UNSOLVED = "Grey: frames the IK left unsolved (interpolated)"
DOUBLE = "Grey: both feet on the ground; the left / right split of the ground reaction is estimated"
LOADS = [  # id, title, unit, [(label, our key, side)], shade, refs, note
  ("L5S1_ext", "Low back (L5/S1) extension moment", "N·m", [("L5/S1", "L5S1_ext", "")], "unc", [],
   "Net moment the trunk extensors must produce, from the motion of everything above L5/S1 and the carried load; no ground reaction needed."),
  ("L5S1_compression", "Low back (L5/S1) compression", "N", [("L5/S1", "L5S1_compression", "")], "unc",
   [(3400, "NIOSH action limit"), (6400, "NIOSH maximum")],
   "Single-equivalent-muscle estimate (erector spinae moment arm 5 cm, sagittal plane): an approximation, not a muscle model."),
  ("L5S1_shear_ap", "Low back (L5/S1) anterior shear", "N", [("L5/S1", "L5S1_shear_ap", "")], "unc", [], ""),
  ("shoulder_flex", "Shoulder flexion moment", "N·m", [("Right", "shoulder_r_flex", "r"), ("Left", "shoulder_l_flex", "l")], "unc", [], ""),
  ("shoulder_abd", "Shoulder abduction moment", "N·m", [("Right", "shoulder_r_abd", "r"), ("Left", "shoulder_l_abd", "l")], "unc", [], ""),
  ("elbow_flex", "Elbow flexion moment", "N·m", [("Right", "elbow_r_flex", "r"), ("Left", "elbow_l_flex", "l")], "unc", [], ""),
  ("hip_ext", "Hip extension moment", "N·m", [("Right", "hip_r_ext", "r"), ("Left", "hip_l_ext", "l")], "dbl", [], ""),
  ("hip_abd", "Hip abduction moment", "N·m", [("Right", "hip_r_abd", "r"), ("Left", "hip_l_abd", "l")], "dbl", [], ""),
  ("knee_ext", "Knee extension moment", "N·m", [("Right", "knee_r_ext", "r"), ("Left", "knee_l_ext", "l")], "dbl", [], ""),
  ("ankle_plantar", "Ankle plantarflexion moment", "N·m", [("Right", "ankle_r_plantar", "r"), ("Left", "ankle_l_plantar", "l")], "dbl", [], ""),
]


def angles(profile):
  """ANGLES with the model's joints: [(id, title, [(angle, joint, sign, OpenSim)], note)];
  angles the model has not are left out (and plots with none)."""
  out = []
  for pid, title, pairs, note in ANGLES:
    got = [(a, *profile.angle(a), o) for a, o in pairs if profile.angle(a) is not None]
    if got:
      out.append((pid, title, got, note))
  return out


def layers(m, Q, idx, scale, human_dz, q16, ps, kin=None, osim=None, osim_markers=None, profile=None,
           conf=None):
  arrays, plots = {}, []
  to_clip = lambda P: P * scale + np.array([0.0, 0.0, human_dz])  # lab (subject) -> clip frame
  if profile is None:
    from mocap2myo import profile as _p
    profile = _p.load("myofullbody")
  # angles: ours from the reference qpos; OpenSim from its IK
  angs = angles(profile)
  cols, sgn, ocols = [], [], []
  for _, _, pairs, _ in angs:
    for a, j, sg, o in pairs:
      cols.append(j)
      sgn.append(sg)
      if o and osim is not None and f"angle_{o}" in osim.files:
        ocols.append(o)
  ang = np.degrees(np.stack([sg * Q[:, m.jnt_qposadr[m.joint(j).id]] if sg != 1 else
                             Q[:, m.jnt_qposadr[m.joint(j).id]] for j, sg in zip(cols, sgn)], 1)).astype(np.float32)
  arrays["ang"] = ang
  if ocols:
    n = len(osim[f"angle_{ocols[0]}"])
    ii = np.minimum(idx, n - 1)
    arrays["osim_ang"] = np.stack([osim[f"angle_{o}"][ii] for o in ocols], 1).astype(np.float32)
  band = None
  if conf is not None:  # the SD of each curve's joint (the sign of an angle does not change it)
    joints = [str(j) for j in conf["joints"]]
    sd = conf["joint_sd_deg"][np.minimum(idx, len(conf["joint_sd_deg"]) - 1)]
    arrays["ang_sd"] = np.stack([sd[:, joints.index(j)] if j in joints else np.full(len(idx), np.nan)
                                 for j in cols], 1).astype(np.float32)
    band = lambda c: dict(array="ang_sd", col=c)
  for pid, title, pairs4, note in angs:
    pairs = [(j, o) for _, j, _, o in pairs4]
    two = len(pairs) > 1
    ser = [dict(label=("Right", "Left")[k] if two else title, array="ang", col=cols.index(j),
                side=("r", "l")[k] if two else "") for k, (j, _) in enumerate(pairs)]
    if band is not None:
      for sr in ser:
        sr["band"] = band(sr["col"])
    cmp = [dict(label="OpenSim", array="osim_ang", col=ocols.index(o), side=("r", "l")[k] if two else "")
           for k, (_, o) in enumerate(pairs) if o in ocols]
    plots.append(dict(id=pid, group="Joint angles", title=title, unit="°", series=ser, cmp=cmp or None, note=note))
  arrows, compare = [], None
  if kin is not None:
    # the model's joint loads (a model without arms has no shoulder / elbow loads)
    loads = [(pid, t, u, [x for x in ser if x[1] in kin.files], sh, r, n)
             for pid, t, u, ser, sh, r, n in LOADS]
    loads = [x for x in loads if x[3]]
    keys = [k for _, _, _, ser, _, _, _ in loads for _, k, _ in ser]
    T = len(kin["grf"])
    ii = np.minimum(idx, T - 1)
    arrays["kin"] = np.stack([kin[k][ii] for k in keys], 1).astype(np.float32)
    grf, share = kin["grf"][ii], kin["foot_share"][ii]
    hands = "hand_force" in kin.files  # the task's load on the hands (none without a task)
    gcols = [grf[:, 2], share[:, 0] * grf[:, 2], share[:, 1] * grf[:, 2]]
    if hands:
      gcols += [-kin["hand_force"][ii, 0, 2], -kin["hand_force"][ii, 1, 2]]
    arrays["grf"] = np.stack(gcols, 1).astype(np.float32)
    arrays["shade_unc"] = kin["uncertain"][ii].astype(np.uint8)
    arrays["shade_dbl"] = (kin["foot_contact"][ii].sum(1) != 1).astype(np.uint8)
    okeys = []
    if osim is not None:
      okeys = [k for k in keys if f"moment_{k}" in osim.files]
      if okeys:
        n = len(osim[f"moment_{okeys[0]}"])
        jj = np.minimum(idx, n - 1)
        arrays["osim_kin"] = np.stack([osim[f"moment_{k}"][jj] for k in okeys], 1).astype(np.float32)
    for pid, title, unit, ser, shade, refs, note in loads:
      plots.append(dict(
        id=pid, group="Joint loads", title=title, unit=unit,
        series=[dict(label=lab, array="kin", col=keys.index(k), side=sd) for lab, k, sd in ser],
        cmp=[dict(label="OpenSim", array="osim_kin", col=okeys.index(k), side=sd)
             for lab, k, sd in ser if k in okeys] or None,
        shade=dict(array=f"shade_{shade}",
                   label=(UNCERTAIN if hands else UNSOLVED) if shade == "unc" else DOUBLE),
        refs=[dict(value=v, label=l) for v, l in refs], note=note))
    plots.append(dict(id="grf_z", group="Ground reaction", title="Vertical ground force", unit="N",
                      series=[dict(label="Total", array="grf", col=0, side=""),
                              dict(label="Right foot", array="grf", col=2, side="r"),
                              dict(label="Left foot", array="grf", col=1, side="l")],
                      shade=dict(array="shade_dbl", label=DOUBLE),
                      note="From the whole-body motion and the carried load (no force plates)." if hands
                      else "From the whole-body motion (no force plates)."))
    if hands:
      plots.append(dict(id="box_hands", group="Ground reaction", title="Carried load on each hand (vertical)", unit="N",
                        series=[dict(label="Right", array="grf", col=4, side="r"), dict(label="Left", array="grf", col=3, side="l")],
                        shade=dict(array="shade_unc", label=UNCERTAIN), note=""))
      palm = to_clip(kin["palm"][ii])
      arrays["arr_hand_o"], arrays["arr_hand_v"] = q16(palm, ps), kin["hand_force"][ii].astype(np.float32)
    fcop = to_clip(np.nan_to_num(kin["foot_cop"][ii]))
    arrays["arr_foot_o"] = q16(fcop, ps)
    arrays["arr_foot_v"] = (share[..., None] * grf[:, None, :]).astype(np.float32)
    for h, nm in enumerate(("left", "right")):
      if hands:
        arrows.append(dict(name=f"Carried load on the {nm} hand", origin="arr_hand_o", vector="arr_hand_v", col=h,
                           m_per_N=0.004, color="--force", anchor="head", min_N=3))
      arrows.append(dict(name=f"Ground reaction, {nm} foot", origin="arr_foot_o", vector="arr_foot_v", col=h,
                         m_per_N=0.0008, color="--accent-2", anchor="tail", min_N=10))
  if osim is not None and "joints" in osim.files:
    n = len(osim["joints"])
    jj = np.minimum(idx, n - 1)
    arrays["osim_joints"] = q16(to_clip(osim["joints"][jj]), ps)
    compare = dict(name="OpenSim (Rajagopal 2016)", joints=[str(b) for b in osim["bodies"]],
                   links=[[int(a), int(b)] for a, b in osim["links"]], pos="osim_joints")
    if osim_markers is not None:
      M = osim_markers[np.minimum(idx, len(osim_markers) - 1)]
      arrays["osim_markers"] = q16(to_clip(M), ps)
      compare["markers"] = "osim_markers"
  return arrays, dict(plots=plots, arrows=arrows, compare=compare)
