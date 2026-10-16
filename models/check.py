"""Check every model profile: compile, canonical map, landmarks, height, counts, images.

  python models/check.py [NAME ...] [--out DIR] [--no-images]

For each profile (mocap2myo/profiles/<name>.toml) whose model file is there (see
models/fetch.py): compiles the model with its free root, prints the canonical segment
map, the IK joints, the joint equalities (and whether mocap2myo.kinematics.
apply_joint_equalities reproduces them), the landmark table at the neutral pose (canonical
frame, feet on z = 0), the bone / model height, the kinetics joints with a sign check of
their components, the sole points, and renders front / side / back images with the
landmarks (MuJoCo offscreen, orthographic; MUJOCO_GL=egl by default). For MyoFullBody it
also compares the landmarks with the anatomical guesses of mocap2myo.markerset
(Plug-in-Gait), the sole points with ik.MarkerModel's and the kinetics joints with
mocap2myo.kinetics.JOINTS. Also: a left / right symmetry test of every IK joint pair (it
flags e.g. the released MyoFullBody's left knee coupling: MYOFULLBODY_MODEL=original).
Writes <out>/<name>/landmarks.json and <out>/<name>_landmarks.png; prints a summary table
at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mocap2myo import landmarks as L  # noqa: E402
from mocap2myo import profile as prof  # noqa: E402
from mocap2myo.kinematics import apply_joint_equalities  # noqa: E402

# Plug-in-Gait marker -> landmark it sits on (for the MyoFullBody comparison), and the
# marker's typical offset from the bone surface (skin + marker base + 7 mm ball radius)
PIG_LANDMARK = {
  "FHD": "forehead", "BHD": "back_head", "SHO": "acromion", "ELB": "epicondyle_lat",
  "WRA": "styloid_rad", "WRB": "styloid_uln", "FIN": "mc3_head", "ASI": "asis", "PSI": "psis",
  "KNE": "femur_epi_lat", "ANK": "malleolus_lat", "HEE": "heel", "TOE": "mt2_head",
  "C7": "c7", "T10": "t10", "CLAV": "jugular", "STRN": "xiphoid"}


def muscles(m):
  return int(sum(1 for a in range(m.nu) if m.actuator_dyntype[a] == mujoco.mjtDyn.mjDYN_MUSCLE
                 or m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_MUSCLE))


def rigid_group(m, bodies):
  """True if the bodies are connected without any joint between them (a weld group)."""
  ids = [m.body(b).id for b in bodies]

  def root(b):  # climb while the body has no joint
    while m.body_jntnum[b] == 0 and m.body_parentid[b] != 0:
      b = m.body_parentid[b]
    return b
  return len({root(b) for b in ids}) == 1


def equality_report(m, p, ik_names, rng):
  """Joint equalities: chains (a leader that itself follows), inactive ones, other types,
  and whether a second pass of apply_joint_equalities changes anything (order effects)."""
  T = mujoco.mjtEq
  lines = []
  kinds = {}
  for e in range(m.neq):
    kinds[mujoco.mjtEq(m.eq_type[e]).name] = kinds.get(mujoco.mjtEq(m.eq_type[e]).name, 0) + 1
  lines.append(f"equalities: {kinds or 'none'}")
  jeq = [e for e in range(m.neq) if m.eq_type[e] == T.mjEQ_JOINT]
  follow = {int(m.eq_obj1id[e]) for e in jeq}
  lead = {int(m.eq_obj2id[e]) for e in jeq if m.eq_obj2id[e] >= 0}
  chain = sorted(m.joint(j).name for j in follow & lead)
  inactive = [m.eq(e).name or str(e) for e in range(m.neq) if not m.eq_active0[e]]
  ikf = sorted(n for n in ik_names if m.joint(n).id in follow)
  if chain:
    lines.append(f"  chained (leader that also follows): {chain}")
  if inactive:
    lines.append(f"  inactive at start (apply_joint_equalities applies them anyway): {inactive}")
  if ikf:
    lines.append(f"  IK joints that are equality followers (conflict): {ikf}")
  # order: apply twice on random IK poses
  adr = [m.jnt_qposadr[m.joint(n).id] for n in ik_names]
  jid = [m.joint(n).id for n in ik_names]
  lo = np.array([m.jnt_range[j][0] if m.jnt_limited[j] else -1.0 for j in jid])
  hi = np.array([m.jnt_range[j][1] if m.jnt_limited[j] else 1.0 for j in jid])
  worst, worst_eq = 0.0, 0.0
  d = mujoco.MjData(m)
  for _ in range(5):
    q = m.qpos0.copy()
    q[adr] = lo + rng.random(len(adr)) * (hi - lo)
    apply_joint_equalities(m, q)
    q2 = q.copy()
    apply_joint_equalities(m, q2)
    worst = max(worst, float(np.abs(q2 - q).max()))
    # residual of every equality constraint (joint: should be 0; connect / weld / tendon: what
    # the IK would still have to satisfy)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    eqr = [abs(d.efc_pos[i]) for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    worst_eq = max(worst_eq, max(eqr, default=0.0))
  lines.append(f"  apply_joint_equalities: second pass changes qpos by {worst:.2e}; "
               f"largest equality residual after it on random IK poses {worst_eq:.2e}")
  return lines, dict(kinds=kinds, chained=chain, inactive=inactive, ik_followers=ikf,
                     second_pass=worst, residual=worst_eq)


def kinetics_report(nt, p):
  """Every [kinetics] joint: bodies / joints exist; for components along a joint axis,
  the sign from moving that joint and watching a distal point (the named motion must be
  the positive one)."""
  m, d = nt.m, nt.d
  lines, ok = [], True
  q0 = nt.qpos.copy()
  for jn, k in p.kinetics.items():
    try:
      m.body(k["distal"]), m.body(k["frame"])
      kind, ref = k["point"].split(":", 1)
      (m.joint if kind == "jnt" else m.body)(ref)
    except KeyError as e:
      lines.append(f"  {jn}: missing {e}")
      ok = False
      continue
    msgs = []
    for name, axis, sign in k["comps"]:
      if not axis.startswith("jnt:"):
        msgs.append(f"{name}={axis}{'+' if sign > 0 else '-'}")
        continue
      j = m.joint(axis[4:]).id
      # a point 20 cm further along the distal segment
      b = m.body(k["distal"]).id
      sub = [bb for bb in range(m.nbody) if bb == b or _is_desc(m, bb, b)]
      far = max(sub, key=lambda bb: np.linalg.norm(nt.d.xpos[bb] - d.xanchor[j]))
      p0 = d.xpos[far].copy()
      q = q0.copy()
      q[m.jnt_qposadr[j]] += 0.2
      apply_joint_equalities(m, q)
      nt.set(q)
      dp = d.xpos[far] - p0
      nt.set(q0)
      # anatomical meaning of +joint motion
      side = 1.0 if jn.endswith("_l") else -1.0
      up, ant, lat = dp[2], -dp[1], side * dp[0]
      if jn.startswith(("hip", "shoulder")) and name in ("ext", "flex"):
        pos = "flex" if ant > 0 else "ext"
      elif jn.startswith(("hip", "shoulder")) and name == "abd":
        pos = "abd" if lat > 0 else "add"
      elif jn.startswith("knee"):
        pos = "flex" if ant < 0 else "ext"
      elif jn.startswith("ankle"):
        pos = "dorsi" if up > 0 else "plantar"
      elif jn.startswith("elbow"):
        pos = "flex" if ant > 0 or up > 0 else "ext"
      elif jn.startswith("L5S1") and name == "ext":
        pos = "ext" if -ant > 0 else "flex"
      else:
        pos = "?"
      want = {"ext": "ext", "flex": "flex", "abd": "abd", "plantar": "plantar"}.get(name, "?")
      opp = {"ext": "flex", "flex": "ext", "abd": "add", "plantar": "dorsi"}.get(name, "?")
      good = (sign > 0 and pos == want) or (sign < 0 and pos == opp)
      ok &= good or pos == "?"
      msgs.append(f"{name}={axis}{'+' if sign > 0 else '-'} (+q = {pos}: {'ok' if good else 'WRONG' if pos != '?' else '?'})")
    lines.append(f"  {jn:11s} distal {k['distal']}, at {k['point']}, frame {k['frame']}: " + ", ".join(msgs))
  return lines, ok


def side_swap(n):
  """Name of the other side's counterpart: *_l / *_l_* <-> _r, *_L <-> *_R, MASS FemurL <-> FemurR."""
  for a, b in ((r"_l(?=_|$)", "_r"), (r"_r(?=_|$)", "_l"), (r"_L(?=_|$)", "_R"), (r"_R(?=_|$)", "_L"),
               (r"(?<=[a-z])L(?=_|$)", "R"), (r"(?<=[a-z])R(?=_|$)", "L")):
    if re.search(a, n):
      return re.sub(a, b, n)
  return None


def symmetry_report(nt, ik, delta=0.3):
  """Move each left / right pair of IK joints by the same amount (same sign, or opposite
  sign when the model's left axis is the mirror of the right one) and compare the mirrored
  body positions of the two sides (beyond their asymmetry at the neutral pose). A pair that
  matches neither way moves the two sides differently, as the left knee coupling of the
  released MyoFullBody did."""
  m, d = nt.m, nt.d
  q0 = nt.qpos.copy()
  bodies = [(b, m.body(side_swap(m.body(b).name)).id) for b in range(1, m.nbody)
            if side_swap(m.body(b).name) and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                                               side_swap(m.body(b).name)) >= 0
            and side_swap(m.body(b).name) > m.body(b).name]
  if not bodies:
    return [], 0.0
  bl, br = np.array([b for b, _ in bodies]), np.array([c for _, c in bodies])

  def asym():
    P = d.xpos.copy()
    Q = P[bl] * [-1, 1, 1]
    return np.linalg.norm(Q - P[br], axis=1)
  base = asym()
  out, worst = [], 0.0
  names = set(ik)
  for jl in ik:
    jr = side_swap(jl)
    if not jr or jr not in names or jr < jl:
      continue
    a = [m.jnt_qposadr[m.joint(n).id] for n in (jl, jr)]
    rng = [m.jnt_range[m.joint(n).id] if m.jnt_limited[m.joint(n).id] else (-np.inf, np.inf) for n in (jl, jr)]
    best = None
    for sl, sr in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
      v = [q0[a[0]] + sl * delta, q0[a[1]] + sr * delta]
      if not all(r[0] <= x <= r[1] for x, r in zip(v, rng)):
        continue
      q = q0.copy()
      q[a[0]], q[a[1]] = v
      apply_joint_equalities(m, q)
      nt.set(q)
      e = float(np.abs(asym() - base).max())
      if best is None or e < best[0]:
        best = (e, "same sign" if sl == sr else "opposite sign")
    nt.set(q0)
    if best is None:
      continue
    worst = max(worst, best[0])
    if best[0] > 0.003:
      out.append(f"{jl} / {jr}: mirrored body positions differ by {1e3 * best[0]:.1f} mm ({best[1]})")
  return out, worst


def _is_desc(m, b, root):
  while b > 0:
    b = m.body_parentid[b]
    if b == root:
      return True
  return False


# --- rendering ------------------------------------------------------------------

VIEWS = {"front": ((0, -1, 0), (1, 0, 0)), "left side": ((1, 0, 0), (0, 1, 0)),
         "back": ((0, 1, 0), (-1, 0, 0))}  # camera side (from the body), image x axis


def render(p, res, sole, out_dir, px=1000):
  """Front, left side and back views (orthographic): whole body, upper body, feet; the
  landmarks drawn on top (filled: on the near side, hollow: far side)."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  lms = res.landmarks
  P = np.array([l.pos for l in lms.values()])
  top = P[:, 2].max()
  zh = min(lms[f"hip_{s}"].pos[2] for s in "lr") if "hip_l" in lms else top / 2
  yf = np.mean([lms[n].pos[1] for n in ("heel_l", "toe_l", "heel_r", "toe_r") if n in lms] or [0.0])
  crops = [("whole body", (0.0, 0.0, top / 2 + 0.01), top + 0.12),
           ("upper body", (0.0, 0.0, (zh - 0.12 + top + 0.04) / 2), top + 0.04 - (zh - 0.12)),
           ("feet", (0.0, yf, 0.12), 0.36)]
  spec = p.spec()
  cams = []
  for ci, (_, zc, ext) in enumerate(crops):
    for vname, (side, x) in VIEWS.items():
      y = np.array([0.0, 0.0, 1.0])
      x = np.asarray(x, float)
      z = np.cross(x, y)
      R = np.stack([x, y, z], 1)
      qq = np.zeros(4)
      mujoco.mju_mat2Quat(qq, R.ravel())
      pos = np.asarray(side, float) * 4 + np.asarray(zc)
      n = f"chk{ci}_{vname.replace(' ', '_')}"
      spec.worldbody.add_camera(name=n, pos=list(pos), quat=list(qq), fovy=ext,
                                proj=mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
      cams.append((ci, vname, n, pos, x, y, ext))
  spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, px)
  spec.visual.global_.offheight = max(spec.visual.global_.offheight, px)
  m = spec.compile()
  d = mujoco.MjData(m)
  d.qpos[:] = res.qpos
  mujoco.mj_forward(m, d)
  # bone geometry of the body (the root's subtree; not scenery) only: meshes if the model
  # has any, else its solid primitives
  root = m.jnt_bodyid[m.joint(p.root_joint).id]
  body = np.array([b == root or _is_desc(m, b, root) for b in range(m.nbody)])
  has_mesh = bool((m.geom_type[body[m.geom_bodyid]] == mujoco.mjtGeom.mjGEOM_MESH).any())
  for g in range(m.ngeom):
    if not body[m.geom_bodyid[g]]:
      show = False
    elif has_mesh:
      show = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    else:
      show = m.geom_group[g] != 3 and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
    m.geom_rgba[g] = (0.86, 0.82, 0.74, 0.9) if show else (0, 0, 0, 0)
  m.geom_matid[:] = -1
  opt = mujoco.MjvOption()
  opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
  opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
  opt.sitegroup[:] = 0
  opt.geomgroup[:] = 1
  m.vis.headlight.ambient[:] = (0.5, 0.5, 0.5)
  m.vis.headlight.diffuse[:] = (0.55, 0.55, 0.55)
  m.vis.rgba.fog[:] = 1
  r = mujoco.Renderer(m, px, px)
  fig, axs = plt.subplots(3, 3, figsize=(19, 19), dpi=100)
  col_jc, col_s, col_sole = "#1f5fbf", "#d6336c", "#2b8a3e"
  far_front = {"psis", "sacrum", "c7", "t10", "heel", "back_head"}
  far_back = {"asis", "jugular", "xiphoid", "forehead", "toe", "mtp"}
  for ci, vname, cam, pos, x, y, ext in cams:
    ax = axs[ci, list(VIEWS).index(vname)]
    r.update_scene(d, camera=cam, scene_option=opt)
    img = r.render().copy()
    img[(img == 0).all(-1)] = 255  # white background
    ax.imshow(img)
    sc = px / ext

    def proj(w):
      w = np.atleast_2d(w) - pos
      return px / 2 + w @ x * sc, px / 2 - w @ y * sc
    for n, l in lms.items():
      base, side = (n[:-2], n[-1]) if n[-2:] in ("_l", "_r") else (n, "")
      far = ((vname == "left side" and side == "r") or (vname == "front" and base in far_front)
             or (vname == "back" and base in far_back))
      col = col_jc if (base in L.JOINT_CENTRES or n in p.joint_centres) else col_s
      u, v = proj(l.pos)
      if not (0 <= u[0] < px and 0 <= v[0] < px):
        continue
      ax.scatter(u, v, s=34, facecolors="none" if far else col, edgecolors=col, linewidths=1.3, zorder=3)
      label = (vname == "front" and side != "r") or (vname == "left side" and side != "r") or \
              (vname == "back" and side != "l")
      if label and not far:
        ax.annotate(base, (u[0], v[0]), xytext=(4, 3), textcoords="offset points", fontsize=7,
                    color=col, zorder=4)
    if sole:
      S = np.array([s["pos"] for s in sole])
      u, v = proj(S)
      ax.scatter(u, v, s=12, c=col_sole, marker="s", zorder=3)
    ax.set_xlim(0, px)
    ax.set_ylim(px, 0)
    ax.set_title(f"{vname} ({crops[ci][0]})", fontsize=10)
    ax.set_axis_off()
  r.close()
  h = f"bone height {res.bone_height:.3f} m" if res.bone_height else ""
  fig.suptitle(f"{p.name}: landmarks at the neutral pose, canonical frame ({h}). Blue: joint centres, "
               f"red: surface landmarks, green squares: sole points; hollow: on the far side.", fontsize=12)
  fig.tight_layout()
  path = Path(out_dir) / f"{p.name}_landmarks.png"
  fig.savefig(path)
  plt.close(fig)
  return path


# --- MyoFullBody comparisons ---------------------------------------------------------

def compare_myofullbody(p, lines):
  from mocap2myo import markerset as MS
  r = L.compute(p, place="root")
  lines.append("  vs Plug-in-Gait's MyoFullBody guesses (markersets/plug_in_gait.toml [override]; root at (0, 0, 0.99)):")
  ref = {"hip": (0.077, 0.056, 0.911), "shoulder": (0.17, 0.10, 1.464), "elbow": (0.158, 0.093, 1.174),
         "wrist": (0.203, 0.075, 0.92)}
  for j, v in ref.items():
    e = np.linalg.norm(r.landmarks[f"{j}_l"].pos - v)
    lines.append(f"    joint centre {j:8s} {np.round(r.landmarks[f'{j}_l'].pos, 3)} vs comment {v}: {1e3 * e:.1f} mm")
  for j, z in (("knee", 0.507), ("ankle", 0.107)):
    lines.append(f"    joint centre {j:8s} z {r.landmarks[f'{j}_l'].pos[2]:.4f} vs comment z {z}: "
                 f"{1e3 * abs(r.landmarks[f'{j}_l'].pos[2] - z):.1f} mm")
  rows = []
  pig = MS.load("plug_in_gait")
  left = pig.data["override"]["left"]
  for mk, lm in PIG_LANDMARK.items():
    for sd in ("L", "R") if mk in left else ("",):
      label = sd + mk
      lname = lm + (f"_{sd.lower()}" if sd else "")
      if lname not in r.landmarks:
        continue
      g = pig.anat_world(label)
      v = g - r.landmarks[lname].pos
      rows.append((label, lname, v, np.linalg.norm(v)))
  for label, lname, v, n in rows:
    if label.startswith("R"):
      continue
    lines.append(f"    {label:5s} guess - {lname:16s} = ({1e3 * v[0]:+5.0f}, {1e3 * v[1]:+5.0f}, "
                 f"{1e3 * v[2]:+5.0f}) mm, |d| {1e3 * n:4.0f} mm")
  # sole points vs ik.MarkerModel
  from mocap2myo.ik import FREE, MarkerModel
  mm = MarkerModel()
  nt = L.Neutral(p, "root")
  sp = L.sole_points(nt)
  same = (len(sp) == len(mm.fp_kind) and
          all(np.allclose(s["local"], mm.fp_local[i], atol=1e-9) and (s["side"], s["kind"]) == mm.fp_kind[i]
              and nt.m.body(s["body"]).id == mm.fp_body[i] for i, s in enumerate(sp)))
  lines.append(f"    sole points == ik.MarkerModel foot points (body, local, side, kind; same order): {same}")
  lines.append(f"    IK joints == ik.FREE: {p.ik_joints(nt.m) == list(FREE)}")
  from mocap2myo.kinetics import JOINTS
  kin = {k: (v["distal"], v["point"], v["frame"], [tuple(c) for c in v["comps"]]) for k, v in p.kinetics.items()}
  lines.append(f"    kinetics joints == kinetics.JOINTS: {kin == JOINTS}")
  return r


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("names", nargs="*", help="profiles (default: all)")
  ap.add_argument("--out", default=str(prof.MODELS_DIR / "_check"),
                  help="output folder for the images and landmarks.json (default: M2M_MODELS_DIR/_check)")
  ap.add_argument("--no-images", action="store_true")
  a = ap.parse_args()
  names = a.names or prof.profile_names()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  summary = []
  rng = np.random.default_rng(0)
  for name in names:
    p = prof.load(name)
    print("=" * 100)
    print(f"{name}: {p.title}")
    try:
      p.check_available()
    except FileNotFoundError as e:
      print(f"  SKIPPED: {e}")
      summary.append((name, "not available"))
      continue
    print(f"  file: {p.xml}")
    m = p.compile()
    lines = []
    nfree = sum(1 for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    ik = p.ik_joints(m)
    nmus = muscles(m)
    lines.append(f"counts: bodies {m.nbody - 1}, joints {m.njnt} (free {nfree}), nq {m.nq}, "
                 f"IK joints {len(ik)} (+6 root), actuators {m.nu}, muscles {nmus}, tendons {m.ntendon}, "
                 f"meshes {m.nmesh}, equalities {m.neq}, mass {m.body_subtreemass[0]:.1f} kg")
    lines.append(f"IK joints: {', '.join(ik)}")
    # canonical map
    lines.append("canonical segments:")
    for sname in prof.CANONICAL_SEGMENTS:
      seg = p.segments.get(sname)
      if seg is None:
        lines.append(f"  {sname:11s} -")
        continue
      try:
        rig = rigid_group(m, seg.bodies)
      except KeyError as e:
        lines.append(f"  {sname:11s} MISSING BODY {e}")
        continue
      g = f" geoms {list(seg.geoms)}" if seg.geoms else ""
      lines.append(f"  {sname:11s} {seg.body}" + (f" (+ {', '.join(seg.bodies[1:])})" if len(seg.bodies) > 1 else "")
                   + g + ("" if rig else "  NOT RIGID"))
    eq_lines, eq = equality_report(m, p, ik, rng)
    lines += eq_lines
    res = L.compute(p, "floor", m=m)
    nt = L.Neutral(p, "floor", m)
    # orientation sanity
    lm = res.landmarks
    checks = []
    if "head_top" in lm and "hip_l" in lm:
      checks.append(("head above hips", lm["head_top"].pos[2] > lm["hip_l"].pos[2]))
    if "hip_l" in lm:
      checks.append(("left hip at +x", lm["hip_l"].pos[0] > 0))
    if "toe_l" in lm and "heel_l" in lm:
      checks.append(("toes in front of heels (-y)", lm["toe_l"].pos[1] < lm["heel_l"].pos[1]))
    if "asis_l" in lm and "psis_l" in lm:
      checks.append(("ASIS in front of PSIS", lm["asis_l"].pos[1] < lm["psis_l"].pos[1]))
    lines.append("orientation: " + ", ".join(f"{c} {'ok' if v else 'FAILED'}" for c, v in checks))
    lines.append(f"height: bone (head top - foot-bone sole) {res.bone_height and round(res.bone_height, 4)} m, "
                 f"model height (+ soft tissue {p.soft_tissue} m) {res.model_height and round(res.model_height, 4)} m")
    lines.append("landmarks (canonical frame, neutral pose, feet on z = 0; m):")
    for n in L.LANDMARKS + L.LANDMARKS_EXTRA + ["lumbar"]:
      if n in lm:
        l = lm[n]
        lines.append(f"  {n:17s} {l.pos[0]:+.3f} {l.pos[1]:+.3f} {l.pos[2]:+.3f}  {l.body:14s} "
                     f"{str(l.segment):10s} {l.how}")
    if res.missing:
      lines.append("not found: " + "; ".join(f"{k} ({v})" for k, v in res.missing.items()))
    absent = [n for n in L.LANDMARKS if n not in lm and n not in res.missing]
    if absent:
      lines.append("absent (segment / joint centre not in the model): " + ", ".join(absent))
    kl, kok = kinetics_report(nt, p)
    lines.append("kinetics joints:" + ("" if kl else " none"))
    lines += kl
    sym, sym_worst = symmetry_report(nt, ik)
    lines.append(f"left / right symmetry (each IK joint pair moved {0.3} rad): largest mirror difference "
                 f"{1e3 * sym_worst:.1f} mm" + ("" if not sym else ": " + "; ".join(sym)))
    sole = L.sole_points(nt) if p.contact.get("sole_bodies") else []
    if sole:
      kinds = {}
      for s in sole:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
      lines.append(f"sole points ({p.contact.get('sole_geoms', 'collision')}): {len(sole)} {kinds}; "
                   f"lowest z {min(s['pos'][2] for s in sole):+.4f}")
    if name == "myofullbody":
      compare_myofullbody(p, lines)
    for l in lines:
      print("  " + l)
    od = out / name
    od.mkdir(exist_ok=True)
    js = dict(profile=name, title=p.title, xml=str(p.xml), bone_height=res.bone_height,
              model_height=res.model_height, ik_joints=ik, muscles=nmus,
              landmarks={n: dict(pos=l.pos.round(5).tolist(), body=l.body, segment=l.segment,
                                 local=l.local.round(5).tolist(), how=l.how) for n, l in lm.items()},
              missing=res.missing, equalities=eq,
              sole=[dict(body=s["body"], local=np.round(s["local"], 5).tolist(), side=s["side"],
                         kind=s["kind"]) for s in sole])
    (od / "landmarks.json").write_text(json.dumps(js, indent=1))
    img = None
    if not a.no_images:
      img = render(p, res, sole, out)
      print(f"  image: {img}")
    summary.append((name, f"IK {len(ik):3d}+6, muscles {nmus:3d}, landmarks {sum(n in lm for n in L.LANDMARKS):2d}/{len(L.LANDMARKS)}, "
                          f"bone height {res.bone_height or float('nan'):.3f}, kinetics signs {'ok' if kok else 'CHECK'}, "
                          f"orientation {'ok' if all(v for _, v in checks) else 'CHECK'}, "
                          f"symmetry {1e3 * sym_worst:.1f} mm"))
  print("=" * 100)
  for n, s in summary:
    print(f"{n:24s} {s}")


if __name__ == "__main__":
  main()
