"""Independent check of an IK run against an Xsens suit recorded at the same time.

One trial (M2M_XSENS_TRIAL) against its Xsens export (M2M_XSENS_C3D: a c3d of MVN's
64 virtual anatomical points). Case study: only S11 Spring NE has a usable one
(60 Hz; the other trials are raw .mvn, which needs MVN Analyze to export; S11
Solid's c3d is empty). The same geometric angles are computed on both sides:
  knee flexion   180 deg - angle(hip centre -> knee centre, knee centre -> ankle centre)
  elbow flexion  same with shoulder -> elbow -> wrist
  trunk lean     angle between vertical and (pelvis centre -> C7)
IK side: joint centres = MyoFullBody body origins (femur, tibia, talus, humerus,
ulna, lunate); trunk from the subject-calibrated virtual markers (ASIS / PSIS
centre -> C7). Xsens side: knee / ankle / elbow / wrist centres = midpoints of
the epicondyles / malleoli / styloids, shoulder = acromion, hip = Harrington
2007 regression from the ASIS / PSIS points (pRightCSI = posterior superior
iliac spine). The two recordings are aligned in time by cross-correlating the
right knee angle. Xsens has its own error (several degrees against optical
systems), so this is supporting evidence, not ground truth.

  python pipeline/validate/xsens.py IK_DIR        (needs ezc3d)
"""

import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pipeline import config  # noqa: E402

TRIAL = config.XSENS_TRIAL
C3D = config.XSENS_C3D


def angle(a, b):
  c = np.einsum("ti,ti->t", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
  return np.degrees(np.arccos(np.clip(c, -1, 1)))


def flex(p, j, d):
  return 180.0 - angle(j - p, d - j)


def lean(pelvis, c7):
  return angle(c7 - pelvis, np.tile([0, 0, 1.0], (len(c7), 1)))


def harrington_hjc(rasi, lasi, rpsi, lpsi):
  """Hip joint centres (Harrington et al. 2007), metres."""
  out = {}
  mid_a, mid_p = (rasi + lasi) / 2, (rpsi + lpsi) / 2
  for side, s in (("r", 1), ("l", -1)):
    y = (rasi - lasi) / np.linalg.norm(rasi - lasi, axis=1, keepdims=True)  # to the right
    fwd = mid_a - mid_p
    fwd -= np.einsum("ti,ti->t", fwd, y)[:, None] * y
    x = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    z = np.cross(x, y)  # down: x forward, y right
    pw = np.linalg.norm(rasi - lasi, axis=1, keepdims=True)
    pd = np.linalg.norm(mid_a - mid_p, axis=1, keepdims=True)
    out[side] = (mid_a + x * (-0.24 * pd - 0.0099) + s * y * (0.33 * pw + 0.0073)
                 + z * (0.30 * pw + 0.0109))
  return out


def xsens_angles(path):
  import ezc3d  # optional dependency (pip install -e ".[xsens]"), only needed here
  c = ezc3d.c3d(path)
  lab = c["parameters"]["POINT"]["LABELS"]["value"]
  rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
  X = c["data"]["points"][:3].transpose(2, 1, 0) / 1000.0  # T, n, 3 (m)
  g = lambda n: X[:, lab.index(n)]
  hjc = harrington_hjc(g("pRightASI"), g("pLeftASI"), g("pRightCSI"), g("pLeftCSI"))
  a = {}
  for s, S in (("r", "Right"), ("l", "Left")):
    knee = (g(f"p{S}KneeLatEpicondyle") + g(f"p{S}KneeMedEpicondyle")) / 2
    ank = (g(f"p{S}LatMalleolus") + g(f"p{S}MedMalleolus")) / 2
    elb = (g(f"p{S}ArmLatEpicondyle") + g(f"p{S}ArmMedEpicondyle")) / 2
    wri = (g(f"p{S}UlnarStyloid") + g(f"p{S}RadialStyloid")) / 2
    a[f"knee_{s}"] = flex(hjc[s], knee, ank)
    a[f"elbow_{s}"] = flex(g(f"p{S}Acromion"), elb, wri)
  pel = (g("pRightASI") + g("pLeftASI") + g("pRightCSI") + g("pLeftCSI")) / 4
  a["trunk"] = lean(pel, g("pC7SpinalProcess"))
  return a, rate


def ik_angles(ik_dir):
  K = np.load(os.path.join(ik_dir, f"{TRIAL}.npz"))
  Q = K["qpos"].astype(np.float64)
  mm = config.marker_model()
  mm.set_offsets(np.load(os.path.join(ik_dir, f"{config.subject_of(TRIAL)}_offsets.npz"))["offsets"])
  m, d, p = mm.m, mm.d, mm.profile
  # hip, knee, ankle, shoulder, elbow, wrist: the origins of the femur, tibia, ankle-joint
  # body, humerus, ulna and wrist-joint body (MyoFullBody: talus, lunate)
  jb = lambda n: m.body(int(m.jnt_bodyid[m.joint(p.joint_centres[n][4:]).id])).name \
    if p.joint_centres[n].startswith("jnt:") else p.joint_centres[n][5:]
  bodies = [b for s in "rl" for b in (p.segments[f"femur_{s}"].body, p.segments[f"tibia_{s}"].body,
                                       jb(f"ankle_{s}"), p.segments[f"humerus_{s}"].body,
                                       p.segments[f"ulna_{s}"].body, jb(f"wrist_{s}"))]
  names = [f"{n}_{s}" for s in "rl" for n in ("femur", "tibia", "talus", "humerus", "ulna", "lunate")]
  bid = [m.body(b).id for b in bodies]
  L = mm.ms.LABELS
  pk = mm.ms.PELVIS
  B = np.full((len(Q), len(bodies), 3), np.nan)
  M = np.full((len(Q), len(L), 3), np.nan)
  for t in range(len(Q)):
    if np.isnan(Q[t]).any():
      continue
    d.qpos[:] = Q[t]
    mujoco.mj_kinematics(m, d)
    B[t] = d.xpos[bid]
    M[t] = d.site_xpos[mm.sid]
  b = {n: B[:, i] for i, n in enumerate(names)}
  a = {}
  for s in "rl":
    a[f"knee_{s}"] = flex(b[f"femur_{s}"], b[f"tibia_{s}"], b[f"talus_{s}"])
    a[f"elbow_{s}"] = flex(b[f"humerus_{s}"], b[f"ulna_{s}"], b[f"lunate_{s}"])
  a["trunk"] = lean(M[:, pk].mean(1), M[:, L.index("C7")])
  return a, float(K["rate"])


def resample(x, rate_in, rate_out, n_out=None):
  t_in = np.arange(len(x)) / rate_in
  n = n_out or int(t_in[-1] * rate_out) + 1
  t = np.arange(n) / rate_out
  ok = ~np.isnan(x)
  return np.interp(t, t_in[ok], x[ok], left=np.nan, right=np.nan)


def main():
  config.usage(__doc__)
  ik_dir = sys.argv[1]
  xa, xr = xsens_angles(C3D)
  ia, ir = ik_angles(ik_dir)
  xs = {k: resample(v, xr, ir) for k, v in xa.items()}
  # lag (in IK frames) maximising the right-knee correlation, |lag| <= 20 s
  a, b = ia["knee_r"], xs["knee_r"]
  best = None
  for lag in range(-int(20 * ir), int(20 * ir) + 1):
    if lag >= 0:
      u, v = a[lag:], b[:len(b)]
    else:
      u, v = a, b[-lag:]
    n = min(len(u), len(v))
    u, v = u[:n], v[:n]
    ok = ~np.isnan(u) & ~np.isnan(v)
    if ok.sum() < 0.5 * min(len(a), len(b)):
      continue
    r = np.corrcoef(u[ok], v[ok])[0, 1]
    if best is None or r > best[0]:
      best = (r, lag)
  r0, lag = best
  print(f"{TRIAL}: Xsens {len(xa['trunk'])} frames @ {xr:g} Hz, IK {len(ia['trunk'])} @ {ir:g} Hz; "
        f"best lag {lag / ir:+.2f} s (right-knee r = {r0:.3f})")
  print(f"{'angle':9s} {'r':>6s} {'mean diff':>10s} {'RMS diff':>9s} {'RMS after offset':>17s} "
        f"{'IK range':>13s} {'Xsens range':>13s}  (deg; diff = IK - Xsens)")
  for k in ia:
    u = ia[k][lag:] if lag >= 0 else ia[k]
    v = xs[k] if lag >= 0 else xs[k][-lag:]
    n = min(len(u), len(v))
    u, v = u[:n], v[:n]
    ok = ~np.isnan(u) & ~np.isnan(v)
    dlt = u[ok] - v[ok]
    print(f"{k:9s} {np.corrcoef(u[ok], v[ok])[0, 1]:6.3f} {dlt.mean():+10.1f} "
          f"{np.sqrt(np.mean(dlt ** 2)):9.1f} {np.std(dlt):17.1f} "
          f"{np.percentile(u[ok], 2):5.0f}-{np.percentile(u[ok], 98):<5.0f} "
          f"{np.percentile(v[ok], 2):5.0f}-{np.percentile(v[ok], 98):<5.0f}")


if __name__ == "__main__":
  main()
