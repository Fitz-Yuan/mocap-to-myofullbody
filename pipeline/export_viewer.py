"""Export MyoFullBody motions to the browser viewer (viewer/elevator_viewer.html).

Two kinds of file, both in the ELVW container (below):
  model  (once)     bone meshes (int16 vertices, uint16 faces per mesh), the body
                    tree, every actuated muscle as its chain of attachment sites
                    (body + local position; drawn as straight lines, wrapping
                    surfaces not exported), the contact capsules / ellipsoids
                    (a rough body volume), and the body poses at qpos0 (shown
                    before a clip is loaded).
  clip   (per carry cycle or whole trial)   body poses per frame (the viewer needs no
                    kinematics of its own), measured and model markers, joint-angle
                    curves, per-frame marker residual, the trial's QA grades, and two
                    task-neutral lists the viewer draws as given: "objects" (shape,
                    size, a fixed pose or per-frame pose arrays: here the box, the
                    shelf, the elevator platform) and "tracks" (named per-frame
                    states, binary or categorical: here each hand's contact with the
                    box and what supports the box). Optional ghost: the same frames
                    from a second IK run (e.g. another calibration) for an overlay.
No subject data goes into the model file; clips stay wherever they are written
and are opened in the viewer from disk (nothing is uploaded).

ELVW container: b"ELVW", uint32 version, uint32 header length, UTF-8 JSON header
{"meta": {...}, "arrays": {name: [dtype, shape, byte offset]}}, zero padding to a
multiple of 8, then the arrays (little endian, each 8-byte aligned). Positions are
int16 with meta["pos_scale"] metres per unit, quaternions int16 / 32767 (w, x, y,
z), missing markers -32768.

  uv run python logs/rsl_rl/fullbody_liftbox/scripts/elev_export_viewer.py model OUT_DIR
  uv run python .../elev_export_viewer.py clip TRIAL OUT_DIR [--clip K] [--fps 50]
      [--ik-dir DIR --ref-dir DIR --npz-dir DIR] [--ghost-ik-dir DIR --ghost-ref-dir DIR]
"""

import argparse
import json
import os
import struct
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmh_ik as mk  # noqa: E402
from mocap2myo.model import MYOFULLBODY_XML  # noqa: E402

ROOT = os.path.expanduser(os.environ.get("ELEV_ROOT", "~/data/mmh_elevator"))
MISSING = -32768
CURVES = ["knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_flexion_l",
          "hip_adduction_r", "hip_adduction_l", "ankle_angle_r", "ankle_angle_l",
          "flex_extension", "shoulder_elv_r", "shoulder_elv_l", "elbow_flex_r", "elbow_flex_l"]


def write_elvw(path, meta, arrays):
  index, blobs, off = {}, [], 0
  for name, a in arrays.items():
    a = np.ascontiguousarray(a)
    index[name] = [a.dtype.str.lstrip("<|"), list(a.shape), off]
    b = a.tobytes()
    blobs.append(b + b"\0" * (-len(b) % 8))
    off += len(blobs[-1])
  head = json.dumps({"meta": meta, "arrays": index}, separators=(",", ":")).encode()
  head += b" " * (-(12 + len(head)) % 8)
  tmp = path + ".part"
  with open(tmp, "wb") as f:
    f.write(b"ELVW" + struct.pack("<II", 1, len(head)) + head)
    for b in blobs:
      f.write(b)
  os.replace(tmp, path)
  return os.path.getsize(path)


def q16(x, scale):
  y = np.round(np.asarray(x, np.float64) / scale)
  out = np.clip(np.nan_to_num(y, nan=MISSING), -32767, 32767).astype(np.int16)
  out[np.isnan(np.asarray(x)).any(-1) if np.ndim(x) > 1 else np.isnan(x)] = MISSING
  return out


def export_model(out_dir):
  m = mujoco.MjModel.from_xml_path(str(MYOFULLBODY_XML))
  bodies = [m.body(b).name for b in range(m.nbody)]
  parents = [int(m.body_parentid[b]) for b in range(m.nbody)]
  meshes, verts, faces = [], [], []
  vofs = fofs = 0
  geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
  items = [(m.geom(g).name, int(m.geom_bodyid[g]), int(m.geom_dataid[g]), m.geom_pos[g],
            m.geom_quat[g], m.geom_rgba[g]) for g in geoms]
  # the thorax (rib cage) mesh is in the model but no geom uses it: placed on the
  # thorax body with the mesh's own recentring transform
  used = {i[2] for i in items}
  for mi in range(m.nmesh):
    if mi not in used and m.mesh(mi).name in bodies:
      items.append((m.mesh(mi).name, bodies.index(m.mesh(mi).name), mi, m.mesh_pos[mi],
                    m.mesh_quat[mi], np.array([0.9, 0.88, 0.8, 1.0])))
  for name, body, mi, pos, quat, rgba in items:
    a, n = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
    v = m.mesh_vert[a:a + n].astype(np.float64)
    fa, fn = m.mesh_faceadr[mi], m.mesh_facenum[mi]
    f = m.mesh_face[fa:fa + fn].astype(np.uint16)
    lo, hi = v.min(0), v.max(0)
    sc = np.maximum(hi - lo, 1e-6) / 65534.0
    verts.append(np.round((v - lo) / sc - 32767).astype(np.int16))
    faces.append(f)
    meshes.append(dict(name=name, body=body, pos=[float(x) for x in pos],
                       quat=[float(x) for x in quat], lo=lo.tolist(), scale=sc.tolist(),
                       vofs=vofs, nv=int(n), fofs=fofs, nf=int(fn),
                       rgba=[round(float(x), 3) for x in rgba]))
    vofs += int(n)
    fofs += int(fn)
  muscles, site_body, site_pos = [], [], []
  site_index = {}
  for a in range(m.nu):
    t = int(m.actuator_trnid[a][0])
    chain = []
    for j in range(m.tendon_adr[t], m.tendon_adr[t] + m.tendon_num[t]):
      if m.wrap_type[j] == mujoco.mjtWrap.mjWRAP_SITE:
        s = int(m.wrap_objid[j])
        if s not in site_index:
          site_index[s] = len(site_body)
          site_body.append(int(m.site_bodyid[s]))
          site_pos.append(m.site_pos[s])
        chain.append(site_index[s])
    muscles.append(dict(name=m.actuator(a).name, sites=chain))
  kinds = {mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid"}
  volumes = [dict(type=kinds[m.geom_type[g]], body=int(m.geom_bodyid[g]),
                  pos=[float(x) for x in m.geom_pos[g]], quat=[float(x) for x in m.geom_quat[g]],
                  size=[float(x) for x in m.geom_size[g]])
             for g in range(m.ngeom)
             if m.geom_type[g] in kinds and (m.geom_contype[g] or m.geom_conaffinity[g])]
  d = mujoco.MjData(m)
  mujoco.mj_kinematics(m, d)
  meta = dict(format="elvw-model-1", source=os.path.basename(str(MYOFULLBODY_XML)),
              bodies=bodies, parents=parents, meshes=meshes, muscles=muscles,
              volumes=volumes, up="z")
  arrays = dict(verts=np.concatenate(verts), faces=np.concatenate(faces),
                site_body=np.array(site_body, np.uint16),
                site_pos=np.array(site_pos, np.float32),
                rest_pos=d.xpos.astype(np.float32), rest_quat=d.xquat.astype(np.float32))
  path = os.path.join(out_dir, "myofullbody_model.elvw")
  size = write_elvw(path, meta, arrays)
  print(f"{path}: {len(meshes)} meshes, {vofs} vertices, {fofs} faces, {len(muscles)} muscles, "
        f"{len(site_body)} sites, {len(volumes)} volumes, {size / 1e6:.2f} MB")


def hull2d(pts):
  """Convex hull (counter-clockwise) of 2-D points."""
  p = sorted(map(tuple, np.asarray(pts, float)))
  cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
  lo, up = [], []
  for q in p:
    while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
      lo.pop()
    lo.append(q)
  for q in reversed(p):
    while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
      up.pop()
    up.append(q)
  return [list(map(float, q)) for q in lo[:-1] + up[:-1]]


def elevator_scene(ref):
  """Objects and tracks of an MMH Elevator trial, in the viewer's generic form."""
  objects = [dict(name="Box", shape="box", size=[float(x) for x in ref["box_size"]],
                  pos_array="obj_box_pos", quat_array="obj_box_quat", style="carried")]
  xy = np.asarray(ref["shelf_xy"], float)
  (x0, y0), (x1, y1), top, th = xy.min(0), xy.max(0), float(ref["shelf_top"]), 0.025
  objects.append(dict(name="Shelf", shape="box", size=[x1 - x0 + 0.04, y1 - y0 + 0.04, th],
                      pos=[(x0 + x1) / 2, (y0 + y1) / 2, top - th / 2], style="furniture"))
  for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
    objects.append(dict(name="Shelf leg", shape="box", size=[0.03, 0.03, top - th],
                        pos=[float(x), float(y), (top - th) / 2], style="furniture"))
  objects.append(dict(name="Elevator platform", shape="polygon",
                      xy=hull2d(ref["elevator_xy"]), height=0.012, style="floor"))
  tracks = [dict(name="Left hand in contact with the box", short="L", array="trk_contact",
                 col=0, kind="binary"),
            dict(name="Right hand in contact with the box", short="R", array="trk_contact",
                 col=1, kind="binary"),
            dict(name="Box supported by", array="trk_box_support", kind="category",
                 values=["floor", "hands", "shelf"])]
  return objects, tracks


def body_poses(m, d, Q):
  pos = np.zeros((len(Q), m.nbody, 3))
  quat = np.zeros((len(Q), m.nbody, 4))
  for t, q in enumerate(Q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    pos[t], quat[t] = d.xpos, d.xquat
  return pos, quat


def export_clip(a):
  trial = a.trial
  ref = np.load(os.path.join(a.ref_dir, f"{trial}.npz"))
  rate, s = float(ref["rate"]), float(ref["scale"])
  T = len(ref["qpos"])
  if a.clip:
    lo, hi = (int(x) for x in ref["clips"][a.clip - 1])
  else:
    lo, hi = 0, T
  step = max(1, int(round(rate / a.fps)))
  idx = np.arange(lo, hi, step)
  fps = rate / step
  mm = mk.MarkerModel()
  m, d = mm.m, mm.d
  subj = trial.split("_")[0]
  mm.set_offsets(np.load(os.path.join(a.ik_dir, f"{subj}_offsets.npz"))["offsets"])
  Q = ref["qpos"][idx].astype(np.float64)
  bp, bq = body_poses(m, d, Q)
  model_mk = np.stack([mm.markers(q) for q in Q])
  # measured markers in the reference frame: scaled like the IK, moved with the human
  P, _, _ = mk.load_trial(os.path.join(a.npz_dir, f"{trial}.npz"))
  meas = P[idx] + np.array([0.0, 0.0, float(ref["human_dz"])])
  head = np.array([l.endswith("HD") for l in mk.LABELS])
  err = np.linalg.norm(model_mk - meas, axis=-1)
  resid = np.nanmedian(np.where(head, np.nan, err), axis=1)
  adr = [m.jnt_qposadr[m.joint(j).id] for j in CURVES]
  curves = np.degrees(Q[:, adr]).astype(np.float32)
  extent = np.nanmax(np.abs(np.concatenate([bp.reshape(-1, 3), ref["box_pos"][idx]])))
  ps = max(2e-4, float(extent) / 32000)
  arrays = dict(
    body_pos=q16(bp, ps), body_quat=q16(bq, 1 / 32767),
    marker_meas=q16(meas, ps), marker_model=q16(model_mk, ps),
    obj_box_pos=q16(ref["box_pos"][idx], ps), obj_box_quat=q16(ref["box_quat"][idx], 1 / 32767),
    trk_contact=(ref["grasp"][idx] > 0.5).astype(np.uint8),
    trk_box_support=ref["support"][idx].astype(np.uint8),
    curves=curves, resid_mm=(1e3 * resid).astype(np.float32))
  objects, tracks = elevator_scene(ref)
  ghost = None
  if a.ghost_ref_dir:
    gref = np.load(os.path.join(a.ghost_ref_dir, f"{trial}.npz"))
    gp, gq = body_poses(m, d, gref["qpos"][idx].astype(np.float64))
    arrays["ghost_pos"], arrays["ghost_quat"] = q16(gp, ps), q16(gq, 1 / 32767)
    arrays["ghost_curves"] = np.degrees(gref["qpos"][idx][:, adr]).astype(np.float32)
    ghost = a.ghost_label or os.path.basename(os.path.normpath(a.ghost_ref_dir))
  qa = {}
  rep = os.path.join(a.ik_dir, "report.json")
  if os.path.exists(rep):
    qa = json.load(open(rep)).get(trial, {})
  meta = dict(
    format="elvw-clip-2", trial=trial, clip=a.clip, clip_label="carry", frames=len(idx),
    fps=fps, start_s=lo / rate, pos_scale=ps, subject_scale=s, labels=mk.LABELS,
    curves=CURVES, objects=objects, tracks=tracks,
    run=a.run_label or os.path.basename(os.path.normpath(a.ik_dir)), ghost=ghost,
    qa={k: qa[k] for k in ("verdict", "resid_med_mm", "resid_p95_mm", "lomo_med_mm",
                           "stand_knee_deg", "stand_hip_deg", "gap_max", "gap_worst")
        if k in qa})
  os.makedirs(a.out_dir, exist_ok=True)
  name = f"{trial}{f'_c{a.clip}' if a.clip else ''}.elvw"
  size = write_elvw(os.path.join(a.out_dir, name), meta, arrays)
  print(f"{name}: {len(idx)} frames @ {fps:g} Hz, residual median {np.nanmedian(resid) * 1e3:.1f} mm, "
        f"{size / 1e6:.2f} MB")


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  pm = sub.add_parser("model")
  pm.add_argument("out_dir")
  pc = sub.add_parser("clip")
  pc.add_argument("trial")
  pc.add_argument("out_dir")
  pc.add_argument("--clip", type=int, default=0, help="carry cycle 1..n (0 = whole trial)")
  pc.add_argument("--fps", type=float, default=50.0)
  pc.add_argument("--ik-dir", default=os.path.join(ROOT, "ik_tak"))
  pc.add_argument("--ref-dir", default=os.path.join(ROOT, "ref_tak"))
  pc.add_argument("--npz-dir", default=os.path.join(ROOT, "npz_tak"))
  pc.add_argument("--run-label", default="")
  pc.add_argument("--ghost-ref-dir", default="")
  pc.add_argument("--ghost-label", default="")
  a = ap.parse_args()
  if a.cmd == "model":
    os.makedirs(a.out_dir, exist_ok=True)
    export_model(a.out_dir)
  else:
    export_clip(a)


if __name__ == "__main__":
  main()
