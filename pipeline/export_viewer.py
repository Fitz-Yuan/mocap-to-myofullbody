"""Export MyoFullBody motions to the browser viewer (docs/viewer/).

Two kinds of file, both in the ELVW container (below):
  model  (once)     bone meshes (int16 vertices, uint16 faces per mesh), the body
                    tree, every actuated muscle as its chain of attachment sites
                    (body + local position; drawn as straight lines, wrapping
                    surfaces not exported), the contact capsules / ellipsoids
                    (a rough body volume), and the body poses at qpos0 (shown
                    before a clip is loaded).
  clip   (per clip of the task's reference, or whole trial)   body poses per frame
                    (the viewer needs no kinematics of its own), measured and model
                    markers, joint-angle curves, per-frame marker residual, the trial's
                    QA grades, and two task-neutral lists the viewer draws as given:
                    "objects" (shape, size, a fixed pose or per-frame pose arrays) and
                    "tracks" (named per-frame states, binary or categorical), both
                    from the task (M2M_TASK; box_lift: the box, the shelf, the
                    platform, each hand's contact with the box and what supports
                    the box, tasks/box_lift/viewer_scene.py). Without a task
                    (M2M_TASK=none) the clip is read from the IK output instead of
                    the reference and has no objects (and --clip is not available).
                    Optional ghost: the same frames from a second run (e.g. another
                    calibration) for an overlay.
                    With the inverse dynamics (pipeline/kinetics.py) the clip also gets
                    joint-load and ground-reaction plots and force arrows, and with the
                    OpenSim comparison (validate/opensim_ik.py, pipeline/opensim_id.py)
                    the OpenSim stick figure, markers and dashed curves
                    (pipeline/viewer_layers.py); both are included when their files exist.
No subject data goes into the model file; clips stay wherever they are written
and are opened in the viewer from disk (nothing is uploaded).

ELVW container: b"ELVW", uint32 version, uint32 header length, UTF-8 JSON header
{"meta": {...}, "arrays": {name: [dtype, shape, byte offset]}}, zero padding to a
multiple of 8, then the arrays (little endian, each 8-byte aligned). Positions are
int16 with meta["pos_scale"] metres per unit, quaternions int16 / 32767 (w, x, y,
z), missing markers -32768.

  python pipeline/export_viewer.py model OUT_DIR
  python pipeline/export_viewer.py clip TRIAL OUT_DIR [--clip K] [--fps 50]
      [--ik-dir DIR] [--ref-dir DIR] [--npz-dir DIR] [--run-label TEXT]
      [--ghost-ref-dir DIR [--ghost-label TEXT]]
      [--kin-dir DIR] [--osim-id-dir DIR] [--osim-ik-dir DIR] [--public NAME]
  --clip K           clip K (1..n) of the task's reference (box_lift: carry cycle K);
                     0 (default) = the whole trial
  --ik-dir, --ref-dir, --npz-dir   inputs (default M2M_IK_DIR, M2M_REF_DIR, M2M_NPZ_DIR)
  --run-label TEXT   name of the run shown in the viewer (default: the IK folder's name)
  --ghost-ref-dir DIR, --ghost-label TEXT   overlay a second run's reference (labelled
                     TEXT, default "second run")
  --kin-dir DIR      inverse dynamics (pipeline/kinetics.py; default M2M_KIN_DIR)
  --osim-id-dir DIR, --osim-ik-dir DIR   the OpenSim comparison (pipeline/opensim_id.py,
                     validate/opensim_ik.py)
  --public NAME      a de-identified clip for a public page, named NAME: the trial name,
                     the clip's position in the trial, the subject's scale and the QA
                     grades are left out, and the run is labelled --run-label or "demo"
                     (never a folder name)
"""

import argparse
import json
import os
import struct
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mocap2myo import ik as mk  # noqa: E402
from mocap2myo.model import MYOFULLBODY_XML  # noqa: E402
from pipeline import config  # noqa: E402
from pipeline import viewer_layers  # noqa: E402

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
  task = config.task()
  # the task's reference (floor-aligned, with its objects and clips), else the IK
  ref = np.load(os.path.join(a.ref_dir if task is not None else a.ik_dir, f"{trial}.npz"))
  rate, s = float(ref["rate"]), float(ref["scale"])
  T = len(ref["qpos"])
  if a.clip:
    if "clips" not in ref.files:
      raise SystemExit("--clip needs the task's reference (clips); without a task "
                       "(M2M_TASK=none) export the whole trial")
    if not 1 <= a.clip <= len(ref["clips"]):
      raise SystemExit(f"--clip {a.clip}: {trial} has clips 1..{len(ref['clips'])}")
    lo, hi = (int(x) for x in ref["clips"][a.clip - 1])
  else:
    lo, hi = 0, T
  step = max(1, int(round(rate / a.fps)))
  idx = np.arange(lo, hi, step)
  fps = rate / step
  # an IK run with the optional neck stores its angles apart: draw the head with them
  ikf = np.load(os.path.join(a.ik_dir, f"{trial}.npz"))
  neck = ikf["neck_qpos"][idx].astype(np.float64) if "neck_qpos" in ikf.files else None
  mm = mk.MarkerModel(neck=neck is not None)
  m, d = mm.m, mm.d
  subj = config.subject_of(trial)
  mm.set_offsets(np.load(os.path.join(a.ik_dir, f"{subj}_offsets.npz"))["offsets"])
  Q = ref["qpos"][idx].astype(np.float64)
  if neck is not None:
    Q = mm.full_qpos(Q, neck)
  bp, bq = body_poses(m, d, Q)
  model_mk = np.stack([mm.markers(q) for q in Q])
  # measured markers in the reference frame: scaled like the IK, moved with the human
  P, _, _ = mk.load_trial(os.path.join(a.npz_dir, f"{trial}.npz"))
  dz = float(ref["human_dz"]) if "human_dz" in ref else 0.0
  meas = P[idx] + np.array([0.0, 0.0, dz])
  head = mk.HEAD
  err = np.linalg.norm(model_mk - meas, axis=-1)
  resid = np.nanmedian(np.where(head, np.nan, err), axis=1)
  adr = [m.jnt_qposadr[m.joint(j).id] for j in CURVES]
  curves = np.degrees(Q[:, adr]).astype(np.float32)
  # task arrays: {name: (kind, array)}, kind "pos" (m) / "quat" (wxyz) / "raw"
  objects, tracks, extra = task.viewer_scene(ref, idx) if task is not None else ([], [], {})
  extent = np.nanmax(np.abs(np.concatenate(
    [bp.reshape(-1, 3)] + [x.reshape(-1, 3) for kind, x in extra.values() if kind == "pos"])))
  ps = max(2e-4, float(extent) / 32000)
  arrays = dict(
    body_pos=q16(bp, ps), body_quat=q16(bq, 1 / 32767),
    marker_meas=q16(meas, ps), marker_model=q16(model_mk, ps))
  for name, (kind, x) in extra.items():
    arrays[name] = q16(x, ps) if kind == "pos" else q16(x, 1 / 32767) if kind == "quat" else x
  arrays.update(curves=curves, resid_mm=(1e3 * resid).astype(np.float32))
  ghost = None
  if a.ghost_ref_dir:
    gref = np.load(os.path.join(a.ghost_ref_dir, f"{trial}.npz"))
    gQ = gref["qpos"][idx].astype(np.float64)
    if mm.neck:  # the ghost run is drawn with a straight neck
      gQ = mm.full_qpos(gQ, np.zeros((len(gQ), len(mk.NECK))))
    gp, gq = body_poses(m, d, gQ)
    arrays["ghost_pos"], arrays["ghost_quat"] = q16(gp, ps), q16(gq, 1 / 32767)
    arrays["ghost_curves"] = np.degrees(gQ[:, adr]).astype(np.float32)
    ghost = a.ghost_label or "second run"
  # inverse dynamics and the OpenSim comparison, when their files exist
  opt = lambda d, f: np.load(os.path.join(d, f)) if d and os.path.exists(os.path.join(d, f)) else None
  kin = opt(a.kin_dir, f"{trial}.npz")
  osim = opt(a.osim_id_dir, f"{trial}.npz")
  omk = opt(a.osim_ik_dir, os.path.join(subj, f"{trial}_markers.npz"))
  la, lm = viewer_layers.layers(m, Q, idx, s, dz, q16, ps, kin=kin, osim=osim,
                                osim_markers=None if omk is None else omk["markers"])
  arrays.update(la)
  qa = {}
  rep = os.path.join(a.ik_dir, "report.json")
  if os.path.exists(rep):
    qa = json.load(open(rep)).get(trial, {})
  meta = dict(
    format="elvw-clip-3", trial=trial, clip=a.clip,
    clip_label=task.CLIP_LABEL if task is not None else "clip", frames=len(idx),
    fps=fps, start_s=lo / rate, pos_scale=ps, subject_scale=s, labels=mk.LABELS,
    curves=CURVES, objects=objects, tracks=tracks,
    run=a.run_label or os.path.basename(os.path.normpath(a.ik_dir)), ghost=ghost, **lm,
    qa={k: qa[k] for k in ("verdict", "resid_med_mm", "resid_p95_mm", "lomo_med_mm",
                           "stand_knee_deg", "stand_hip_deg", "gap_max", "gap_worst")
        if k in qa})
  if a.public:  # nothing that identifies the trial or the subject
    meta.update(trial=a.public, clip=0, start_s=0.0, run=a.run_label or "demo")
    meta.pop("subject_scale")
    meta.pop("qa")  # its values match the published per-trial QA table
  os.makedirs(a.out_dir, exist_ok=True)
  name = f"{trial}{f'_c{a.clip}' if a.clip else ''}.elvw"
  if a.public:
    name = a.public.lower().replace(" ", "_") + ".elvw"
  size = write_elvw(os.path.join(a.out_dir, name), meta, arrays)
  print(f"{name}: {len(idx)} frames @ {fps:g} Hz, residual median {np.nanmedian(resid) * 1e3:.1f} mm, "
        f"{len(lm['plots'])} plots, kinetics {'yes' if kin is not None else 'no'}, "
        f"OpenSim {'yes' if lm['compare'] else 'no'}, {size / 1e6:.2f} MB")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  pm = sub.add_parser("model")
  pm.add_argument("out_dir")
  pc = sub.add_parser("clip")
  pc.add_argument("trial")
  pc.add_argument("out_dir")
  pc.add_argument("--clip", type=int, default=0,
                  help="clip of the task's reference 1..n (box_lift: carry cycle; 0 = whole trial)")
  pc.add_argument("--fps", type=float, default=50.0, help="frame rate of the clip")
  pc.add_argument("--ik-dir", default=config.IK_DIR)
  pc.add_argument("--ref-dir", default=config.REF_DIR)
  pc.add_argument("--npz-dir", default=config.NPZ_DIR)
  pc.add_argument("--run-label", default="", help="run name shown in the viewer")
  pc.add_argument("--ghost-ref-dir", default="", help="a second run's reference to overlay")
  pc.add_argument("--ghost-label", default="", help='its name in the viewer (default "second run")')
  pc.add_argument("--kin-dir", default=config.KIN_DIR, help="inverse dynamics (kinetics.py)")
  pc.add_argument("--osim-id-dir", default=config.OSIM_ID_DIR, help="OpenSim joints / ID (opensim_id.py)")
  pc.add_argument("--osim-ik-dir", default=config.OSIM_DIR, help="OpenSim IK (validate/opensim_ik.py)")
  pc.add_argument("--public", default="", help="name for a de-identified demo clip")
  a = ap.parse_args()
  if a.cmd == "model":
    os.makedirs(a.out_dir, exist_ok=True)
    export_model(a.out_dir)
  else:
    export_clip(a)


if __name__ == "__main__":
  main()
