"""Motion clips for training (imitation learning): every trial's IK as clips at a fixed rate, in
the format of the box-lifting task's dataset without its task fields.

  python pipeline/export_motion.py OUT_DIR [--fps 100] [--stem NAME] [--min-s 1.0] [trial ...]

Per trial: the frames the IK solved, cut where it lost a limb for LOST_CUT_S or longer
(mocap2myo.ik.limb_lost) and where it did not solve a frame; stretches of at least --min-s
seconds are the clips. Each clip is resampled to --fps (joint angles and the root position
linearly, the root orientation by slerp) and lowered so that the feet stand on z = 0 (the
median floor height under the still feet, mocap2myo.kinetics.floor_from_soles). Positions are
the model's (the subject scaled to the model's size, as the IK solves them).

Writes OUT_DIR/motion_<stem>_<fps>hz.npz: gts [N, bodies, 3] (tracked bodies' positions), gvs
(their velocities), root_pos, root_vel, root_quat (w, x, y, z), motion_num_frames,
length_starts, motion_fps, motion_dt, motion_lengths (per clip), tracked_bodies, clip_file;
OUT_DIR/initial_poses_<stem>_<fps>hz.npz: qpos_joints [N, nq - 7], joint_names; and
OUT_DIR/index.json: per clip the trial, its start and end in the trial (s), frames, the trial's
verdict in ik/report.json (PASS / WARN / FAIL) and QA numbers.
"""

import argparse
import json
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config  # noqa: E402
from mocap2myo import ik as mk  # noqa: E402
from mocap2myo.kinetics import floor_from_soles  # noqa: E402
from tasks.box_lift.body import tracked_bodies  # noqa: E402

LOST_CUT_S = 0.2  # s: a limb lost this long or longer splits the trial


def runs(mask):
  """[(a, b)] of the runs of True in a 1-D bool array (b exclusive)."""
  d = np.diff(np.r_[0, mask.astype(int), 0])
  return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def good_frames(K, ms, rate):
  """Frames to keep: solved, and not inside a lost-limb stretch of LOST_CUT_S or longer."""
  E = K["marker_err"]
  ok = ~np.isnan(E).all(1)
  lost = mk.limb_lost(E, K["rejected"] if "rejected" in K.files else None, [str(x) for x in K["labels"]], ms)
  for v in lost.values():
    for a, b in runs(v):
      if (b - a) / rate >= LOST_CUT_S:
        ok[a:b] = False
  return ok


def resample(Q, rate, fps):
  """qpos [T, nq] at rate -> fps: root position and joints linearly, root orientation by slerp."""
  T = len(Q)
  t0 = np.arange(T) / rate
  t1 = np.arange(0.0, t0[-1] + 1e-9, 1.0 / fps)
  out = np.empty((len(t1), Q.shape[1]))
  for j in [0, 1, 2, *range(7, Q.shape[1])]:
    out[:, j] = np.interp(t1, t0, Q[:, j])
  rot = Rotation.from_quat(Q[:, [4, 5, 6, 3]])
  q = Slerp(t0, rot)(t1).as_quat()
  out[:, 3:7] = q[:, [3, 0, 1, 2]]
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("out_dir")
  ap.add_argument("trials", nargs="*")
  ap.add_argument("--fps", type=float, default=100.0)
  ap.add_argument("--stem", default=config.PROFILE_NAME or "motion")
  ap.add_argument("--min-s", type=float, default=1.0)
  a = ap.parse_args()
  os.makedirs(a.out_dir, exist_ok=True)
  names = a.trials or config.trials(config.IK_DIR)
  qa_p, rep_p = os.path.join(config.IK_DIR, "qa.json"), os.path.join(config.IK_DIR, "report.json")
  qa = json.load(open(qa_p)) if os.path.exists(qa_p) else {}
  rep = json.load(open(rep_p)) if os.path.exists(rep_p) else {}
  ms = config.marker_set()
  models = {}
  clips, index = [], []
  for n in names:
    K = np.load(os.path.join(config.IK_DIR, f"{n}.npz"))
    subj = config.subject_of(n)
    if subj not in models:
      mm = config.run_marker_model(config.IK_DIR, subj)
      tb = tracked_bodies(mm.profile, mm.m, partial=True)
      models[subj] = (mm, tb, [mm.m.body(b).id for b in tb])
    mm, tb, bid = models[subj]
    m, d = mm.m, mm.d
    rate = float(K["rate"])
    Q = K["qpos"].astype(np.float64)
    ok = good_frames(K, ms, rate)
    # the floor under the still feet, over the whole trial (model units)
    sole = np.zeros((len(Q), len(mm.fp_kind), 3))
    for t in np.flatnonzero(ok):
      d.qpos[:] = Q[t]
      mujoco.mj_kinematics(m, d)
      sole[t] = mm.foot_points()
    feet = [np.array([i for i, k in enumerate(mm.fp_kind) if k[0] == s]) for s in "lr"]
    floor = float(np.median(floor_from_soles(sole[ok], feet, rate)))
    for c, (s0, s1) in enumerate(runs(ok)):
      if (s1 - s0) / rate < a.min_s:
        continue
      q = resample(Q[s0:s1], rate, a.fps)
      q[:, 2] -= floor
      G = np.zeros((len(q), len(bid), 3))
      for i in range(len(q)):
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        G[i] = d.xpos[bid]
      clips.append((q, G))
      index.append(dict(clip=len(index), file=f"{n}_c{c + 1}", trial=n, subject=subj,
                        start_s=round(s0 / rate, 3), end_s=round(s1 / rate, 3), frames=len(q),
                        seconds=round(len(q) / a.fps, 2), floor_m=round(floor, 4),
                        trial_verdict=rep.get(n, {}).get("verdict", "-"), **{
                          k: qa[n][k] for k in ("marker_err_body_mm", "limb_lost_s") if k in qa.get(n, {})}))
    kept = sum((b - a_) for a_, b in runs(ok)) / rate
    print(f"{n}: {len(Q) / rate:.1f} s, {kept:.1f} s usable, "
          f"{sum(1 for x in index if x['trial'] == n)} clips", flush=True)
  if not clips:
    raise SystemExit("no clip long enough")
  fps = a.fps
  nfr = np.array([len(q) for q, _ in clips])
  tag = f"{a.stem}_{fps:g}hz"
  m0 = next(iter(models.values()))[0].m
  np.savez(os.path.join(a.out_dir, f"motion_{tag}.npz"),
           gts=np.concatenate([G for _, G in clips]).astype(np.float32),
           gvs=np.concatenate([np.gradient(G, 1 / fps, axis=0) for _, G in clips]).astype(np.float32),
           root_pos=np.concatenate([q[:, :3] for q, _ in clips]).astype(np.float32),
           root_vel=np.concatenate([np.gradient(q[:, :3], 1 / fps, axis=0) for q, _ in clips]).astype(np.float32),
           root_quat=np.concatenate([q[:, 3:7] for q, _ in clips]).astype(np.float32),
           motion_num_frames=nfr.astype(np.int64),
           length_starts=np.r_[0, np.cumsum(nfr)[:-1]].astype(np.int64),
           motion_fps=np.full(len(nfr), fps, np.float32), motion_dt=np.full(len(nfr), 1 / fps, np.float32),
           motion_lengths=(nfr / fps).astype(np.float32),
           tracked_bodies=np.array(next(iter(models.values()))[1]),
           clip_file=np.array([x["file"] for x in index]))
  np.savez(os.path.join(a.out_dir, f"initial_poses_{tag}.npz"),
           qpos_joints=np.concatenate([q[:, 7:] for q, _ in clips]).astype(np.float32),
           joint_names=np.array([m0.joint(j).name for j in range(1, m0.njnt)]))
  json.dump(dict(fps=fps, model=config.MODEL, clips=index), open(os.path.join(a.out_dir, "index.json"), "w"), indent=1)
  print(f"{len(clips)} clips, {nfr.sum() / fps:.1f} s at {fps:g} Hz -> {a.out_dir}/motion_{tag}.npz")


if __name__ == "__main__":
  main()
