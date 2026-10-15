"""Motion-dataset writer of the lift datasets (the format the imitation environment
reads: one npz with all clips concatenated + their initial poses; box pose, grasp
and phase per frame)."""

import os

import numpy as np

FPS = 100.0
TRACKED = ("pelvis", "tibia_l", "tibia_r", "talus_l", "talus_r", "toes_l", "toes_r",
           "thorax", "head", "humerus_l", "ulna_l", "thirdmc_l",
           "humerus_r", "ulna_r", "thirdmc_r")


def write(out_dir, tag, out, index, m, keep=None, stem="lift_mmh"):
  """Save clips (all, or the `keep` ids in order) in our motion format."""
  keep = list(range(len(index))) if keep is None else keep
  sel = lambda k: [out[k][i] for i in keep]
  n = np.array([index[i]["frames"] for i in keep])
  cat = lambda k: np.concatenate(sel(k)).astype(np.float32)
  np.savez(os.path.join(out_dir, f"motion_{stem}{tag}_100hz.npz"),
           gts=cat("gts"),
           gvs=np.concatenate([np.gradient(g, 1 / FPS, axis=0) for g in sel("gts")]).astype(np.float32),
           root_pos=cat("root_pos"),
           root_vel=np.concatenate([np.gradient(r, 1 / FPS, axis=0)
                                    for r in sel("root_pos")]).astype(np.float32),
           root_quat=cat("root_quat"), box_pos=cat("box_pos"), box_quat=cat("box_quat"),
           grasp=cat("grasp"), phase=np.concatenate(sel("phase")),
           motion_num_frames=n.astype(np.int64),
           length_starts=np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64),
           motion_fps=np.full(len(n), FPS, np.float32), motion_dt=np.full(len(n), 1 / FPS, np.float32),
           motion_lengths=(n / FPS).astype(np.float32), tracked_bodies=np.array(TRACKED),
           obj_kind=np.array([index[i]["object"] for i in keep]),
           obj_size=np.array([index[i]["obj_size"] for i in keep], np.float32),
           box_mass=np.float32(2.0),
           grab_step=np.array([index[i]["grab_step"] for i in keep]),
           place_step=np.array([index[i]["place_step"] for i in keep]),
           clip_technique=np.array([index[i]["technique"] for i in keep]),
           clip_file=np.array([index[i]["file"] for i in keep]),
           source_clip=np.array(keep))
  np.savez(os.path.join(out_dir, f"initial_poses_{stem}{tag}_100hz.npz"),
           qpos_joints=cat("qj"),
           joint_names=np.array([m.joint(j).name for j in range(1, m.njnt)]))
