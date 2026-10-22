"""Virtual markers on the SMPL mesh from a GVHMR result (video -> SMPL-X parameters).

Runs with GVHMR's own Python, in GVHMR's folder (it needs GVHMR's body-model utilities and
the SMPL-X model files the user downloaded for GVHMR); `mocap2myo video` calls it:

  cd GVHMR && python /path/to/mocap2myo/gvhmr_markers.py RESULT.pt OUT.npz [BETAS.npy]

GVHMR's world: gravity-aligned, Y up, metres. Its per-frame shape (betas) is averaged over
the frames, so the body keeps one shape; BETAS.npy (another video's betas) gives the shape
instead, so that every video of one person has the same body. Each marker is one SMPL
vertex (the SMPL-X mesh mapped onto SMPL's with GVHMR's smplx2smpl_sparse.pt); the vertex
choice is OpenCapBench's
(Gozlan et al. 2024, MIT License; see NOTICE). Writes OUT.npz: names, markers [T, n, 3]
(Y up), lowest [T, 3] (the mesh's lowest vertex per frame: the sole), static [n, 3] and
static_lowest [3] (the same body standing in SMPL's zero pose), stature (m, that pose's
height), betas [10].

This file imports nothing from mocap2myo: it runs in GVHMR's environment, not ours.
"""

import os
import sys

# marker -> SMPL vertex (OpenCapBench, vertices_keypoints_corr_smpl.csv); names as in
# mocap2myo/markersets/smpl_opencap.toml (OpenCap's, shoulders as l_ / r_shoulder)
MARKERS = {
  "l_ASIS": 3156, "r_ASIS": 6573, "l_PSIS": 3097, "r_PSIS": 6521,
  "l_knee": 1009, "l_mknee": 1013, "r_knee": 4496, "r_mknee": 4499,
  "l_ankle": 3327, "l_mankle": 3432, "r_ankle": 6728, "r_mankle": 6832,
  "l_calc": 3387, "l_toe": 3340, "l_5meta": 3349, "r_calc": 6796, "r_toe": 6741, "r_5meta": 6748,
  "C7": 3012, "T6": 3028, "T11": 3017, "sternum": 3171,
  "l_shoulder": 1239, "r_shoulder": 4724,
  "l_lelbow": 1666, "l_melbow": 1731, "r_lelbow": 5135, "r_melbow": 5200,
  "l_lwrist": 2235, "l_mwrist": 2106, "r_lwrist": 5696, "r_mwrist": 5567,
}


def main():
  if len(sys.argv) not in (3, 4) or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip())
    sys.exit(0 if len(sys.argv) > 1 else 2)
  import numpy as np
  import torch
  res_path, out_path = (os.path.abspath(p) for p in sys.argv[1:3])
  fixed = np.load(sys.argv[3]) if len(sys.argv) == 4 else None
  sys.path.insert(0, os.getcwd())
  from hmr4d.utils.smplx_utils import make_smplx  # GVHMR

  res = torch.load(res_path, map_location="cpu", weights_only=False)
  p = res["smpl_params_global"]
  T = len(p["transl"])
  betas = torch.as_tensor(fixed).float()[None] if fixed is not None else p["betas"].float().mean(0, keepdim=True)
  smplx = make_smplx("supermotion")
  to_smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt")
  idx = torch.tensor(list(MARKERS.values()))
  markers, lowest = [], []
  with torch.no_grad():
    for a in range(0, T, 200):
      n = min(200, T - a)
      v = smplx(body_pose=p["body_pose"][a:a + n].float(), betas=betas.expand(n, -1),
                global_orient=p["global_orient"][a:a + n].float(), transl=p["transl"][a:a + n].float()).vertices
      vs = torch.stack([torch.matmul(to_smpl, x) for x in v])  # [n, 6890, 3]
      markers.append(vs[:, idx])
      lowest.append(vs[torch.arange(n), vs[..., 1].argmin(1)])
    z = torch.zeros
    v0 = smplx(body_pose=z(1, 63), betas=betas, global_orient=z(1, 3), transl=z(1, 3)).vertices[0]
    v0 = torch.matmul(to_smpl, v0)
  np.savez(out_path, names=np.array(list(MARKERS)), markers=torch.cat(markers).numpy(),
           lowest=torch.cat(lowest).numpy(), static=v0[idx].numpy(), static_lowest=v0[v0[:, 1].argmin()].numpy(),
           stature=float(v0[:, 1].max() - v0[:, 1].min()), betas=betas[0].numpy())
  print(f"{out_path}: {T} frames, {len(MARKERS)} markers, stature {float(v0[:, 1].max() - v0[:, 1].min()):.3f} m")


if __name__ == "__main__":
  main()
