"""GPU version of the tracking IK (ik.ik_trial with contact=True and the robust loss).

The same problem as the CPU solver, frame by frame: the marker cloud fitted with the
subject's calibrated offsets, Cauchy loss on every marker but the pelvis, toe prior,
floor, stance-foot anchors, smoothing toward the previous frame, the joint speed
limit (bounds around the previous frame) and the joint limits. The CPU solver walks
through a trial one frame after the other, each frame warm-started from and
smoothed toward the one before (~120 ms a frame on one core). Here:

  - Kinematics and the marker / sole-point Jacobians of MyoFullBody on the GPU (only
    the 50 bodies that carry markers or sole points; two warp kernels, float64;
    MuJoCo's joint equalities as polynomials of their leaders).
  - Per frame a projected Levenberg-Marquardt on [root position, root rotation (a
    world-frame rotation vector, re-centred every iteration), the free joints]:
    float64 cost and gradient, float32 Gauss-Newton matrix (the GPU's float64 is
    slow; it only shapes the step, the stationary point is the float64 one), bounds
    by an active set with the crossing joints put on their bound.
  - The CPU's frame-to-frame recursion, exactly, in parallel over time. The trial is
    cut into chunks of 64 frames, each a slot stepping through its frames on its own
    (CUDA graphs; a slot moves to its next frame as soon as its frame converged).
    First pass: every chunk starts from a coarse sequential sweep over the chunk
    starts (so the slowly carried states, e.g. the plane of elevation of a hanging
    arm, come out on the CPU's branch) and runs a 64-frame burn-in before its own
    frames. Later passes re-run the chunks that start elsewhere than the chunk before
    them ended (joints, root, stance anchors; anchor targets taken from their
    touch-down frame directly); a slot that finishes its chunk elsewhere than where
    the next one last started takes the next one over, and a slot stops as soon as
    it is back on the stored solution. When every chunk starts where the one before
    it ended, the result is the sequential recursion itself, to the solver
    tolerance (on the case-study trials: joint angles within ~0.1 deg RMS of the CPU).
  - The first solved frame is the CPU's multi-start (same seed, same draws).

  Q, E = ik_trial_gpu(mm, P, rate=rate, contact=True, outlier=..., mtp_prior=...,
                      anchor_centroid=..., pelvis_gap=..., robust=..., root_init=...)
returns what ik.ik_trial(contact=True, robust=...) returns. Needs torch (CUDA) and
warp (mujoco_warp's warp).
"""

from __future__ import annotations

import time
import warnings

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation as sRot

from mocap2myo import ik as mk
from mocap2myo import markerset as _markerset

wp.config.quiet = True
_F64 = wp.float64
_V3 = wp.vec3d
_M33 = wp.mat33d
_Q = wp.quatd


# --- the marker model's settings (ik.MarkerModel; older versions lack some) -------------

def _mm_info(mm):
  """What the solver needs from a MarkerModel: its marker set (labels, head / pelvis
  markers), active markers, reference pose, toe prior target, the joints smoothed
  harder and the gated ones (MyoFullBody: ankle / subtalar, elv_angle by shoulder_elv)."""
  ms = getattr(mm, "ms", None) or _markerset
  nf = len(mm.free)
  heavy = getattr(mm, "smooth_heavy", None)
  if heavy is None:
    heavy = np.array([n.startswith(("ankle_angle", "subtalar")) for n in mm.free])
  gated = getattr(mm, "smooth_gated", None)
  if gated is None:
    gated = [(mm.free.index(f"elv_angle_{sd}"), mm.free.index(f"shoulder_elv_{sd}"))
             for sd in ("r", "l")]
  return dict(labels=list(ms.LABELS), head=np.asarray(ms.HEAD, bool),
              pelvis=list(ms.PELVIS), pelvis_mask=np.asarray(ms.PELVIS_MASK, bool),
              active=np.asarray(getattr(mm, "active", np.ones(len(ms.LABELS), bool)), bool),
              qref=np.asarray(getattr(mm, "qref", mm.m.qpos0), float),
              mtp=np.asarray(mm.mtp, int),
              mtp_q0=np.asarray(getattr(mm, "mtp_q0", np.zeros(len(mm.mtp))), float),
              heavy=np.asarray(heavy, bool), gated=list(gated), nf=nf, ms=getattr(mm, "ms", None))


# --- kinematics: warp kernels ---------------------------------------------------

@wp.func
def _axis_rot(a: _V3, th: _F64) -> _M33:
  s = wp.sin(th)
  c = wp.cos(th)
  z = _F64(0.0)
  k = _M33(z, -a[2], a[1], a[2], z, -a[0], -a[1], a[0], z)
  return wp.identity(n=3, dtype=_F64) + s * k + (_F64(1.0) - c) * (k @ k)


@wp.kernel
def _fk_kernel(root_pos: wp.array(dtype=_V3), root_R: wp.array(dtype=_M33),
               q: wp.array2d(dtype=_F64), br_start: wp.array(dtype=int),
               br_body: wp.array(dtype=int), b_pos: wp.array(dtype=_V3),
               b_quat: wp.array(dtype=_Q), b_jadr: wp.array(dtype=int),
               b_jnum: wp.array(dtype=int), j_type: wp.array(dtype=int),
               j_axis: wp.array(dtype=_V3), j_pos: wp.array(dtype=_V3),
               j_off: wp.array(dtype=int), j_mode: wp.array(dtype=int),
               j_k: wp.array(dtype=int), j_lq0: wp.array(dtype=_F64),
               j_coef: wp.array2d(dtype=_F64),
               xpos: wp.array2d(dtype=_V3), xmat: wp.array2d(dtype=_M33),
               jaxis: wp.array2d(dtype=_V3), janc: wp.array2d(dtype=_V3),
               jdq: wp.array2d(dtype=_F64)):
  """Body poses (mj_kinematics on the pruned tree, with quaternions as MuJoCo) along
  one root-to-leaf branch per thread (legs, arms, head side by side; bodies the
  branches share are computed by each of them, identically). Hinge / slide joints
  use qpos - qpos0. MyoFullBody's joints all sit at their body's origin; j_off marks
  the ones that do not (the optional neck), turned about their anchor as MuJoCo
  does."""
  b, br = wp.tid()
  pos = root_pos[b]
  qt = wp.quat_from_matrix(root_R[b])
  xpos[b, 0] = pos
  xmat[b, 0] = root_R[b]
  for n in range(br_start[br] + 1, br_start[br + 1]):
    i = br_body[n]
    pos = pos + wp.quat_rotate(qt, b_pos[i])
    qt = qt * b_quat[i]
    for jj in range(b_jnum[i]):
      j = b_jadr[i] + jj
      mode = j_mode[j]
      ang = j_coef[j, 0]  # value - qpos0 (mode 2: constant)
      dq = _F64(0.0)
      if mode == 0:  # free joint of the IK
        ang = q[b, j_k[j]] - j_lq0[j]
        dq = _F64(1.0)
      if mode == 1:  # joint equality follower of free joint j_k (polynomial)
        x = q[b, j_k[j]] - j_lq0[j]
        ang = j_coef[j, 0] + x * (j_coef[j, 1] + x * (j_coef[j, 2] + x * (
          j_coef[j, 3] + x * j_coef[j, 4])))
        dq = j_coef[j, 1] + x * (_F64(2.0) * j_coef[j, 2] + x * (
          _F64(3.0) * j_coef[j, 3] + x * _F64(4.0) * j_coef[j, 4]))
      ax = wp.quat_rotate(qt, j_axis[j])
      anc = pos
      if j_off[j] != 0:
        anc = pos + wp.quat_rotate(qt, j_pos[j])
      jaxis[b, j] = ax
      janc[b, j] = anc
      jdq[b, j] = dq
      if j_type[j] == 2:  # slide
        pos = pos + ax * ang
      else:  # hinge
        h = ang * _F64(0.5)
        qt = qt * _Q(j_axis[j] * wp.sin(h), wp.cos(h))
        if j_off[j] != 0:
          pos = anc - wp.quat_rotate(qt, j_pos[j])
    qt = wp.normalize(qt)
    xpos[b, i] = pos
    xmat[b, i] = wp.quat_to_matrix(qt)


@wp.kernel
def _pts_kernel(root_pos: wp.array(dtype=_V3), group: wp.array(dtype=int),
                pt_body: wp.array(dtype=int), pt_off: wp.array2d(dtype=_V3),
                anc: wp.array2d(dtype=int), pj_start: wp.array(dtype=int),
                pj_joint: wp.array(dtype=int), j_type: wp.array(dtype=int),
                xpos: wp.array2d(dtype=_V3), xmat: wp.array2d(dtype=_M33),
                jaxis: wp.array2d(dtype=_V3), janc: wp.array2d(dtype=_V3),
                jdq: wp.array2d(dtype=_F64), pts: wp.array2d(dtype=_V3),
                jac: wp.array3d(dtype=_V3)):
  """Point k's world position and column c of its Jacobian wrt [root pos 3, root
  world rotation 3, free joints]: the joints driven by free joint c - 6 (itself and
  its equality followers, pj_*) that move point k (anc), one thread per column."""
  b, k, c = wp.tid()
  bi = pt_body[k]
  pnt = xpos[b, bi] + xmat[b, bi] @ pt_off[group[b], k]
  z = _F64(0.0)
  o = _F64(1.0)
  v = _V3(z, z, z)
  r = pnt - root_pos[b]
  if c == 0:
    v = _V3(o, z, z)
    pts[b, k] = pnt
  elif c == 1:
    v = _V3(z, o, z)
  elif c == 2:
    v = _V3(z, z, o)
  elif c == 3:
    v = _V3(z, -r[2], r[1])
  elif c == 4:
    v = _V3(r[2], z, -r[0])
  elif c == 5:
    v = _V3(-r[1], r[0], z)
  else:
    for n in range(pj_start[c - 6], pj_start[c - 5]):
      j = pj_joint[n]
      if anc[k, j] != 0:
        u = jaxis[b, j]
        if j_type[j] != 2:
          u = wp.cross(jaxis[b, j], pnt - janc[b, j])
        v = v + u * jdq[b, j]
  jac[b, k, c] = v


class GpuKinematics:
  """Marker sites and sole points of a MarkerModel, batched on the GPU."""

  def __init__(self, mm: mk.MarkerModel, device="cuda:0"):
    wp.init()
    self.device = device
    self.wdev = wp.get_device(device)
    m = mm.m
    self.mm = mm
    qref = _mm_info(mm)["qref"]
    self.nfree = len(mm.free)
    self.nx = 6 + self.nfree
    pt_body_full = np.r_[mm.bid, mm.fp_body].astype(int)
    need = set()
    for b in pt_body_full:
      while b > 0:
        need.add(int(b))
        b = int(m.body_parentid[b])
    order = sorted(need)  # MuJoCo bodies are in topological order
    root_body = int(m.jnt_bodyid[0])
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and order[0] == root_body
    idx = {b: i for i, b in enumerate(order)}
    # joint equalities: follower joint -> (leader joint, coefficients)
    lead = {}
    for e in range(m.neq):
      if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
        continue
      lead[int(m.eq_obj1id[e])] = (int(m.eq_obj2id[e]), m.eq_data[e][:5].copy())
    free_jid = [m.joint(n).id for n in mm.free]
    kfree = {j: k for k, j in enumerate(free_jid)}
    bp, bpos, bR, bja, bjn = [], [], [], [], []
    jt, jax, jpos, jmode, jk, jlq0, jcoef = [], [], [], [], [], [], []
    for b in order:
      bp.append(idx[int(m.body_parentid[b])] if b != root_body else -1)
      bpos.append(m.body_pos[b])
      R = np.zeros(9)
      mujoco.mju_quat2Mat(R, m.body_quat[b])
      bR.append(R.reshape(3, 3))
      bja.append(len(jt))
      js = [j for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b])
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE] if m.body_jntnum[b] > 0 else []
      bjn.append(len(js))
      for j in js:
        assert m.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
        jt.append(int(m.jnt_type[j]))
        jax.append(m.jnt_axis[j])
        jpos.append(m.jnt_pos[j])
        a = m.jnt_qposadr[j]
        c = np.zeros(5)
        if j in kfree:
          jmode.append(0)
          jk.append(kfree[j])
          jlq0.append(m.qpos0[a])
        elif j in lead and lead[j][0] >= 0:
          l, cc = lead[j]
          assert l in kfree, f"equality leader {m.joint(l).name} is not a free IK joint"
          jmode.append(1)
          jk.append(kfree[l])
          jlq0.append(m.qpos0[m.jnt_qposadr[l]])
          c = cc
        else:  # held at the reference pose (qpos0 + c0 for an equality without a leader)
          jmode.append(2)
          jk.append(0)
          jlq0.append(0.0)
          c[0] = lead[j][1][0] if j in lead else qref[a] - m.qpos0[a]
        jcoef.append(c)
    self.nb, self.nj = len(order), len(jt)
    # root-to-leaf paths (indices into order) for the branch-parallel kinematics
    par = np.array(bp)
    leaves = [i for i in range(self.nb) if i not in set(par[par >= 0].tolist())]
    paths = []
    for lf in leaves:
      pth = [lf]
      while par[pth[-1]] >= 0:
        pth.append(par[pth[-1]])
      paths.append(pth[::-1])
    self.nbr = len(paths)
    br_start = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int32)
    self.npt = len(pt_body_full)
    self.nmk = len(mm.bid)
    dv = self.wdev
    self._br_start = wp.array(br_start, dtype=int, device=dv)
    self._br_body = wp.array(np.concatenate(paths).astype(np.int32), dtype=int, device=dv)
    self._b_pos = wp.array(np.array(bpos), dtype=_V3, device=dv)
    self._b_quat = wp.array(np.array([m.body_quat[b][[1, 2, 3, 0]] for b in order]), dtype=_Q,
                            device=dv)
    self._b_jadr = wp.array(np.array(bja, np.int32), dtype=int, device=dv)
    self._b_jnum = wp.array(np.array(bjn, np.int32), dtype=int, device=dv)
    self._j_type = wp.array(np.array(jt, np.int32), dtype=int, device=dv)
    self._j_axis = wp.array(np.array(jax), dtype=_V3, device=dv)
    self._j_pos = wp.array(np.array(jpos), dtype=_V3, device=dv)
    self._j_off = wp.array(np.array([int(np.any(p)) for p in jpos], np.int32), dtype=int,
                           device=dv)
    self._j_mode = wp.array(np.array(jmode, np.int32), dtype=int, device=dv)
    self._j_k = wp.array(np.array(jk, np.int32), dtype=int, device=dv)
    self._j_lq0 = wp.array(np.array(jlq0), dtype=_F64, device=dv)
    self._j_coef = wp.array(np.array(jcoef), dtype=_F64, device=dv)
    # per free joint, the joints it drives (itself, its equality followers); per point,
    # the joints that move it (on its body or an ancestor)
    jmode_a, jk_a = np.array(jmode), np.array(jk)
    pj = [np.flatnonzero((jmode_a != 2) & (jk_a == k)) for k in range(self.nfree)]
    self._pj_start = wp.array(np.r_[0, np.cumsum([len(x) for x in pj])].astype(np.int32),
                              dtype=int, device=dv)
    self._pj_joint = wp.array(np.concatenate(pj).astype(np.int32), dtype=int, device=dv)
    jbody = np.repeat(np.arange(len(order)), bjn)  # pruned body of each joint
    ancm = np.zeros((len(pt_body_full), len(jt)), np.int32)
    for kp, bb in enumerate(pt_body_full):
      i = idx[int(bb)]
      while i >= 0:
        ancm[kp, jbody == i] = 1
        i = bp[i]
    self._anc = wp.array(ancm, dtype=int, device=dv)
    self._pt_body = wp.array(np.array([idx[int(b)] for b in pt_body_full], np.int32),
                             dtype=int, device=dv)
    self._fp_local = np.asarray(mm.fp_local, float)
    self._scratch = {}
    self._streams = {}
    self.set_offsets([mm.m.site_pos[mm.sid]])

  def set_offsets(self, offsets):
    """Marker offsets per group: a list of [n_markers, 3] (a batch element's group
    picks its subject's offsets)."""
    off = np.stack([np.r_[np.asarray(o, float), self._fp_local] for o in offsets])
    self._pt_off = wp.array(off, dtype=_V3, device=self.wdev)

  def _buffers(self, B):
    # one set per batch size, kept for the object's life (CUDA graphs point at them)
    if B not in self._scratch:
      dv, t = self.wdev, self.device
      xpos = wp.zeros((B, self.nb), dtype=_V3, device=dv)
      xmat = wp.zeros((B, self.nb), dtype=_M33, device=dv)
      jaxis = wp.zeros((B, self.nj), dtype=_V3, device=dv)
      janc = wp.zeros((B, self.nj), dtype=_V3, device=dv)
      jdq = wp.zeros((B, self.nj), dtype=_F64, device=dv)
      pts = torch.zeros((B, self.npt, 3), dtype=torch.float64, device=t)
      jac = torch.zeros((B, self.npt, self.nx, 3), dtype=torch.float64, device=t)
      self._scratch[B] = (xpos, xmat, jaxis, janc, jdq, pts, jac,
                          wp.from_torch(pts, dtype=_V3), wp.from_torch(jac, dtype=_V3))
    return self._scratch[B]

  def fk(self, pos, R, q, group):
    """pos [B, 3], R [B, 3, 3], q [B, nfree], group [B] int32 (torch, float64 on the
    device) -> points [B, npt, 3] (markers in LABELS order, then sole points) and
    their Jacobians [B, npt, 3, nx] (new tensors)."""
    B = pos.shape[0]
    xpos, xmat, jaxis, janc, jdq, pts, jac, wpts, wjac = self._buffers(B)
    # launch on torch's current stream (no stream switch: that would make warp wait on
    # its own stream, which breaks CUDA-graph capture)
    ts = torch.cuda.current_stream(self.device)
    stream = self._streams.get(ts.cuda_stream)
    if stream is None:
      stream = self._streams[ts.cuda_stream] = wp.stream_from_torch(ts)
    wpos = wp.from_torch(pos.contiguous(), dtype=_V3)
    wR = wp.from_torch(R.contiguous(), dtype=_M33)
    wq = wp.from_torch(q.contiguous())
    wg = wp.from_torch(group.contiguous())
    wp.launch(_fk_kernel, dim=(B, self.nbr), device=self.wdev, stream=stream,
              inputs=[wpos, wR, wq, self._br_start, self._br_body, self._b_pos, self._b_quat,
                      self._b_jadr, self._b_jnum, self._j_type, self._j_axis, self._j_pos,
                      self._j_off, self._j_mode, self._j_k, self._j_lq0, self._j_coef,
                      xpos, xmat, jaxis, janc, jdq])
    wp.launch(_pts_kernel, dim=(B, self.npt, self.nx), device=self.wdev, stream=stream,
              inputs=[wpos, wg, self._pt_body, self._pt_off, self._anc, self._pj_start,
                      self._pj_joint, self._j_type, xpos, xmat, jaxis, janc, jdq, wpts, wjac])
    return pts.clone(), jac.transpose(2, 3).clone()


# --- small torch helpers ----------------------------------------------------------

def _rotvec_to_mat(v):
  """[B, 3] rotation vectors -> [B, 3, 3] (Rodrigues)."""
  th = torch.linalg.norm(v, dim=-1, keepdim=True)
  a = v / th.clamp_min(1e-30)
  z = torch.zeros_like(th[:, 0])
  K = torch.stack([z, -a[:, 2], a[:, 1], a[:, 2], z, -a[:, 0], -a[:, 1], a[:, 0], z],
                  -1).reshape(-1, 3, 3)
  s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
  eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
  return eye + s * K + (1 - c) * (K @ K)


# --- batched frame solver ----------------------------------------------------------

def _masked_solve(A, b, free):
  """A x = b on the free variables (free [B, n] 0/1; the others get 0)."""
  M = A * (free[:, :, None] * free[:, None, :]) + torch.diag_embed(1 - free)
  L, info = torch.linalg.cholesky_ex(M)
  y = torch.linalg.solve_triangular(L, (b * free)[..., None], upper=False)
  x = torch.linalg.solve_triangular(L.transpose(1, 2), y, upper=True)[..., 0]
  return x * free, info


class _Batch:
  """Static buffers for B frame problems and their solver state (the same tensors
  are reused, so the work on them can be captured as CUDA graphs).
  smooth: smoothing toward x_prev and speed bounds (every frame after the first);
  contact: floor and stance anchors."""

  def __init__(self, ik, B, smooth, contact):
    t = dict(dtype=torch.float64, device=ik.device)
    nf, nx, nm, nfp = ik.nf, ik.nx, ik.nmk, ik.nfp
    self.B, self.smooth, self.contact = B, smooth, contact
    self.xtol = ik.xtol  # step tolerance of this batch's frames (rad, m)
    z = lambda *s, **k: torch.zeros(*s, **(k or t))  # noqa: E731
    self.P, self.ok = z(B, nm, 3), z(B, nm, dtype=torch.bool, device=ik.device)
    self.group = z(B, dtype=torch.int32, device=ik.device)
    self.lo, self.hi, self.x_prev, self.sm = z(B, nf), z(B, nf), z(B, nf), z(B, nf)
    self.floor = z(B)
    self.anc_on = z(B, nfp, dtype=torch.bool, device=ik.device)
    self.anc_tgt = z(B, nfp, 3)
    self.pos, self.q = z(B, 3), z(B, nf)
    self.R = torch.eye(3, **t).repeat(B, 1, 1)
    self.F, self.g = z(B), z(B, nx)
    self.H = z(B, nx, nx, dtype=torch.float32, device=ik.device)
    self.pts = z(B, ik.kin.npt, 3)
    self.lam = z(B)
    self.done = z(B, dtype=torch.bool, device=ik.device)
    self.iters = z(B, dtype=torch.int32, device=ik.device)


class GpuIK:
  """MarkerModel.solve(robust=..., contact=..., x_prev=..., smooth=...) for a batch
  of frames: projected Levenberg-Marquardt, see iterate()."""

  def __init__(self, mm: mk.MarkerModel, device="cuda:0", robust=0.025, mtp_prior=0.1,
               w_floor=10.0, w_anchor=3.0, centroid=True, xtol=1e-5, lam0=1e-6):
    self.mm = mm
    self.kin = GpuKinematics(mm, device)
    self.device = device
    self.nx, self.nf, self.nmk = self.kin.nx, self.kin.nfree, self.kin.nmk
    self.nfp = len(mm.fp_kind)
    t = dict(dtype=torch.float64, device=device)
    self.sw = torch.tensor(np.sqrt(mm.w), **t)
    info = _mm_info(mm)
    self.pelvis = torch.tensor(info["pelvis_mask"], device=device)
    self.jlo = torch.tensor(mm.lo, **t)
    self.jhi = torch.tensor(mm.hi, **t)
    self.mtp_mask = torch.zeros(self.nf, **t)
    self.mtp_mask[torch.tensor(info["mtp"], dtype=torch.long, device=device)] = 1
    self.mtp_q0 = torch.zeros(self.nf, **t)
    self.mtp_q0[torch.tensor(info["mtp"], dtype=torch.long, device=device)] = \
      torch.tensor(info["mtp_q0"], **t)
    self.fp_side = torch.tensor([[k[0] == sd for k in mm.fp_kind] for sd in ("l", "r")], **t)
    self.robust, self.mtp_prior = robust, mtp_prior
    self.w_floor, self.w_anchor, self.centroid = w_floor, w_anchor, centroid
    self.xtol, self.lam0 = xtol, lam0

  def set_offsets(self, offsets):
    self.kin.set_offsets(offsets)

  def _eval(self, bt, pos, R, q):
    """Cost [B] (scipy's: half the sum of squares), Gauss-Newton Hessian [B, nx, nx],
    gradient [B, nx], points [B, npt, 3] at (pos, R, q) for the problems in bt."""
    pts, Jp = self.kin.fk(pos, R, q, bt.group)
    nm, B = self.nmk, bt.B
    okf = bt.ok[..., None].to(pts.dtype)
    e = (pts[:, :nm] - bt.P) * self.sw[:, None] * okf
    Je = Jp[:, :nm] * (self.sw[:, None, None] * okf[..., None])
    # Robust markers: Cauchy cost c(n) = f^2 / 2 log(1 + (n / f)^2) per marker (the
    # CPU's residual f sqrt(log(1 + (n / f)^2)) along e, squared; the pelvis markers
    # keep the squared loss). Gradient a Je^T e, a = 1 / (1 + u^2), u = n / f.
    # Model Hessian: a across the residual, along it the Gauss-Newton value of the
    # transformed residual (d rn / dn)^2 = u^2 / (log(1 + u^2) (1 + u^2)^2) (>= 0).
    f = self.robust
    n = torch.linalg.norm(e, dim=-1, keepdim=True)
    u2 = (n / f) ** 2
    Lg = torch.log1p(u2)
    pel = self.pelvis[None, :, None]
    one = torch.ones_like(u2)
    a = torch.where(pel, one, 1 / (1 + u2))
    bg = torch.where(pel | (u2 < 1e-12), one, u2 / (Lg.clamp_min(1e-300) * (1 + u2) ** 2))
    cost = torch.where(pel[..., 0], n[..., 0] ** 2, f * f * Lg[..., 0]).sum(1)
    eh = e / n.clamp_min(1e-15)
    v = torch.einsum("bmc,bmck->bmk", eh, Je)  # along the residual
    g = torch.einsum("bmc,bmck->bk", a * e, Je)
    sa = torch.sqrt(a)
    Jx = sa[..., None] * Je - (sa * eh)[..., None] * v[:, :, None, :]  # across
    jrows = [Jx.reshape(B, -1, self.nx), torch.sqrt(bg) * v]
    fp, Jfp = pts[:, nm:], Jp[:, nm:]
    if bt.contact:
      dz = fp[..., 2] - bt.floor[:, None]
      act = (dz < 0).to(dz.dtype)
      rf = self.w_floor * dz * act
      Jf = self.w_floor * Jfp[:, :, 2] * act[..., None]
      cost = cost + (rf * rf).sum(1)
      g = g + torch.einsum("bp,bpk->bk", rf, Jf)
      jrows.append(Jf)
      on = bt.anc_on.to(fp.dtype)
      if self.centroid:  # one point per foot: the centroid of its anchored points
        for sd in range(2):
          w = on * self.fp_side[sd]
          wn = w / w.sum(1).clamp_min(1)[:, None]
          ra = self.w_anchor * (wn[..., None] * (fp - bt.anc_tgt)).sum(1)
          Ja = self.w_anchor * (wn[..., None, None] * Jfp).sum(1)
          cost = cost + (ra * ra).sum(1)
          g = g + torch.einsum("bc,bck->bk", ra, Ja)
          jrows.append(Ja)
      else:
        ra = self.w_anchor * on[..., None] * (fp - bt.anc_tgt)
        Ja = (self.w_anchor * on[..., None, None] * Jfp).reshape(B, -1, self.nx)
        cost = cost + (ra * ra).sum((1, 2))
        g = g + torch.einsum("bc,bck->bk", ra.reshape(B, -1), Ja)
        jrows.append(Ja)
    # the model Hessian in float32 (this GPU's float64 is slow); cost and gradient stay
    # float64, so the stationary point does not change, only the step is inexact
    J = torch.cat(jrows, 1).float()
    H = J.transpose(1, 2) @ J
    # diagonal terms: smoothing toward the previous frame, toe prior
    hd = torch.zeros_like(g)
    if bt.smooth:
      rs = bt.sm * (q - bt.x_prev)
      cost = cost + (rs * rs).sum(1)
      g = g + torch.nn.functional.pad(bt.sm * rs, (6, 0))
      hd = hd + torch.nn.functional.pad(bt.sm ** 2, (6, 0))
    if self.mtp_prior:
      rm = self.mtp_prior * (q - self.mtp_q0) * self.mtp_mask
      cost = cost + (rm * rm).sum(1)
      g = g + torch.nn.functional.pad(self.mtp_prior * rm, (6, 0))
      hd = hd + torch.nn.functional.pad(self.mtp_prior ** 2 * self.mtp_mask.expand_as(q), (6, 0))
    H = H + torch.diag_embed(hd.float())
    return 0.5 * cost, H, g, pts

  def start(self, bt):
    """Evaluate the start point in bt (pos, R, q; q clipped into the bounds as the
    CPU's least_squares does) and reset the solver state; frames already marked
    done stay as they are."""
    bt.q.copy_(torch.minimum(torch.maximum(bt.q, bt.lo + 1e-9), bt.hi - 1e-9))
    F, H, g, pts = self._eval(bt, bt.pos, bt.R, bt.q)
    bt.F.copy_(F)
    bt.H.copy_(H)
    bt.g.copy_(g)
    bt.pts.copy_(pts)
    bt.lam.fill_(self.lam0)
    bt.iters.zero_()

  def iterate(self, bt, fresh=None, active=None):
    """One projected Levenberg-Marquardt iteration (in place) on the frames not done.
    Joints at a bound whose gradient points out are held; joints whose step would
    cross a bound are put on it and the others re-solved with them held there
    (projected Newton), instead of clipping the step (a clipped Gauss-Newton step
    often goes uphill, and the damping then has to climb before it gets anywhere).
    A step is taken when it does not raise the cost; a frame is done when its step
    is below xtol (rad / m), its cost stops changing, no descent step is left, or
    after _MAX_IT iterations (the CPU's max_nfev).
    fresh (optional, [B] bool): frames whose start point (pos, R, q) was just set:
    this iteration only evaluates it. active (optional, [B] bool): frames to work on
    (default: not done). Returns the frames that converged in this iteration."""
    q, g, H, lo, hi = bt.q, bt.g, bt.H, bt.lo, bt.hi
    work = ~bt.done if active is None else active
    if fresh is None:
      fresh = torch.zeros_like(work)
    stepping = work & ~fresh
    gq = g[:, 6:]
    held = ((q <= lo + 1e-12) & (gq > 0)) | ((q >= hi - 1e-12) & (gq < 0))
    pad = torch.zeros_like(held[:, :6])
    free0 = (~torch.cat([pad, held], 1)).to(H.dtype)
    dH = torch.diagonal(H, dim1=1, dim2=2)
    A = H + torch.diag_embed(bt.lam[:, None].float() * dH + 1e-7 * dH.amax(1, keepdim=True))
    gf = g.float()
    dx0, info0 = _masked_solve(A, -gf, free0)
    dx0 = dx0.double()
    qn = q + dx0[:, 6:]
    cross = ~held & ((qn < lo) | (qn > hi))
    dc = torch.where(cross, torch.where(qn < lo, lo, hi) - q, torch.zeros_like(q))
    dcf = torch.nn.functional.pad(dc, (6, 0))
    free1 = free0 * (1 - torch.cat([pad, cross], 1).to(H.dtype))
    dx1, info1 = _masked_solve(A, -gf - (A @ dcf.float()[..., None])[..., 0], free1)
    dx1 = dx1.double()
    anyc = cross.any(1)
    dx = torch.where(anyc[:, None], dx1 + dcf, dx0)
    info = torch.where(anyc, info1, info0)
    ok = stepping & (info == 0)
    dx = torch.where(ok[:, None], dx, torch.zeros_like(dx))
    pos_t = bt.pos + dx[:, :3]
    R_t = _rotvec_to_mat(dx[:, 3:6]) @ bt.R
    q_t = torch.minimum(torch.maximum(q + dx[:, 6:], lo), hi)
    F_t, H_t, g_t, pts_t = self._eval(bt, pos_t, R_t, q_t)
    acc = (ok & (F_t <= bt.F)) | (work & fresh)
    step = dx.abs().amax(1)
    iters = bt.iters + stepping.to(torch.int32)
    conv = stepping & ((ok & (F_t <= bt.F) & (step < bt.xtol))
                       | (ok & ((bt.F - F_t).abs() <= 1e-13 * bt.F))
                       | (bt.lam > 1e10) | (ok & (step == 0)) | (iters >= _MAX_IT))
    a1, a2 = acc[:, None], acc[:, None, None]
    bt.pos.copy_(torch.where(a1, pos_t, bt.pos))
    bt.R.copy_(torch.where(a2, R_t, bt.R))
    bt.q.copy_(torch.where(a1, q_t, q))
    bt.H.copy_(torch.where(a2, H_t, H))
    bt.g.copy_(torch.where(a1, g_t, g))
    bt.pts.copy_(torch.where(a2, pts_t, bt.pts))
    bt.F.copy_(torch.where(acc, F_t, bt.F))
    lam = torch.where(acc, (bt.lam * 0.2).clamp_min(1e-9), bt.lam * 8)
    lam = torch.where(fresh, torch.full_like(lam, self.lam0), lam)
    bt.lam.copy_(torch.where(work, lam, bt.lam))
    bt.iters.copy_(torch.where(fresh, torch.zeros_like(iters), iters))
    bt.done.copy_(bt.done | conv)
    return conv

  def solve_eager(self, bt, max_iters=100, check_every=4):
    """start() must have been called. Iterate until every frame is done."""
    for it in range(1, max_iters + 1):
      self.iterate(bt)
      if it % check_every == 0 and bool(bt.done.all()):
        break


class _Graph:
  """A function of static tensors captured as a CUDA graph (or run eagerly)."""

  def __init__(self, fn, use_graph):
    self.fn = fn
    self.graph = None
    if use_graph:
      s = torch.cuda.Stream()
      s.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(s):
        fn()  # warm-up (allocations, kernels compiled) outside the capture
      torch.cuda.current_stream().wait_stream(s)
      self.graph = torch.cuda.CUDAGraph()
      with torch.cuda.graph(self.graph):
        fn()

  def __call__(self):
    if self.graph is None:
      self.fn()
    else:
      self.graph.replay()


# --- whole trial -----------------------------------------------------------------

def _kabsch_roots(mm, P):
  """ik.root_from_pelvis for every frame with all pelvis markers: pos [T, 3], R
  [T, 3, 3], quat [T, 4] (wxyz, as the CPU's), ok [T]."""
  m, d = mm.m, mm.d
  info = _mm_info(mm)
  PELVIS = info["pelvis"]
  d.qpos[:] = info["qref"]
  d.qpos[:3] = 0
  d.qpos[3:7] = [1, 0, 0, 0]
  mujoco.mj_kinematics(m, d)
  A = d.site_xpos[mm.sid[PELVIS]].copy()
  T = len(P)
  ok = ~np.isnan(P[:, PELVIS]).any((1, 2))
  pos = np.zeros((T, 3))
  R = np.tile(np.eye(3), (T, 1, 1))
  quat = np.tile([1.0, 0, 0, 0], (T, 1))
  if ok.any():
    B = P[ok][:, PELVIS]
    ca, cb = A.mean(0), B.mean(1)
    H = np.einsum("ni,tnj->tij", A - ca, B - cb[:, None])
    U, _, Vt = np.linalg.svd(H)
    det = np.linalg.det(np.swapaxes(Vt, 1, 2) @ np.swapaxes(U, 1, 2))
    D = np.tile(np.eye(3), (len(B), 1, 1))
    D[:, 2, 2] = np.sign(det)
    Rk = np.swapaxes(Vt, 1, 2) @ D @ np.swapaxes(U, 1, 2)
    pos[ok] = cb - np.einsum("tij,j->ti", Rk, ca)
    quat[ok] = sRot.from_matrix(Rk).as_quat()[:, [3, 0, 1, 2]]
    R[ok] = sRot.from_quat(quat[ok][:, [1, 2, 3, 0]]).as_matrix()
  return pos, R, quat, ok


def _apply_eq(m, Q):
  """kinematics.apply_joint_equalities on every row of Q [T, nq]."""
  for e in range(m.neq):
    if m.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
      continue
    j1, j2 = m.eq_obj1id[e], m.eq_obj2id[e]
    a1 = m.jnt_qposadr[j1]
    c = m.eq_data[e][:5]
    if j2 < 0:
      Q[:, a1] = m.qpos0[a1] + c[0]
    else:
      a2 = m.jnt_qposadr[j2]
      x = Q[:, a2] - m.qpos0[a2]
      Q[:, a1] = m.qpos0[a1] + c[0] + c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4


def _bucket(n, step=16):
  return int(-(-n // step) * step)


def check_supported(mm, rate):
  """The GPU IK reproduces ik_trial for MyoFullBody with the Plug-in-Gait marker set at
  100 Hz only (the per-profile parts of the marker model -- other reference poses,
  marker sets, smoothing scaled to other rates -- are not ported); SystemExit else."""
  prof = getattr(getattr(mm, "profile", None), "name", "myofullbody")
  mset = getattr(getattr(mm, "ms", None), "key", "plug_in_gait")
  if prof != "myofullbody" or mset != "plug_in_gait" or abs(float(rate) - 100.0) > 1e-9:
    raise SystemExit(f"M2M_IK_DEVICE=gpu: only the myofullbody profile with the plug_in_gait "
                     f"marker set at 100 Hz (here {prof}, {mset}, {float(rate):g} Hz); use "
                     "M2M_IK_DEVICE=cpu")


def ik_trial_gpu(mm, P, rate=None, stats=None, **kw):
  """ik.ik_trial(contact=True, robust=...) on the GPU: same arguments, same output
  (Q [T, nq], E [T, n]); see ik_trials_gpu for the GPU's own arguments."""
  st = {} if stats is not None else None
  (out,) = ik_trials_gpu([mm], [P], rate=rate, stats=st, **kw)
  if stats is not None:
    stats.update(st)
    stats.update(st["trials"][0])
  return out


def ik_trials_gpu(mms, Ps, smooth=0.05, restarts=8, seed=0, rate=None, contact=True,
                  v_joint=15.0, w_floor=10.0, w_anchor=3.0, ankle_smooth=0.3, outlier=None,
                  mtp_prior=0.0, anchor_centroid=False, pelvis_gap="carry", robust=None,
                  root_init="prev", device="cuda:0", chunk=64, burn_in=64, max_passes=20,
                  tol=1e-3, xtol=1e-5, sweep_xtol=1e-3, graphs=True,
                  verbose=False, stats=None):
  """ik.ik_trial(contact=True, robust=...) for several trials at once (one batch on
  the GPU; their frames solved side by side): mms, Ps, the trials' marker models
  (the same model; each with its subject's offsets) and markers; the other arguments
  as ik_trial's, the same for every trial. Returns [(Q, E)] per trial.
  GPU arguments: device; chunk, frames per chunk; burn_in, frames before each chunk
  in the first pass; passes stop when every chunk starts within tol (rad, m) of
  where the chunk before it ended (at most max_passes; a warning says so if they do
  not); xtol, the per-frame solver's step tolerance (rad, m), sweep_xtol the same
  for the coarse first-pass sweep; graphs, CUDA graphs for the per-frame
  work (False: plain launches, for debugging); stats (dict, optional) receives
  timing, passes and iteration counts."""
  if not contact:
    raise NotImplementedError("GPU IK: contact=True only (the pipeline's setting)")
  if not robust:
    raise NotImplementedError("GPU IK: the robust (Cauchy) loss only (M2M_ROBUST > 0)")
  if rate is None:
    raise ValueError("rate is needed with contact=True")
  for x in mms:
    check_supported(x, rate)
  t_start = time.time()
  td = dict(dtype=torch.float64, device=device)
  mm = mms[0]
  ik = GpuIK(mm, device, robust=robust, mtp_prior=mtp_prior, w_floor=w_floor,
             w_anchor=w_anchor, centroid=anchor_centroid, xtol=xtol)
  ik.set_offsets([x.m.site_pos[x.sid] for x in mms])  # kinematics group = trial
  m = mm.m
  info = _mm_info(mm)
  PELVIS, HEAD, active, qref = info["pelvis"], info["head"], info["active"], info["qref"]
  nm, nf, nfp, nt = len(info["labels"]), ik.nf, ik.nfp, len(Ps)
  results = [(np.zeros((len(P), m.nq)), np.full((len(P), nm), np.nan)) for P in Ps]

  # per trial: the frames the CPU solves (the others are carried / held), root fits,
  # stance; then everything concatenated (frame f of trial i at fo[i] + f, sequence
  # position p at so[i] + p)
  seqs, kab, stf, mst, hasp = [], [], [], [], []
  for x, P in zip(mms, Ps):
    has_pel = ~np.isnan(P[:, PELVIS]).any((1, 2))
    first = np.flatnonzero(has_pel)
    if pelvis_gap == "carry":
      seq = first
    else:
      seq = np.arange(first[0], len(P)) if len(first) else np.zeros(0, int)
    seqs.append(seq)
    hasp.append(has_pel)
    kab.append(_kabsch_roots(x, P))
    stance = mk.sole_stance(P, rate, ms=info["ms"]) if info["ms"] is not None else \
      mk.sole_stance(P, rate)
    s = np.zeros((len(P), nfp), bool)
    for i, (sd, kind) in enumerate(mm.fp_kind):
      if kind != "-":
        s[:, i] = stance[(sd, kind)]
    stf.append(s)
    mst.append(np.r_[0.0, v_joint * np.diff(seq) / rate] if len(seq) else np.zeros(0))
  live = [i for i in range(nt) if len(seqs[i])]
  if not live:
    return results
  Ts = np.array([len(P) for P in Ps])
  Ns = np.array([len(s) for s in seqs])
  fo = np.r_[0, np.cumsum(Ts)[:-1]]
  so = np.r_[0, np.cumsum(Ns)[:-1]]
  NN = int(Ns.sum())
  seq_all = np.concatenate([seqs[i] + fo[i] for i in range(nt)])
  trial_of = np.repeat(np.arange(nt), Ns)  # per sequence position
  # smoothing weights (ik_trial's smooth_at): per frame at ik.SMOOTH_RATE, scaled by
  # sqrt(rate / SMOOTH_RATE) as ik_trial does (1 at 100 Hz, the only rate check_supported
  # lets through)
  fs = np.sqrt(rate / mk.SMOOTH_RATE) if hasattr(mk, "SMOOTH_RATE") else 1.0
  base = np.where(info["heavy"], ankle_smooth, 0.05) * fs
  elv = [e for e, _ in info["gated"]]
  selv = [g for _, g in info["gated"]]
  elv_mask = torch.zeros(nf, **td)
  selv_of = torch.arange(nf, device=device)
  if elv:
    elv_mask[elv] = 1
    selv_of[elv] = torch.tensor(selv, device=device)  # gated joint -> its gate joint

  # device copies
  Pall = np.concatenate(Ps)
  Pg = torch.tensor(np.nan_to_num(Pall), **td)
  okg = torch.tensor(~np.isnan(Pall[..., 0]) & active, device=device)
  seqg = torch.tensor(seq_all, device=device)
  st_g = torch.tensor(np.concatenate(stf), device=device)
  kpos_g = torch.tensor(np.concatenate([k[0] for k in kab]), **td)
  kR_g = torch.tensor(np.concatenate([k[1] for k in kab]), **td)
  hp = np.concatenate(hasp)
  usek_g = torch.tensor(hp if root_init == "pelvis" else np.zeros(len(hp), bool), device=device)
  mstep_g = torch.tensor(np.concatenate(mst), **td)
  base_g = torch.tensor(base, **td)
  grp_g = torch.tensor(trial_of, dtype=torch.int32, device=device)  # per sequence position

  def multistart(t_idx, grp):
    """Independent solve of frames t_idx (no previous frame, no contact), the CPU's
    first-frame multi-start: start 0 neutral, the others uniform in the joint limits
    (the CPU's draws: each trial's generator seeded alike); the best by mean marker
    error (outliers out)."""
    nb = len(t_idx)
    bt = _Batch(ik, nb * restarts, smooth=False, contact=False)
    tt = torch.tensor(np.tile(t_idx, restarts), device=device)
    bt.P.copy_(Pg[tt])
    bt.ok.copy_(okg[tt])
    bt.group.copy_(torch.tensor(np.tile(grp, restarts), dtype=torch.int32, device=device))
    bt.lo.copy_(ik.jlo.expand(len(tt), nf))
    bt.hi.copy_(ik.jhi.expand(len(tt), nf))
    xs = np.repeat(qref[mm.adr][None], nb * restarts, 0)
    rng = np.random.default_rng(seed)
    for k in range(1, restarts):
      xs[k * nb:(k + 1) * nb] = mm.lo + rng.random(nf) * (mm.hi - mm.lo)
    if mtp_prior:
      xs[:, info["mtp"]] = info["mtp_q0"]
    bt.q.copy_(torch.tensor(xs, **td))
    bt.pos.copy_(kpos_g[tt])
    bt.R.copy_(kR_g[tt])
    ik.start(bt)
    bt.done.zero_()
    ik.solve_eager(bt, max_iters=100)
    err = torch.linalg.norm(bt.pts[:, :nm] - Pg[tt], dim=-1)
    err = torch.where(okg[tt], err, torch.full_like(err, float("nan")))
    if outlier is not None:
      bad = (torch.nan_to_num(err) > outlier) & ~torch.tensor(HEAD, device=device)
      nbad = bad.sum(1, keepdim=True)
      err = torch.where(bad & (nbad <= 3), torch.full_like(err, float("nan")), err)
    score = torch.nanmean(err, 1).reshape(restarts, nb)
    k = torch.argmin(torch.nan_to_num(score, nan=np.inf), 0)  # first minimum, as min()
    idx = k * nb + torch.arange(nb, device=device)
    return bt.pos[idx], bt.R[idx], bt.q[idx], bt.pts[idx]

  def touch_down(on, tgt, fp, t, floor):
    """ik_trial's touch_down: points newly in stance get an anchor where they are
    (not below the floor; floor [n] per row)."""
    new = st_g[t] & ~on
    tg = torch.cat([fp[..., :2], torch.maximum(fp[..., 2:], floor[:, None, None])], -1)
    return on | new, torch.where(new[..., None], tg, tgt)

  # first solved frame of every trial; its floor (lowest sole point)
  t0s = np.array([seq_all[so[i]] for i in live])
  pos0, R0, q0, pts0 = multistart(t0s, np.array(live))
  floors = torch.zeros(nt, **td)
  floors[torch.tensor(live, device=device)] = pts0[:, nm:, 2].amin(1)
  on0, tg0 = touch_down(torch.zeros((len(live), nfp), dtype=torch.bool, device=device),
                        torch.zeros((len(live), nfp, 3), **td), pts0[:, nm:],
                        torch.tensor(t0s, device=device), floors[torch.tensor(live, device=device)])

  def outputs(n):
    """Per position of a frame sequence (+1 dummy row for the writes of idle slots):
    the solution and the chain state after it."""
    return dict(pos=torch.zeros((n + 1, 3), **td), R=torch.zeros((n + 1, 3, 3), **td),
                q=torch.zeros((n + 1, nf), **td), pts=torch.zeros((n + 1, nm, 3), **td),
                it=torch.zeros(n + 1, dtype=torch.int32, device=device),
                on=torch.zeros((n + 1, nfp), dtype=torch.bool, device=device),
                tgt=torch.zeros((n + 1, nfp, 3), **td))

  out = outputs(NN)
  p0 = torch.tensor(so[live], device=device)
  out["pos"][p0], out["R"][p0], out["q"][p0], out["pts"][p0] = pos0, R0, q0, pts0[:, :nm]
  out["on"][p0], out["tgt"][p0] = on0, tg0
  state0 = {i: dict(pos=pos0[k:k + 1], R=R0[k:k + 1], q=q0[k:k + 1], on=on0[k:k + 1],
                    tgt=tg0[k:k + 1]) for k, i in enumerate(live)}
  # Anchors depend on the stance flags alone for which sole points are held (after a
  # frame: exactly those in stance), and their targets were set where the current
  # stance run of the point began (its touch-down): td_g[p, i] = that position.
  st_seq = np.concatenate(stf)[seq_all]
  prev = np.r_[np.zeros((1, nfp), bool), st_seq[:-1]]
  prev[so[Ns > 0]] = False  # a trial's first frame begins its runs
  begin = st_seq & ~prev
  td_g = torch.tensor(np.maximum.accumulate(np.where(begin, np.arange(NN)[:, None], 0), axis=0),
                      device=device)

  # Chunks over every trial's sequence positions 1..N-1, and per chunk (+1 dummy row
  # for masked writes) the chain state it last started from (Sd) and ended with (Ed),
  # and a run version (a slot that takes a chunk over cancels the run that had it).
  starts, ends, ctrial = [], [], []
  for i in live:
    st_i = so[i] + np.arange(1, Ns[i], chunk)
    starts.append(st_i)
    ends.append(np.minimum(st_i + chunk, so[i] + Ns[i]))
    ctrial.append(np.full(len(st_i), i))
  starts, ends, ctrial = (np.concatenate(x).astype(int) for x in (starts, ends, ctrial))
  C = len(starts)
  head = np.r_[True, ctrial[1:] != ctrial[:-1]] if C else np.zeros(0, bool)  # a trial's first
  has_next = np.r_[~head[1:], False] if C else np.zeros(0, bool)
  init_sp = np.maximum(starts - burn_in - 1, so[ctrial])
  Sd = {k: torch.cat([state0[i][k] for i in ctrial] + [state0[live[0]][k]]) for k in state0[live[0]]}
  Ed = {k: v.clone() for k, v in Sd.items()}
  ver = torch.zeros(C + 1, dtype=torch.long, device=device)
  starts_d = torch.tensor(np.r_[starts, NN], device=device)
  ends_d = torch.tensor(np.r_[ends, NN], device=device)
  next_d = torch.tensor(np.r_[has_next, False], device=device)
  root_d = torch.cat([~usek_g[seqg[starts_d[:C]]], torch.zeros(1, dtype=torch.bool,
                                                               device=device)])

  def differs(a, b, rmat):
    """Chain states a, b (dicts of [n, ...]) differ by more than tol; the root counts
    where rmat (the next frame has no pelvis fit to start from)."""
    d = ((a["q"] - b["q"]).abs().amax(1) > tol) | (a["on"] != b["on"]).any(1) \
      | (((a["tgt"] - b["tgt"]).abs().amax(2) * a["on"]).amax(1) > tol)
    return d | ((((a["pos"] - b["pos"]).abs().amax(1) > tol)
                 | ((a["R"] - b["R"]).abs().amax((1, 2)) > tol)) & rmat)

  runners = {}

  def runner(name, B, seq_t, mstep_t, grp_t, out, chained):
    """Static buffers and CUDA graphs for B slots stepping through frames seq_t
    (global frame indices; mstep_t the speed bound of each step, grp_t the trial,
    out the outputs per position). Each slot steps through its stretch on its own:
    as soon as its frame has converged it records it, updates its chain state
    (ik_trial's x_prev, q_ref, anchors) and sets up its next frame, so no slot waits
    for the slowest frame of the batch. chained: the stretches are the chunks (Sd,
    Ed, ver)."""
    key = (name, B)
    if key in runners:
      return runners[key]
    n_seq = len(seq_t)
    bt = _Batch(ik, B, smooth=True, contact=True)
    if not chained:  # the coarse sweep only gives the chunks their first start
      bt.xtol = sweep_xtol
    cs = dict(pos=torch.zeros((B, 3), **td), R=torch.eye(3, **td).repeat(B, 1, 1),
              q=torch.zeros((B, nf), **td), on=torch.zeros((B, nfp), dtype=torch.bool,
                                                           device=device),
              tgt=torch.zeros((B, nfp, 3), **td))
    lng = dict(dtype=torch.long, device=device)
    bol = dict(dtype=torch.bool, device=device)
    ctl = dict(sp=torch.zeros(B, **lng), end=torch.zeros(B, **lng),
               rec=torch.zeros(B, **lng), chunk=torch.full((B,), C, **lng),
               token=torch.zeros(B, **lng), fresh=torch.zeros(B, **bol),
               busy=torch.zeros(B, **bol), cont=torch.zeros(B, **bol))

    def put(dst, val, a):
      dst.copy_(torch.where(a, val, dst))

    def prep(mask):
      """Slots in mask: the problem of their frame from the chain state (ik_trial's
      loop body: contact_args, warm start, bounds, smoothing weights); slots past
      their stretch's end go idle."""
      sp = ctl["sp"]
      act = mask & (sp < ctl["end"])
      spc = torch.minimum(sp, torch.full_like(sp, n_seq - 1))
      t = seq_t[spc]
      a1, a2 = act[:, None], act[:, None, None]
      put(bt.P, Pg[t], a2)
      put(bt.ok, okg[t], a1)
      put(bt.group, grp_t[spc], act)
      put(bt.floor, floors[grp_t[spc].long()], act)
      put(bt.anc_on, cs["on"] & st_g[t], a1)  # contact_args: anchors out of stance go
      put(bt.anc_tgt, cs["tgt"], a2)
      xp = cs["q"]
      put(bt.x_prev, xp, a1)
      st = mstep_t[spc][:, None]
      put(bt.lo, torch.maximum(ik.jlo, xp - st), a1)
      put(bt.hi, torch.minimum(ik.jhi, xp + st), a1)
      elv_w = (0.05 + 1.0 * torch.exp(-(xp[:, selv_of] / 0.35) ** 2)) * fs
      put(bt.sm, torch.where(elv_mask > 0, elv_w, base_g), a1)
      uk = usek_g[t]
      put(bt.pos, torch.where(uk[:, None], kpos_g[t], cs["pos"]), a1)
      put(bt.R, torch.where(uk[:, None, None], kR_g[t], cs["R"]), a2)
      # the CPU's least_squares clips the start strictly into the bounds
      put(bt.q, torch.minimum(torch.maximum(xp, bt.lo + 1e-9), bt.hi - 1e-9), a1)
      put(ctl["fresh"], act, mask)
      put(ctl["busy"], act, mask)
      put(bt.done, ~act, mask)

    def post(mask):
      """Slots in mask finished their frame: touch-down, chain state, outputs; then
      their next frame."""
      sp = ctl["sp"]
      spc = torch.minimum(sp, torch.full_like(sp, n_seq - 1))
      on, tgt = touch_down(bt.anc_on, bt.anc_tgt, bt.pts[:, nm:], seq_t[spc], bt.floor)
      a1, a2 = mask[:, None], mask[:, None, None]
      put(cs["pos"], bt.pos, a1)
      put(cs["R"], bt.R, a2)
      put(cs["q"], bt.q, a1)
      put(cs["on"], on, a1)
      put(cs["tgt"], tgt, a2)
      rec = mask & (sp >= ctl["rec"])
      # (A run does not stop half-way through a chunk where it meets the stored
      # solution: a run it took the chunk over from may have written further ahead
      # before it stopped, so "the rest stands" would not hold. It goes on to the
      # chunk's end, then takes the next chunk over only if that one started
      # elsewhere.)
      w = torch.where(rec, spc, torch.full_like(spc, n_seq))
      out["pos"][w] = bt.pos
      out["R"][w] = bt.R
      out["q"][w] = bt.q
      out["pts"][w] = bt.pts[:, :nm]
      out["it"][w] = bt.iters
      out["on"][w] = on
      out["tgt"][w] = tgt
      if chained:
        ch = ctl["chunk"]
        fin = mask & (sp + 1 >= ctl["end"])  # the chunk's last frame
        wE = torch.where(fin, ch, torch.full_like(ch, C))
        for k in Ed:
          Ed[k][wE] = cs[k]
        # chunk done and the next one (same trial) last started elsewhere: take it
        # over and go on
        nx_c = torch.minimum(ch + 1, torch.full_like(ch, C))
        go = fin & ctl["cont"] & next_d[ch] \
          & differs(cs, {k: v[nx_c] for k, v in Sd.items()}, root_d[nx_c])
        wS = torch.where(go, nx_c, torch.full_like(ch, C))
        for k in Sd:
          Sd[k][wS] = cs[k]
        ver[wS] = ver[wS] + 1
        put(ctl["token"], ver[nx_c], go)
        put(ctl["chunk"], nx_c, go)
        put(ctl["end"], ends_d[nx_c], go)
        put(ctl["rec"], starts_d[nx_c], go)
      ctl["sp"].add_(mask.to(torch.long))
      prep(mask)

    def tick():
      if chained:  # a slot whose chunk another one took over stops
        gone = ctl["busy"] & (ctl["token"] != ver[ctl["chunk"]])
        put(ctl["busy"], ~gone, gone)
        put(bt.done, gone, gone)
      busy = ctl["busy"]
      conv = ik.iterate(bt, fresh=ctl["fresh"] & busy, active=busy & ~bt.done)
      ctl["fresh"].zero_()
      post(conv)

    def ticks():
      for _ in range(_TICKS):
        tick()

    ctl["end"].zero_()  # capture with every slot idle
    g_prep = _Graph(lambda: prep(torch.ones(B, dtype=torch.bool, device=device)), graphs)
    g_ticks = _Graph(ticks, graphs)
    runners[key] = (cs, ctl, g_prep, g_ticks)
    return runners[key]

  n_ticks = [0]

  def run(name, seq_t, mstep_t, grp_t, out, start, sp0, end, rec, chunks=None, cont=False):
    """Every slot i steps from position sp0[i] to end[i] (exclusive) of seq_t from
    the chain state start[k][i], recording positions >= rec[i]. chunks (optional):
    the slots' chunk indices; with cont, a slot whose chunk ends elsewhere than where
    the next chunk last started takes the next chunk over."""
    nb = len(sp0)
    B = _bucket(nb)
    chained = chunks is not None
    cs, ctl, g_prep, g_ticks = runner(name, B, seq_t, mstep_t, grp_t, out, chained)
    pad = lambda x: np.r_[x, np.zeros(B - nb, int)]  # noqa: E731
    sel = torch.tensor(pad(np.arange(nb)), device=device)
    for k in cs:
      cs[k].copy_(start[k][sel])
    ctl["sp"].copy_(torch.tensor(pad(sp0), device=device))
    ctl["end"].copy_(torch.tensor(pad(end), device=device))
    ctl["rec"].copy_(torch.tensor(pad(rec), device=device))
    if chained:
      chg = torch.tensor(np.r_[chunks, np.full(B - nb, C, int)], device=device)
      ctl["chunk"].copy_(chg)
      ctl["token"].copy_(ver[chg])
    ctl["cont"].fill_(chained and cont)
    g_prep()
    while True:
      g_ticks()
      n_ticks[0] += _TICKS
      if not bool(ctl["busy"].any()):
        break

  # Pass-1 start states: a coarse sequential sweep (per trial, the trials side by
  # side) over the positions burn_in frames before each chunk, each solved from the
  # one before with the same smoothing and anchors and the speed bound scaled to the
  # gap: the slow states the CPU carries from frame to frame (the plane of elevation
  # of a hanging arm and the forearm with it, held for tens of seconds; stance
  # anchors) come out as the sequential solve has them, where independent solves of
  # those frames picked other branches that the later passes then had to correct.
  t_sweep = time.time()
  swp = init_sp > so[ctrial]
  cpos = np.unique(init_sp[swp])
  if len(cpos):
    ctr = trial_of[cpos]
    cprev = np.r_[-1, cpos[:-1]]
    cprev = np.where(np.r_[True, ctr[1:] != ctr[:-1]], so[ctr], cprev)  # gap from t0
    out_c = outputs(len(cpos))
    tr_c = [i for i in live if (ctr == i).any()]
    c0 = np.array([np.flatnonzero(ctr == i)[0] for i in tr_c])
    c1 = np.array([np.flatnonzero(ctr == i)[-1] + 1 for i in tr_c])
    run("sweep", seqg[torch.tensor(cpos, device=device)],
        torch.tensor(v_joint * (seq_all[cpos] - seq_all[cprev]) / rate, **td),
        grp_g[torch.tensor(cpos, device=device)], out_c,
        {k: torch.cat([state0[i][k] for i in tr_c]) for k in state0[live[0]]}, c0, c1, c0)
    ki = torch.tensor(np.r_[np.searchsorted(cpos, np.where(swp, init_sp, cpos[0])), 0],
                      device=device)
    has = torch.tensor(np.r_[swp, False], device=device)
    for key in Sd:
      v = out_c[key][ki]
      Sd[key].copy_(torch.where(has.reshape(-1, *([1] * (v.dim() - 1))), v, Sd[key]))
  t_sweep = time.time() - t_sweep
  for k in Ed:
    Ed[k].copy_(Sd[k])

  need = np.ones(C, bool)
  first_sp = np.where(swp, init_sp + 1, so[ctrial] + 1)  # pass 1 includes the burn-in
  passes = 0
  hist = []
  t_pass = time.time()
  while C:
    passes += 1
    ci = np.flatnonzero(need)
    cig = torch.tensor(ci, device=device)
    sp0 = first_sp[ci] if passes == 1 else starts[ci]
    # pass 1: every chunk from its sweep state through its burn-in; later passes: the
    # chunks that start elsewhere than the chunk before them ended, with take-over
    run("chunks", seqg, mstep_g, grp_g, out, {k: v[cig] for k, v in Sd.items()}, sp0,
        ends[ci], starts[ci], chunks=ci, cont=passes > 1)
    S = {k: v[:C] for k, v in Sd.items()}
    # does every chunk start where the one before it ended? (a trial's first chunk
    # starts from its first frame)
    hd = torch.tensor(head, device=device)
    nxt = {k: torch.where(hd.reshape(-1, *([1] * (S[k].dim() - 1))), S[k],
                          torch.cat([S[k][:1], Ed[k][:C - 1]])) for k in S}
    # anchor targets straight from the latest solution at their touch-down: a foot
    # planted for seconds would otherwise pass its target on one chunk per pass
    tdp = td_g[torch.tensor(starts - 1, device=device)]  # [C, nfp]
    tg_b = out["tgt"][tdp, torch.arange(nfp, device=device)]
    nxt["tgt"] = torch.where(nxt["on"][..., None] & ~hd[:, None, None], tg_b, nxt["tgt"])
    bad = differs(nxt, S, root_d[:C]) & ~hd
    dq = (nxt["q"] - S["q"]).abs().amax(1)
    nbad = int(bad.sum())
    hist.append((passes, len(ci), nbad, float(np.degrees(float(dq.max())))))
    if verbose:
      jw = (nxt["q"] - S["q"]).abs().amax(0)
      top = ", ".join(f"{mm.free[int(j)]} {np.degrees(float(jw[j])):.2f}"
                      for j in torch.argsort(jw, descending=True)[:3])
      print(f"  pass {passes}: ran {len(ci)} chunks, {nbad} start elsewhere than the "
            f"chunk before ended (max {np.degrees(float(dq.max())):.4f} deg: {top})",
            flush=True)
    if nbad == 0:
      break
    if passes >= max_passes:
      warnings.warn(f"GPU IK: {nbad} of {C} chunks still start up to "
                    f"{np.degrees(float(dq.max())):.3g} deg from where the chunk before "
                    f"them ended after {passes} passes (max_passes); the result has "
                    "those jumps")
      break
    need = bad.cpu().numpy()
    for k in Sd:
      Sd[k][:C].copy_(torch.where(bad.reshape(-1, *([1] * (S[k].dim() - 1))), nxt[k], S[k]))
  t_pass = time.time() - t_pass

  # back to the CPU's output layout, per trial
  o_pos, o_R, o_q, o_pts = (out[k][:NN].cpu().numpy() for k in ("pos", "R", "q", "pts"))
  o_it = out["it"][:NN].cpu().numpy()
  quat_all = sRot.from_matrix(o_R).as_quat()[:, [3, 0, 1, 2]]
  per_trial = []
  for i in range(nt):
    seq, P = seqs[i], Ps[i]
    Q, E = results[i]
    if not len(seq):
      per_trial.append({})
      continue
    sl = slice(so[i], so[i] + Ns[i])
    quat = quat_all[sl]
    kq, has_pel = kab[i][2], hasp[i]
    prev_q = None
    for j, t in enumerate(seq):  # quaternion sign as the CPU's (it follows q_ref)
      ref = kq[t] if (has_pel[t] and root_init == "pelvis") or prev_q is None else prev_q
      if np.dot(quat[j], ref) < 0:
        quat[j] = -quat[j]
      prev_q = quat[j]
    Qs = np.repeat(qref[None], Ns[i], 0)
    Qs[:, :3] = o_pos[sl]
    Qs[:, 3:7] = quat
    Qs[:, mm.adr] = o_q[sl]
    _apply_eq(m, Qs)
    err = np.linalg.norm(o_pts[sl] - P[seq], axis=-1)
    err[:, ~active] = np.nan
    if outlier is not None:  # MarkerModel.solve with robust: 1-3 far markers -> NaN
      bad = (np.nan_to_num(err) > outlier) & ~HEAD[None]
      few = (bad.sum(1) > 0) & (bad.sum(1) <= 3)
      err[few] = np.where(bad[few], np.nan, err[few])
    Q[seq] = Qs
    E[seq] = err
    if pelvis_gap == "carry":  # unsolved frames after the first repeat the previous pose
      solved = np.zeros(len(P), bool)
      solved[seq] = True
      last = np.maximum.accumulate(np.where(solved, np.arange(len(P)), -1))
      later = (last >= 0) & ~solved
      Q[later] = Q[last[later]]
    Q[:seq[0]] = Q[seq[0]]
    it = o_it[sl][1:]
    per_trial.append(dict(frames=int(Ns[i]), lm_iters=it,
                          lm_iters_mean=float(it.mean()) if len(it) else 0.0,
                          lm_iters_max=int(it.max()) if len(it) else 0))
  if stats is not None:
    stats.update(time_s=time.time() - t_start, sweep_s=t_sweep, passes_s=t_pass,
                 passes=passes, chunks=C, ticks=n_ticks[0], history=hist, trials=per_trial,
                 gpu_mem_gb=torch.cuda.max_memory_allocated(device) / 1e9)
  return results

_TICKS = 16  # solver iterations per graph replay (then a check whether all slots are idle)
_MAX_IT = 100  # per frame, as the CPU's max_nfev
