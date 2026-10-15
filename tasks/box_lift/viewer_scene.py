"""Box lifting, viewer: the objects and tracks of a reference clip
(pipeline/export_viewer.py, M2M_TASK=box_lift), in the viewer's generic form."""

import numpy as np


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


def viewer_scene(ref, idx):
  """Objects and tracks of a reference (build_reference.py output) for the frames
  idx. Returns (objects, tracks, arrays), arrays = {name: (kind, per-frame array)}
  with kind "pos" (m), "quat" (wxyz) or "raw" (stored as given)."""
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
  arrays = dict(obj_box_pos=("pos", ref["box_pos"][idx]),
                obj_box_quat=("quat", ref["box_quat"][idx]),
                trk_contact=("raw", (ref["grasp"][idx] > 0.5).astype(np.uint8)),
                trk_box_support=("raw", ref["support"][idx].astype(np.uint8)))
  return objects, tracks, arrays
