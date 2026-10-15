"""Box lifting (the case study: MMH Elevator). Subjects lift a box off a platform,
carry it, put it on a shelf and back; the box, shelf, platform and floor are
tracked with their own markers.

The generic stages call these hooks when M2M_TASK=box_lift (the default):
  prepare_setup(read)       pipeline/prepare_markers.py, once: the box marker template
  prepare_trial(...)        pipeline/prepare_markers.py, per trial: box pose, shelf,
                            platform and floor from the object markers -> extra
                            fields of the marker npz (box_fit.py)
  qa_trial(...)             pipeline/qa.py: box rest heights, hands on the box
                            while carrying, carry cycles (checks.py)
  viewer_scene(ref, idx)    pipeline/export_viewer.py: box, shelf and platform,
                            hand contact and box support tracks (viewer_scene.py)
  CLIP_LABEL                what one clip of the reference is (a carry cycle)
Its own stages, after the IK: build_reference.py (floor alignment, box size,
grasp, support, cycles -> <M2M_REF_DIR>/<trial>.npz) and build_dataset.py (one
clip per carry cycle -> data/<M2M_DATA>/, written with motion_io.py).
"""

from tasks.box_lift.box_fit import prepare_setup, prepare_trial  # noqa: F401
from tasks.box_lift.checks import qa_trial  # noqa: F401
from tasks.box_lift.viewer_scene import viewer_scene  # noqa: F401

CLIP_LABEL = "carry"
