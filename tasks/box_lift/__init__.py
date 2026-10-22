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
  kinetics_loads(...)       pipeline/kinetics.py: the box's load on each hand while
                            it is carried (loads.py)
  check_model(profile)      pipeline/config.py task(): the model (M2M_MODEL) must have
                            hands (body.py; MyoLeg, Gait2354 and MS-Human-700 Locomotion
                            are refused: M2M_TASK=none)
  CLIP_LABEL                what one clip of the reference is (a carry cycle)
The task's bodies (hands and palms, feet, the dataset's tracked bodies) come from the
model profile's canonical segments and hand / sole definitions (body.py), so it runs on
every model with arms: MyoFullBody, MS-Human-700, MASS, Rajagopal 2016, Lai-Uhlrich 2022.
Its own stages, after the IK: build_reference.py (floor alignment, box size,
grasp, support, cycles -> <M2M_REF_DIR>/<trial>.npz) and build_dataset.py (one
clip per carry cycle -> data/<M2M_DATA>/, written with motion_io.py).
"""

from tasks.box_lift.body import check_model  # noqa: F401
from tasks.box_lift.box_fit import prepare_setup, prepare_trial  # noqa: F401
from tasks.box_lift.checks import qa_trial  # noqa: F401
from tasks.box_lift.loads import kinetics_loads  # noqa: F401
from tasks.box_lift.viewer_scene import viewer_scene  # noqa: F401

CLIP_LABEL = "carry"
