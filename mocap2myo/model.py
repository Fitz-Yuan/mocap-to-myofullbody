"""Path of the MyoFullBody model used by the pipeline.

MyoFullBody (MuscleMimic team, built upon MyoSuite / MyoSim models; Apache-2.0, see
assets/myofullbody/LICENSE and NOTICE): 416 muscles. Two variants ship here:
  kneefix   (default) the left knee's coupled rotation / translation fixed: in the
            released file its axes were mirrored and its coefficients negated again,
            so the left shank turned the wrong way with flexion.
  original  the released file unchanged (set MYOFULLBODY_MODEL=original).
"""

import os
from pathlib import Path

_BODY_DIR = Path(__file__).resolve().parent / "assets" / "myofullbody" / "body"
MYOFULLBODY_XML_ORIGINAL = _BODY_DIR / "myofullbody_clean.xml"
MYOFULLBODY_XML_KNEEFIX = _BODY_DIR / "myofullbody_clean_kneefix.xml"
MYOFULLBODY_MODEL = os.environ.get("MYOFULLBODY_MODEL", "kneefix")
MYOFULLBODY_XML = {"original": MYOFULLBODY_XML_ORIGINAL,
                   "kneefix": MYOFULLBODY_XML_KNEEFIX}[MYOFULLBODY_MODEL]
