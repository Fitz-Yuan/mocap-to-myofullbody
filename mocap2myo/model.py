"""Path of the MyoFullBody model used by the pipeline.

MyoFullBody (MuscleMimic team, built upon MyoSuite / MyoSim models): 416 muscles. Its
licence terms are in assets/myofullbody/LICENSE and NOTICE (Apache-2.0, with a
non-commercial restriction on the arm and hand parts derived from MoBL-ARMS; see the
NOTICE). Two variants ship here, both copies of the released body/myofullbody.xml
without its scene include (floor and skybox) and with a headlight / offscreen-size
block added:
  kneefix   (default, body/myofullbody_clean_kneefix.xml) the left knee's coupled
            rotation / translation fixed in leg/assets/myolegs_assets_kneefix.xml: in
            the released file its axes were mirrored and its coefficients negated
            again, so the left shank turned the wrong way with flexion.
  original  (MYOFULLBODY_MODEL=original, body/myofullbody_clean.xml) the released
            leg assets, knee coupling as released.
"""

import os
from pathlib import Path

_BODY_DIR = Path(__file__).resolve().parent / "assets" / "myofullbody" / "body"
MYOFULLBODY_XML_ORIGINAL = _BODY_DIR / "myofullbody_clean.xml"
MYOFULLBODY_XML_KNEEFIX = _BODY_DIR / "myofullbody_clean_kneefix.xml"
MYOFULLBODY_MODEL = os.environ.get("MYOFULLBODY_MODEL", "kneefix")
_XML = {"original": MYOFULLBODY_XML_ORIGINAL, "kneefix": MYOFULLBODY_XML_KNEEFIX}
if MYOFULLBODY_MODEL not in _XML:
  raise SystemExit(f"MYOFULLBODY_MODEL={MYOFULLBODY_MODEL!r}: expected kneefix or original")
MYOFULLBODY_XML = _XML[MYOFULLBODY_MODEL]
