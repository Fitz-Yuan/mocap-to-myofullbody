"""The Motive CSV reader and the marker sets."""
import numpy as np
import pytest

from mocap2myo import markerset, readers

HEAD = ("Format Version,1.23,Take Name,t,Capture Frame Rate,120.000000,Export Frame Rate,120.000000,"
        "Length Units,Millimeters,Coordinate Space,Global\n\n")


def _csv(tmp_path, body):
  p = tmp_path / "take.csv"
  p.write_text(HEAD + body)
  return str(p)


def test_markers_read_bone_markers_left_out(tmp_path):
  body = (",Type,Bone Marker,Bone Marker,Bone Marker,Marker,Marker,Marker,Marker,Marker,Marker\n"
          ",Name,S:LHeel,S:LHeel,S:LHeel,S:LHeel,S:LHeel,S:LHeel,S:HeadTop,S:HeadTop,S:HeadTop\n"
          ",ID,1,1,1,2,2,2,3,3,3\n"
          ",,Position,Position,Position,Position,Position,Position,Position,Position,Position\n"
          "Frame,Time (Seconds),X,Y,Z,X,Y,Z,X,Y,Z\n"
          "0,0.0,9,9,9,100,50,200,120,1700,210\n"
          "1,0.0083,9,9,9,,,,121,1701,211\n")
  r = readers.read_trial(_csv(tmp_path, body), up="auto")
  assert r["kind"] == "csv" and r["rate"] == 120.0 and r["up"] == "y"
  assert r["labels"] == ["LHeel", "HeadTop"]  # the measured markers, prefix stripped
  X = r["xyz"]
  assert X.shape == (2, 2, 3) and np.isnan(X[1, 0]).all()  # an empty cell is a gap
  assert np.allclose(X[0, 1], [0.12, -0.21, 1.70])  # mm -> m, Y up -> Z up


def test_bones_only_export_is_refused(tmp_path):
  body = (",Type,Bone,Bone,Bone\n,Name,S:Hip,S:Hip,S:Hip\n,ID,1,1,1\n,,Position,Position,Position\n"
          "Frame,Time (Seconds),X,Y,Z\n0,0,1,2,3\n")
  with pytest.raises(ValueError, match="Markers ticked"):
    readers.read_trial(_csv(tmp_path, body))


@pytest.mark.parametrize("name", markerset.names())
def test_marker_sets_load(name):
  ms = markerset.load(name)
  assert len(ms.PELVIS_LABELS) == 4 and len(ms.LABELS) >= 16
