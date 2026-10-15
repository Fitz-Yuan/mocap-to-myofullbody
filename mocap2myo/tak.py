"""Read 3D marker trajectories straight from OptiTrack Motive .tak files (no Motive).

A .tak is a Microsoft OLE compound file (olefile opens it). Layout, worked out
from Motive 2.x/3.x takes of the MMH Elevator study (2024) and checked against
Motive's own TRC export of the same take (S03 Solid NE: 62/62 labels, max
difference 7 um = float32 rounding):
  Channels.dat  time series, one per channel: utf16 type name ("FloatChannel",
                "Vector3fChannel", "QuaternionChannel", "Float16Channel"), 0x04,
                uint16 length + utf16 channel name ("Translation", "Rotation", ...),
                16-byte owner GUID, the channel's default value, uint32 count, then
                count x (uint32 frame, value). Missing frames have no record.
  Nodes.dat     the scene graph: per node uint16 length + utf16 type ("Marker",
                "RigidBody", "Skeleton", "Constraint", "Bone", "CameraNode", ...),
                uint16 length + utf16 JSON {"properties": [{"name", "value"}]},
                then a trailer: parent GUID, 4b 7d 05, own GUID. A "Marker" trailer
                goes on with 7 bytes (marker diameter), the GUID of the asset
                (rigid body / skeleton) it is labelled to, uint32 index in that
                asset (rigid-body markers are named "Marker 00k" = index k).
                Rigid-body constraints carry ConstraintPosition (the marker's
                position in the rigid-body frame = Motive's asset definition).
  Calibration.dat, MetaData.dat, DeviceData/*.trk  not used (camera calibration,
                2D camera data, analog devices).
Coordinates: Motive world (Y-up), metres, as in a TRC export.
Not an official format: check a new Motive version against one of its TRC
exports before trusting it (compare_trc below).

  python -m mocap2myo.tak IN.tak [OUT.trc]        (needs olefile: pip install -e ".[tak]")
  python -m mocap2myo.tak IN.tak --compare EXPORT.trc
"""

import re
import struct
import sys

import numpy as np
import olefile

_VALUE_BYTES = {"FloatChannel": 4, "Vector3fChannel": 12, "QuaternionChannel": 16,
                "Float16Channel": 2}
_CHANNEL = re.compile(rb"((?:[A-Za-z0-9]\x00){3,40}C\x00h\x00a\x00n\x00n\x00e\x00l\x00)")
_JSON = '{\n    "properties"'.encode("utf-16le")
_LINK = b"\x4b\x7d\x05"


def read_channels(buf):
  """Channels.dat -> list of dict(type, name, guid, frames [n], values [n, k])."""
  out = []
  for m in _CHANNEL.finditer(buf):
    kind = m.group(1).decode("utf-16le")
    p = m.end()
    if kind not in _VALUE_BYTES or buf[p] != 4:  # a byte pattern inside the data
      continue
    n = struct.unpack_from("<H", buf, p + 1)[0]
    try:
      name = buf[p + 3:p + 3 + 2 * n].decode("utf-16le")
    except UnicodeDecodeError:
      continue
    p += 3 + 2 * n
    guid = buf[p:p + 16].hex()
    p += 16 + _VALUE_BYTES[kind]  # skip the default value
    count = struct.unpack_from("<I", buf, p)[0]
    p += 4
    vt = ("<f2",) if kind == "Float16Channel" else ("<f4", (_VALUE_BYTES[kind] // 4,))
    dt = np.dtype([("f", "<u4"), ("v", *vt)])
    if p + count * dt.itemsize > len(buf):
      continue
    rec = np.frombuffer(buf, dt, count, p)
    k = max(1, _VALUE_BYTES[kind] // 4)
    out.append(dict(type=kind, name=name, guid=guid, frames=rec["f"].astype(np.int64),
                    values=rec["v"].astype(np.float64).reshape(count, k)))
  return out


def read_nodes(buf):
  """Nodes.dat -> list (file order) of dict(type, props, guid, parent, asset, index)."""
  nodes = []
  i = buf.find(_JSON)
  while i >= 0:
    for n in range(1, 64):  # walk back over "uint16 length + utf16 type"
      s = i - 2 - 2 * n
      if s < 2 or struct.unpack_from("<H", buf, s - 2)[0] != n:
        continue
      try:
        kind = buf[s:i - 2].decode("utf-16le")
      except UnicodeDecodeError:
        continue
      if not kind.isalnum():
        continue
      end = buf.find("\n}".encode("utf-16le"), i) + 4
      js = buf[i:end].decode("utf-16le")
      props = dict(re.findall(r'"name": "([^"]*)",\s*"value": "([^"]*)"', js))
      node = dict(type=kind, props=props, guid=None, parent=None, asset=None, index=None)
      if buf[end + 16:end + 19] == _LINK:
        node["parent"] = buf[end:end + 16].hex()
        node["guid"] = buf[end + 19:end + 35].hex()
        if kind == "Marker":
          q = end + 35 + 7
          node["asset"] = buf[q:q + 16].hex()
          node["index"] = struct.unpack_from("<I", buf, q + 16)[0]
      nodes.append(node)
      break
    i = buf.find(_JSON, i + 2)
  return nodes


def read_tak(path):
  """Labelled 3D markers of a take.

  Returns dict(rate, n_frames, names [m], assets [m], xyz [T, m, 3] (NaN = not
  seen), rigid_bodies {name: {k: (x, y, z) in the body frame}}). Marker names:
  skeleton markers keep their label (e.g. LASI); rigid-body markers become
  "<body>_<k>" with the subject prefix dropped (Subject03_Box -> Box_1..6);
  unlabelled ones keep "Unlabeled NNNN"."""
  ole = olefile.OleFileIO(path)
  nodes = read_nodes(ole.openstream("Nodes.dat").read())
  chans = read_channels(ole.openstream("Channels.dat").read())
  ole.close()
  by_guid = {n["guid"]: n for n in nodes if n["guid"]}
  rate = next((float(n["props"]["kSyncFinalCameraRate"]) for n in nodes
               if "kSyncFinalCameraRate" in n["props"]), None)
  if rate is None:
    rate = float(next(n["props"]["CameraNodeCameraFrameRate"] for n in nodes
                      if "CameraNodeCameraFrameRate" in n["props"]))

  def short(asset_name):
    return re.sub(r"^Subject\d+_", "", asset_name)

  # rigid-body definitions: constraints follow their RigidBody node in file order
  rigid, cur = {}, None
  for n in nodes:
    if n["type"] in ("RigidBody", "Skeleton"):
      cur = n["props"].get("NodeName") if n["type"] == "RigidBody" else None
      if cur:
        rigid[short(cur)] = {}
    elif n["type"] == "Constraint" and cur and "ConstraintPosition" in n["props"]:
      k = int(re.sub(r"\D", "", n["props"]["NodeName"]) or 0)
      rigid[short(cur)][k] = tuple(float(x) for x in n["props"]["ConstraintPosition"].split(","))
  names, assets, tracks = [], [], []
  for c in chans:
    node = by_guid.get(c["guid"])
    if c["type"] != "Vector3fChannel" or c["name"] != "Translation" or node is None \
       or node["type"] != "Marker" or not len(c["frames"]):
      continue
    label = node["props"].get("NodeName", "?")
    owner = by_guid.get(node["asset"])
    if owner is not None and owner["type"] == "RigidBody":
      asset = short(owner["props"]["NodeName"])
      # Motive's own name is "Marker 00k" (k = index in the asset); a name given
      # by hand in Motive (e.g. Box_1 in a processed take) is kept: its number
      # need not be k
      if re.fullmatch(r"Marker \d+", label):
        label = f"{asset}_{node['index']}"
    else:
      asset = owner["props"]["NodeName"] if owner is not None else ""
    names.append(label)
    assets.append(asset)
    tracks.append(c)
  T = 1 + max(int(c["frames"].max()) for c in tracks)
  xyz = np.full((T, len(tracks), 3), np.nan)
  for j, c in enumerate(tracks):
    xyz[c["frames"], j] = c["values"]
  return dict(rate=rate, n_frames=T, names=names, assets=assets, xyz=xyz, rigid_bodies=rigid)


def read_trc(path):
  """TRC -> (names, frames [T], xyz [T, m, 3])."""
  lines = open(path).read().splitlines()
  names = [n for n in lines[3].split("\t")[2:] if n.strip()]
  rows = [ln.split("\t") for ln in lines[5:] if ln.strip()]
  w = 2 + 3 * len(names)
  A = np.array([[float(x) if x.strip() else np.nan for x in r[:w]] + [np.nan] * (w - len(r))
                for r in rows])
  return names, A[:, 0].astype(int), A[:, 2:].reshape(len(A), len(names), 3)


def write_trc(path, take):
  names, X, rate = take["names"], take["xyz"], take["rate"]
  T = len(X)
  with open(path, "w") as f:
    f.write(f"PathFileType\t4\t(X/Y/Z)\t{path}\n")
    f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
            "OrigDataStartFrame\tOrigNumFrames\n")
    f.write(f"{rate:g}\t{rate:g}\t{T}\t{len(names)}\tm\t{rate:g}\t0\t{T}\n")
    f.write("Frame#\tTime\t" + "".join(f"{n}\t\t\t" for n in names) + "\n")
    f.write("\t\t" + "\t".join(f"X{k}\tY{k}\tZ{k}" for k in range(1, len(names) + 1)) + "\n\n")
    for t in range(T):
      v = ["" if np.isnan(x) else f"{x:.6f}" for x in X[t].ravel()]
      f.write(f"{t}\t{t / rate:.5f}\t" + "\t".join(v) + "\n")


def compare_trc(take, trc_path):
  """Marker-by-marker check against Motive's TRC export of the same take."""
  names, frames, Y = read_trc(trc_path)
  worst, missing = 0.0, []
  for j, n in enumerate(names):
    if n not in take["names"]:
      missing.append(n)
      continue
    X = take["xyz"][frames, take["names"].index(n)]
    both = ~np.isnan(X[:, 0]) & ~np.isnan(Y[:, j, 0])
    only = np.isnan(X[:, 0]) != np.isnan(Y[:, j, 0])
    if only.any():
      missing.append(f"{n} (presence differs in {only.sum()} frames)")
    if both.any():
      worst = max(worst, float(np.abs(X[both] - Y[both, j]).max()))
  return dict(trc_markers=len(names), not_matched=missing, max_abs_diff_m=worst)


if __name__ == "__main__":
  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__.strip())
    sys.exit(0)
  tk = read_tak(sys.argv[1])
  lab = [n for n in tk["names"] if not n.startswith("Unlabeled")]
  print(f"{sys.argv[1]}: {tk['n_frames']} frames @ {tk['rate']:g} Hz, {len(lab)} labelled + "
        f"{len(tk['names']) - len(lab)} unlabelled markers; rigid bodies "
        f"{ {k: len(v) for k, v in tk['rigid_bodies'].items()} }")
  if len(sys.argv) > 3 and sys.argv[2] == "--compare":
    print(compare_trc(tk, sys.argv[3]))
  elif len(sys.argv) > 2:
    write_trc(sys.argv[2], tk)
    print("wrote", sys.argv[2])
