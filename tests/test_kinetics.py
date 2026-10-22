"""Segment masses (bsip_factors) and the plate-to-foot assignment (feet_from_plates)."""
import numpy as np
import pytest

from mocap2myo import kinetics


@pytest.fixture(scope="module")
def fullbody():
  from mocap2myo import profile
  p = profile.load("myofullbody")
  return p, p.compile()


@pytest.mark.parametrize("table,sex", [("deleva", "m"), ("deleva", "f"), ("deleva", None), ("dempster", None)])
def test_bsip_keeps_total_and_sets_groups(fullbody, table, sex):
  p, m = fullbody
  f, info = kinetics.bsip_factors(m, p, sex, table)
  assert np.isclose((m.body_mass * f).sum(), m.body_mass.sum())
  ref = kinetics.DEMPSTER_MASS if table == "dempster" else kinetics.DE_LEVA_MASS.get(sex) or {
    k: (kinetics.DE_LEVA_MASS["m"][k] + kinetics.DE_LEVA_MASS["f"][k]) / 2 for k in kinetics.DE_LEVA_MASS["m"]}
  assert info["thigh_r"][1] == pytest.approx(ref["thigh"], abs=0.01)
  assert info["head"][1] == pytest.approx(ref["head"], abs=0.01)


def test_tables_sum_to_100():
  for t in (*kinetics.DE_LEVA_MASS.values(), kinetics.DEMPSTER_MASS):
    total = t["head"] + t["trunk"] + 2 * sum(t[k] for k in ("upperarm", "forearm_hand", "thigh", "shank", "foot"))
    assert total == pytest.approx(100.0, abs=0.05)


def _feet_case(foot_r_z):
  """One loaded plate at z = 0.3 (a stair step) under the left foot; the right foot is 0.1 m
  away horizontally at height foot_r_z."""
  T = 1
  sole = np.zeros((T, 4, 3))
  sole[:, :2] = [[0.0, 0.0, 0.3], [0.05, 0.0, 0.3]]          # left foot points
  sole[:, 2:] = [[0.10, 0.05, foot_r_z], [0.15, 0.05, foot_r_z]]  # right foot points
  idx = {"l": np.array([0, 1]), "r": np.array([2, 3])}
  plates = dict(force=np.array([[[0.0, 0.0, 600.0]]]), cop=np.array([[[0.02, 0.0, 0.3]]]),
                free=np.zeros((T, 1)))
  z2 = np.zeros((T, 2, 3))
  return kinetics.feet_from_plates(plates, z2, z2, np.zeros((T, 2)), np.zeros((T, 2), bool), sole, idx,
                                   np.array([[0.0, 0.0, 600.0]]), np.zeros((T, 3)), np.array([0.3]))


@pytest.mark.parametrize("z", [0.62, 0.0])  # swinging over the step / on the step below
def test_plate_goes_to_the_foot_on_it(z):
  foot_F, _, _, source, contact = _feet_case(z)
  assert source[0].tolist() == [1, 0] and foot_F[0, 0, 2] == 600.0 and contact[0, 0]


def test_both_feet_on_the_plate_is_ambiguous():
  _, _, _, source, _ = _feet_case(0.3)
  assert source[0].tolist() == [0, 0]
