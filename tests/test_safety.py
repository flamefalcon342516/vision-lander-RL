"""Safety layer: every rule, and its priority order."""
import numpy as np
import pytest

from lander.deploy.safety import Decision, SafetyConfig, VehicleState, check

CFG = SafetyConfig()


def st(**kw) -> VehicleState:
    base = dict(mode="GUIDED", armed=True, heartbeat_age_s=0.1, alt_m=5.0, dist_from_home_m=5.0,
                marker_lost_s=0.0, marker_xy_err_m=0.1)
    return VehicleState(**{**base, **kw})


def test_nominal_command_passes_unchanged():
    out = check(st(), np.array([1.0, -0.5, 0.3]), CFG)
    assert out.decision == Decision.SEND and np.allclose(out.velocity, [1.0, -0.5, 0.3]) and not out.reasons


@pytest.mark.parametrize("mode,armed", [("LOITER", True), ("RTL", True), ("GUIDED", False)])
def test_pilot_override_or_disarm_silences_us(mode, armed):
    assert check(st(mode=mode, armed=armed), np.ones(3), CFG).decision == Decision.SILENT


def test_heartbeat_loss_silences_us():
    assert check(st(heartbeat_age_s=3.0), np.ones(3), CFG).decision == Decision.SILENT


def test_nan_policy_output_hovers():
    out = check(st(), np.array([np.nan, 0, 0]), CFG)
    assert out.decision == Decision.HOVER and np.allclose(out.velocity, 0)


def test_geofence_triggers_rtl():
    assert check(st(dist_from_home_m=60.0), np.zeros(3), CFG).decision == Decision.RTL


def test_marker_lost_hover_then_rtl():
    assert check(st(marker_lost_s=3.0), np.ones(3), CFG).decision == Decision.HOVER
    assert check(st(marker_lost_s=9.0), np.ones(3), CFG).decision == Decision.RTL


def test_ceiling_blocks_climb_but_not_descent():
    up = check(st(alt_m=16.0), np.array([0, 0, -1.0]), CFG)
    down = check(st(alt_m=16.0), np.array([0, 0, 1.0]), CFG)
    assert up.velocity[2] == 0 and down.velocity[2] == 1.0


def test_floor_blocks_blind_descent_only():
    blind = check(st(alt_m=1.0, marker_lost_s=1.0), np.array([0, 0, 1.0]), CFG)
    off_centre = check(st(alt_m=1.0, marker_xy_err_m=1.0), np.array([0, 0, 1.0]), CFG)
    confirmed = check(st(alt_m=1.0), np.array([0, 0, 1.0]), CFG)
    assert blind.velocity[2] == 0 and off_centre.velocity[2] == 0 and confirmed.velocity[2] == 1.0


def test_speed_limits_clamp_direction_preserving():
    out = check(st(), np.array([6.0, 8.0, 3.0]), CFG)
    assert np.isclose(np.linalg.norm(out.velocity[:2]), CFG.v_xy_max_mps)
    assert np.isclose(out.velocity[1] / out.velocity[0], 8 / 6)   # direction kept
    assert out.velocity[2] == CFG.v_down_max_mps


def test_priority_pilot_beats_everything():
    out = check(st(mode="LOITER", dist_from_home_m=99, marker_lost_s=99), np.array([np.nan] * 3), CFG)
    assert out.decision == Decision.SILENT
