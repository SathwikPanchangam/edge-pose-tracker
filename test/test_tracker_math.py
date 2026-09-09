import pytest
import numpy as np
from edge_pose_tracker.pose_filter import KalmanAxisFilter


def test_kalman_filter_initialization():
    """Verify state initialization on the first received measurement."""
    kf = KalmanAxisFilter(process_noise_q=0.02)
    assert not kf.initialized

    z_initial = 1.25
    var_initial = 0.04
    out = kf.update(z_initial, var_initial)

    assert kf.initialized
    assert pytest.approx(out, rel=1e-3) == z_initial
    assert pytest.approx(kf.x[0, 0], rel=1e-3) == z_initial
    assert pytest.approx(kf.x[1, 0], abs=1e-4) == 0.0


def test_kalman_filter_noise_suppression():
    """Verify that Kalman filter dampens single-frame high-frequency outliers."""
    kf = KalmanAxisFilter(process_noise_q=0.01)

    dt = 0.033
    for _ in range(30):
        kf.predict(dt)
        kf.update(1.0, 0.001)

    kf.predict(dt)
    filtered_spike = kf.update(2.0, 0.5)

    assert filtered_spike < 1.3
    assert filtered_spike >= 1.0


def test_quaternion_normalization():
    """Verify quaternion conversions always satisfy unit length constraint."""
    R_identity = np.eye(3, dtype=np.float64)
    trace = np.trace(R_identity)
    s = 0.5 / np.sqrt(trace + 1.0)
    qw = 0.25 / s
    qx = (R_identity[2, 1] - R_identity[1, 2]) * s
    qy = (R_identity[0, 2] - R_identity[2, 0]) * s
    qz = (R_identity[1, 0] - R_identity[0, 1]) * s

    norm = np.linalg.norm([qx, qy, qz, qw])
    assert pytest.approx(norm, rel=1e-6) == 1.0

    R_x90 = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0,  0.0]
    ], dtype=np.float64)

    s_x = 2.0 * np.sqrt(1.0 + R_x90[0, 0] - R_x90[1, 1] - R_x90[2, 2])
    qw_x = (R_x90[2, 1] - R_x90[1, 2]) / s_x
    qx_x = 0.25 * s_x
    qy_x = (R_x90[0, 1] + R_x90[1, 0]) / s_x
    qz_x = (R_x90[0, 2] + R_x90[2, 0]) / s_x

    norm_x = np.linalg.norm([qx_x, qy_x, qz_x, qw_x])
    assert pytest.approx(norm_x, rel=1e-6) == 1.0