import pytest
from numpy import allclose

from pysit2stand.detectors.detectors import _get_still, _integrate_acc, Stillness, Displacement


def test_get_still(filt_accel_rm, stillness, still_starts, still_stops):
    still, starts, stops = _get_still(filt_accel_rm, 1/128, window=0.3, gravity=9.81,
                                      thresholds={'stand displacement': 0.125, 'transition velocity': 0.2,
                                                  'accel moving avg': 0.2, 'accel moving std': 0.1,
                                                  'jerk moving avg': 2.5, 'jerk moving std': 3})

    assert allclose(still, stillness)
    assert allclose(starts, still_starts)
    assert allclose(stops, still_stops)


def test_integrate_acc(integrate_data):
    vel, pos = _integrate_acc(integrate_data[0], integrate_data[-1], True)

    assert allclose(vel, integrate_data[1], atol=5e-3)  # fairly loose tolerance due to drift, etc
    assert allclose(pos, integrate_data[2], atol=2e-2)
