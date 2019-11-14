import pytest
from numpy import allclose
from pandas import to_datetime

from sit2standpy.detectors.detectors import _get_still, _integrate_acc, Stillness, Displacement


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


def test_stillness_detection(raw_accel, time, filt_accel_rm, rm_accel_rm, power_peaks_rm, still_times,
                             still_durations, still_max_acc, still_min_acc, still_sparc):
    timestamps = to_datetime(time, unit='us')

    sts = Stillness().apply(raw_accel, filt_accel_rm, rm_accel_rm, timestamps, 1/128, power_peaks_rm)

    times = [to_datetime(key).timestamp() * 1e6 for key in sts]
    durations = [sts[key].duration for key in sts]
    max_acc = [sts[key].max_acceleration for key in sts]
    min_acc = [sts[key].min_acceleration for key in sts]
    sparc = [sts[key].sparc for key in sts]

    assert allclose(times, still_times)
    assert allclose(durations, still_durations)
    assert allclose(max_acc, still_max_acc)
    assert allclose(min_acc, still_min_acc)
    assert allclose(sparc, still_sparc)


def test_displacement_detection(raw_accel, time, filt_accel_rm, rm_accel_rm, power_peaks_rm, disp_times,
                                disp_durations, disp_max_acc, disp_min_acc, disp_sparc):
    timestamps = to_datetime(time, unit='us')

    sts = Displacement().apply(raw_accel, filt_accel_rm, rm_accel_rm, timestamps, 1/128, power_peaks_rm)

    times = [to_datetime(key).timestamp() * 1e6 for key in sts]
    durations = [sts[key].duration for key in sts]
    max_acc = [sts[key].max_acceleration for key in sts]
    min_acc = [sts[key].min_acceleration for key in sts]
    sparc = [sts[key].sparc for key in sts]

    assert allclose(times, disp_times)
    assert allclose(durations, disp_durations)
    assert allclose(max_acc, disp_max_acc)
    assert allclose(min_acc, disp_min_acc)
    assert allclose(sparc, disp_sparc)
