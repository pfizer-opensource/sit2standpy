import pytest
from pandas import to_datetime
from numpy import allclose

from sit2standpy import Sit2Stand


def test_error(raw_accel, time):
    sts = Sit2Stand(method='TESTING')

    with pytest.raises(ValueError) as e_info:
        sts.apply(raw_accel, time, time_units='us')


def test_sit2stand_stillness(raw_accel, time, still_times, still_durations, still_max_acc, still_min_acc, still_sparc):
    sts = Sit2Stand(method='stillness')

    sist = sts.apply(raw_accel, time, time_units='us')

    times = [to_datetime(key).timestamp() * 1e6 for key in sist]
    durations = [sist[key].duration for key in sist]
    max_acc = [sist[key].max_acceleration for key in sist]
    min_acc = [sist[key].min_acceleration for key in sist]
    sparc = [sist[key].sparc for key in sist]

    assert allclose(times, still_times)
    assert allclose(durations, still_durations)
    assert allclose(max_acc, still_max_acc)
    assert allclose(min_acc, still_min_acc)
    assert allclose(sparc, still_sparc)


def test_sit2stand_displacement(raw_accel, time, disp_times, disp_durations, disp_max_acc, disp_min_acc, disp_sparc):
    sts = Sit2Stand(method='displacement')

    sist = sts.apply(raw_accel, time, time_units='us')

    times = [to_datetime(key).timestamp() * 1e6 for key in sist]
    durations = [sist[key].duration for key in sist]
    max_acc = [sist[key].max_acceleration for key in sist]
    min_acc = [sist[key].min_acceleration for key in sist]
    sparc = [sist[key].sparc for key in sist]

    assert allclose(times, disp_times)
    assert allclose(durations, disp_durations)
    assert allclose(max_acc, disp_max_acc)
    assert allclose(min_acc, disp_min_acc)
    assert allclose(sparc, disp_sparc)

