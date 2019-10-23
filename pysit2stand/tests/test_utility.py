import pytest
from pandas import to_datetime
from numpy import isclose, allclose
from pysit2stand.utility import Transition, mov_stats


@pytest.mark.parametrize(('start_time', 'stop_time'), (
        ("to_datetime(1567616049649, unit='ms')", "to_datetime(1567616049649 - 1e3, unit='ms')"),
        ("to_datetime(1567616049649, unit='ms')", "to_datetime(1567616049649 + 20e3, unit='ms')")))
def test_transition_errors(start_time, stop_time):
    with pytest.raises(ValueError) as e_info:
        Transition((start_time, stop_time))


def test_transition_duration():
    t1 = to_datetime(1567616049649, unit='ms')
    t2 = to_datetime(1567616049649 + 1e3, unit='ms')

    trans = Transition((t1, t2))

    assert isclose(trans.duration, 1.0)
