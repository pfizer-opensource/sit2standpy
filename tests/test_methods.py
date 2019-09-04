from numpy import array
import pysit2stand as s2s
from pandas import to_datetime


# UTILITY
def test_mov_stats():
    x = array([0, 1, 2, 3, 4, 5])

    mn, sd, pad = s2s.utility.mov_stats(x, 3)

    assert (mn == array([1., 1., 1., 2., 3., 4.])).all()
    assert (sd == array([1., 1., 1., 1., 1., 1.])).all()
    assert pad == 2


def test_transition():
    t1 = to_datetime(1567616049649, unit='ms')
    t2 = to_datetime(1567616049649 + 1e3, unit='ms')

    trans = s2s.Transition(times=(t1, t2))

    assert trans.duration == 1.
