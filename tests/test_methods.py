from numpy import array
import pysit2stand as s2s


def test_utility():
    x = array([0, 1, 2, 3, 4, 5])

    mn, sd, pad = s2s.utility.mov_stats(x, 3)

    assert (mn == array([1., 1., 1., 2., 3., 4.])).all()
    assert (sd == array([1., 1., 1., 1., 1., 1.])).all()
    assert pad == 2
