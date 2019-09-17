# test importing of required modules and pysit2stand package


def test_numpy():
    import numpy

    return


def test_scipy():
    import scipy

    return


def test_pywt():
    import pywt

    return


def test_pysit2stand():
    import pysit2stand
    from pysit2stand.utility import Transition, AccFilter, mov_stats
    from pysit2stand import Sit2Stand
    from pysit2stand.quantify import TransitionQuantifier
    from pysit2stand.detectors import Stillness, Displacement

    return
