# test importing of required modules and sit2standpy package


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
    import sit2standpy
    from sit2standpy import Sit2Stand, detectors, mov_stats, Transition, TransitionQuantifier, \
        AccelerationFilter, process_timestamps, __version__
    from sit2standpy.detectors import Stillness, Displacement

    return
