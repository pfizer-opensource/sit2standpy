import pytest
from numpy import allclose

from pysit2stand.detectors.detectors import _get_still, _integrate_acc, Stillness, Displacement


def test_integrate_acc(integrate_data):
    vel, pos = _integrate_acc(integrate_data[0], integrate_data[-1], True)

    assert allclose(vel, integrate_data[1], atol=5e-3)  # fairly loose tolerance due to drift, etc
    assert allclose(pos, integrate_data[2], atol=2e-2)
