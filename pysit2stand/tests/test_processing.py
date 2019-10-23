import pytest
from numpy import isclose, allclose, array
from pysit2stand.processing import AccelerationFilter, process_timestamps


def test_accelerationfilter_none_inputs():
    af = AccelerationFilter()

    assert isclose(af.power_start_f, 0.0)
    assert isclose(af.power_end_f, 0.5)
    assert af.power_peak_kw == {'height': 90}
