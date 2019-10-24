import pytest
from pandas import to_datetime
from numpy import loadtxt
from importlib import resources


@pytest.fixture
def start_t1():
    return to_datetime(1567616049649, unit='ms')


@pytest.fixture
def end_t1():
    return to_datetime(1567616049649 + 1e3, unit='ms')


@pytest.fixture
def raw_accel():
    # pull sample data
    with resources.path('pysit2stand.data', 'sample.csv') as file_path:
        acc = loadtxt(file_path, dtype=float, delimiter=',', usecols=(1, 2, 3))

    return acc


@pytest.fixture
def filt_accel():
    # pull the filtered data
    with resources.path('pysit2stand.data', '.filter_result.csv') as file_path:
        filt_accel = loadtxt(file_path, dtype=float, delimiter=',', usecols=0)

    return filt_accel


@pytest.fixture
def rm_accel():
    # pull the rolling mean acceleration
    with resources.path('pysit2stand.data', '.filter_result.csv') as file_path:
        rm_accel = loadtxt(file_path, dtype=float, delimiter=',', usecols=1)

    return rm_accel


@pytest.fixture
def power():
    # pull the power measure
    with resources.path('pysit2stand.data', '.filter_result.csv') as file_path:
        power = loadtxt(file_path, dtype=float, delimiter=',', usecols=2)

    return power


@pytest.fixture
def power_peaks():
    # pull the power peaks
    with resources.path('pysit2stand.data', '.filter_result.csv') as file_path:
        power_peaks = loadtxt(file_path, dtype=int, delimiter=',', usecols=3)

    power_peaks = power_peaks[power_peaks != -1]  # remove filler values

    return power_peaks
