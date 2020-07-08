from pytest import fixture
from importlib import resources
from numpy import loadtxt, random
from pandas import date_range, to_datetime


# -------------------------------------------------------------------------------------------------
#                               DWT RECONSTRUCTED FILTERED DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='package')
def filt_accel_dwt():
    # pull the filtered data
    with resources.path('sit2standpy.data', 'filter_results_dwt.csv') as file_path:
        filt_accel = loadtxt(file_path, dtype=float, delimiter=',', usecols=0)

    return filt_accel


@fixture(scope='package')
def rec_accel_dwt():
    # pull the rolling mean acceleration
    with resources.path('sit2standpy.data', 'filter_results_dwt.csv') as file_path:
        rm_accel = loadtxt(file_path, dtype=float, delimiter=',', usecols=1)

    return rm_accel


@fixture(scope='package')
def power_dwt():
    # pull the power measure
    with resources.path('sit2standpy.data', 'filter_results_dwt.csv') as file_path:
        power = loadtxt(file_path, dtype=float, delimiter=',', usecols=2)

    return power


@fixture(scope='package')
def power_peaks_dwt():
    # pull the power peaks
    with resources.path('sit2standpy.data', 'filter_results_dwt.csv') as file_path:
        power_peaks = loadtxt(file_path, dtype=int, delimiter=',', usecols=3)

    power_peaks = power_peaks[power_peaks != -1]  # remove filler values

    return power_peaks


# -------------------------------------------------------------------------------------------------
#                               GENERATED TIMESTAMP DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='package')
def overnight_time_accel():
    # generate some time that spans overnight, to testing windowing
    ts = date_range(start='2019-10-10 16:00', end='2019-10-11 12:00', freq='1H').astype(int)
    # generate random acceleration values with the same shape
    acc = random.rand(ts.size)

    return ts, acc


@fixture(scope='package')
def windowed_timestamps():
    ts = {'Day 1': to_datetime(['2019-10-10 16:00:00', '2019-10-10 17:00:00', '2019-10-10 18:00:00',
                                '2019-10-10 19:00:00', '2019-10-10 20:00:00'], format='%Y-%m-%d %H:%M:%S'),
          'Day 2': to_datetime(['2019-10-11 08:00:00', '2019-10-11 09:00:00', '2019-10-11 10:00:00',
                                '2019-10-11 11:00:00', '2019-10-11 12:00:00'], format='%Y-%m-%d %H:%M:%S')}
    return ts


@fixture(scope='package')
def timestamps_time_accel():
    ts = date_range(start='2019-10-10 16:00', end='2019-10-10 16:02', freq='0.05S')
    time = ts.astype(int)
    acc = random.rand(ts.size)

    return ts, time, acc
