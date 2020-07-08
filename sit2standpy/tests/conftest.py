from pytest import fixture
from importlib import resources
from numpy import loadtxt, ascontiguousarray
import h5py


# -------------------------------------------------------------------------------------------------
#                               RAW DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='package')
def raw_accel():
    # pull sample data
    with resources.path('sit2standpy.data', 'sample.csv') as file_path:
        acc = loadtxt(file_path, dtype=float, delimiter=',', usecols=(1, 2, 3))

    return ascontiguousarray(acc)


@fixture(scope='package')
def time():
    # pull sample time data
    with resources.path('sit2standpy.data', 'sample.csv') as file_path:
        time = loadtxt(file_path, dtype=float, delimiter=',', usecols=0)

    return ascontiguousarray(time)


# -------------------------------------------------------------------------------------------------
#                               ROLLING MEAN FILTERED DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='package')
def filt_accel_rm():
    # pull the filtered data
    with resources.path('sit2standpy.data', 'filter_results_rm.csv') as file_path:
        filt_accel = loadtxt(file_path, dtype=float, delimiter=',', usecols=0)

    return filt_accel


@fixture(scope='package')
def rm_accel_rm():
    # pull the rolling mean acceleration
    with resources.path('sit2standpy.data', 'filter_results_rm.csv') as file_path:
        rm_accel = loadtxt(file_path, dtype=float, delimiter=',', usecols=1)

    return rm_accel


@fixture(scope='package')
def power_rm():
    # pull the power measure
    with resources.path('sit2standpy.data', 'filter_results_rm.csv') as file_path:
        power = loadtxt(file_path, dtype=float, delimiter=',', usecols=2)

    return power


@fixture(scope='package')
def power_peaks_rm():
    # pull the power peaks
    with resources.path('sit2standpy.data', 'filter_results_rm.csv') as file_path:
        power_peaks = loadtxt(file_path, dtype=int, delimiter=',', usecols=3)

    power_peaks = power_peaks[power_peaks != -1]  # remove filler values

    return power_peaks


# -------------------------------------------------------------------------------------------------
#                               STILLNESS DETECTION DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='package')
def still_times():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['times'][()]
    return res


@fixture(scope='package')
def still_durations():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['durations'][()]
    return res


@fixture(scope='package')
def still_max_acc():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['max_acc'][()]
    return res


@fixture(scope='package')
def still_min_acc():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['min_acc'][()]
    return res


@fixture(scope='package')
def still_sparc():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['sparc'][()]
    return res


# -------------------------------------------------------------------------------------------------
#                               DISPLACEMENT DETECTION DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='module')
def disp_times():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['times'][()]
    return res


@fixture(scope='module')
def disp_durations():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['durations'][()]
    return res


@fixture(scope='module')
def disp_max_acc():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['max_acc'][()]
    return res


@fixture(scope='module')
def disp_min_acc():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['min_acc'][()]
    return res


@fixture(scope='module')
def disp_sparc():
    with resources.path('sit2standpy.data', 'detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['sparc'][()]
    return res

