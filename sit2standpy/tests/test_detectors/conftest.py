from pytest import fixture
from numpy import sin, cos, arange, pi, zeros
import h5py
from importlib import resources


@fixture(scope='package')
def integrate_data():
    dt = 0.01
    x = arange(0, 2 * pi, dt)
    f = cos(x)

    F = sin(x)
    FF = -cos(x) + 1  # need to add one due to intial value of the integration being 0

    return f, F, FF, dt


@fixture(scope='package')
def stillness():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            still = file['stillness'][()]

    return still


@fixture(scope='package')
def still_starts():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            starts = file['starts'][()]

    return starts


@fixture(scope='package')
def still_stops():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            stops = file['stops'][()]

    return stops


# -------------------------------------------------------------------------------------------------
#                               STILLNESS DETECTION DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='package')
def still_times():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['times'][()]
    return res


@fixture(scope='package')
def still_durations():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['durations'][()]
    return res


@fixture(scope='package')
def still_max_acc():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['max_acc'][()]
    return res


@fixture(scope='package')
def still_min_acc():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['min_acc'][()]
    return res


@fixture(scope='package')
def still_sparc():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Stillness']['sparc'][()]
    return res


# -------------------------------------------------------------------------------------------------
#                               DISPLACEMENT DETECTION DATA
# -------------------------------------------------------------------------------------------------
@fixture(scope='module')
def disp_times():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['times'][()]
    return res


@fixture(scope='module')
def disp_durations():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['durations'][()]
    return res


@fixture(scope='module')
def disp_max_acc():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['max_acc'][()]
    return res


@fixture(scope='module')
def disp_min_acc():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['min_acc'][()]
    return res


@fixture(scope='module')
def disp_sparc():
    with resources.path('sit2standpy.data', '.detector_results.h5') as file_name:
        with h5py.File(file_name, 'r') as file:
            res = file['Displacement']['sparc'][()]
    return res
