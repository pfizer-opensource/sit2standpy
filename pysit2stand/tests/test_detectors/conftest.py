from pytest import fixture
from numpy import sin, cos, arange, pi, zeros
import h5py


@fixture
def integrate_data():
    dt = 0.01
    x = arange(0, 2 * pi, dt)
    f = cos(x)

    F = sin(x)
    FF = -cos(x) + 1  # need to add one due to intial value of the integration being 0

    return f, F, FF, dt


@fixture
def stillness():
    file_name = '/Users/LukasAdamowicz/Documents/Python Packages/Sit to Stand/pysit2stand/pysit2stand/data/' \
                '.detector_results.h5'
    with h5py.File(file_name, 'r') as file:
        still = file['stillness'][()]

    return still


@fixture
def still_starts():
    file_name = '/Users/LukasAdamowicz/Documents/Python Packages/Sit to Stand/pysit2stand/pysit2stand/data/' \
                '.detector_results.h5'
    with h5py.File(file_name, 'r') as file:
        starts = file['starts'][()]

    return starts


@fixture
def still_stops():
    file_name = '/Users/LukasAdamowicz/Documents/Python Packages/Sit to Stand/pysit2stand/pysit2stand/data/' \
                '.detector_results.h5'
    with h5py.File(file_name, 'r') as file:
        stops = file['starts'][()]

    return stops
