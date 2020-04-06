from pytest import fixture
import h5py
import tempfile
from importlib import resources


# RAW DATA
# --------
@fixture(scope='package')
def sample_h5():
    tf = tempfile.TemporaryFile()
    data = h5py.File(tf, 'a')
    with resources.path('sit2standpy.data', 'sample.h5') as path:
        with h5py.File(path, 'a') as f:
            data['Sensors/Lumbar/Accelerometer'] = f['/Sensors/Lumbar/Accelerometer'][()]
            data['Sensors/Lumbar/Unix Time'] = f['/Sensors/Lumbar/Unix Time'][()]
    data.close()
    return tf


@fixture(scope='package')
def sample_dict():
    data = {'Sensors': {'Lumbar': {}}}
    with resources.path('sit2standpy.data', 'sample.h5') as path:
        with h5py.File(path, 'r') as f:
            data['Sensors']['Lumbar']['Accelerometer'] = f['/Sensors/Lumbar/Accelerometer'][()]
            data['Sensors']['Lumbar']['Unix Time'] = f['/Sensors/Lumbar/Unix Time'][()]

    return data

