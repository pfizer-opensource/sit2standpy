# Sit2StandPy

[![DOI](https://joss.theoj.org/papers/10.21105/joss.02449/status.svg)](https://doi.org/10.21105/joss.02449)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3988351.svg)](https://doi.org/10.5281/zenodo.3988351)

``Sit2StandPy`` is an open source Python package that uses novel algorithms to first detect Sit-to-Stand transitions 
from lumbar-mounted accelerometer data, and then provide quantitative metrics assessing the performance of the 
transitions. A modular framework is employed that would allow for easy modification of parts of the algorithm to suit 
other specific requirements, while still keeping core elements of the algorithm intact. As gyroscopes impose a 
significant detriment to battery life due to power consumption, ``Sit2StandPy``'s use of acceleration only allows for
a single sensor to collect days worth of analyzable data.

## Documentation

[Full documentation](https://sit2standpy.readthedocs.io/en/latest/) is available, containing API references, 
installation instructions, and usage examples.


## Requirements

- Python >= 3.7
- Numpy
- pandas
- Scipy 
- pywavelets
- udatetime

To run the tests, additionally the following are needed

- pytest
- h5py

## Installation

Run in the command line/terminal:

```shell script
pip install sit2standpy
```

pip will automatically collect and install the required packages by default. If you do not want this behavior, run

```shell script
pip install sit2standpy --no-deps
```


## Testing

Automated tests can be run with ``pytest`` through the terminal:

```shell script
pytest --pyargs sit2standpy.tests -v
```

To run the v2 interface tests:
```shell script
pytest --pyargs sit2standpy.v2.tests -v
```

## V2 Interface

Starting with version 1.1.0 a new "v2" interface is available alongside the old interface. Following a sequential
pipeline layout, a basic usage example is:

```python
import sit2standpy as s2s

# transform the data into the appropriate format for H5 or dictionary
# note that "data_transform_function" is your own function to achieve the appropriate format
# if you are looking for a quick example data loader function, you can use the one at
# https://gist.github.com/LukasAdamowicz/b8481ef32e4beeb77c80f29f34c8045e
data = <data_transform/loader_function>(acceleration_data)

sequence = s2s.v2.Sequential()
sequence.add(s2s.v2.WindowDays(hours=[8, 20]))  # window the data into days using only the hours from 8:00 to 20:00
sequence.add(s2s.v2.AccelerationFilter())  # Do the initial filtering and processing required
sequence.add(s2s.v2.Detector(stillness_constraint=True))  # Detect the transitions using the stillness constraint

sequence.predict(data)  # predict and save the results into data

s2s.v2.tabulate_results(data, path_to_csv_output, method='stillness')  # tabulate the results to a csv for easy reading
```


## Old Usage

Basic use is accomplished through the ``Sit2Stand`` object:

```python
import sit2standpy as s2s
import numpy as np  # importing sample data
from sys import version_info
if version_info < (3, 7):
    from pkg_resources import resource_filename
else:
    from importlib import resources

# locate the sample data and load it (depending on python version)
if version_info < (3, 7):
    file_path = resource_filename('sit2standpy', 'data/sample.csv')
    data = np.loadtxt(file_path, delimiter=',')
else:
    with resources.path('sit2standpy.data', 'sample.csv') as file_path:
        data = np.loadtxt(file_path, delimiter=',')

# separate the stored sample data
time = data[:, 0]
accel = data[:, 1:]

# initialize the framework for detection
ths = {'stand displacement': 0.125, 'transition velocity': 0.3, 'accel moving avg': 0.15,
                   'accel moving std': 0.1, 'jerk moving avg': 2.5, 'jerk moving std': 3}
sts = s2s.Sit2Stand(method='stillness', gravity=9.84, thresholds=ths, long_still=0.3, still_window=0.3,
                    duration_factor=4, displacement_factor=0.6, lmin_kwargs={'height': -9.5}, power_band=[0, 0.5],
                    power_peak_kwargs={'distance': 128}, power_stdev_height=True)

# run the sit-to-stand detection
SiSt = sts.apply(accel, time, time_units='us')

# print the list of Transition objects, stored as a dictionary with the time they occurred
print(SiSt)
```

`sit_to_stands` is then a dictionary of `Transition` objects containing information about each of the transitions 
detected


## Contributing

Contributions are welcome.  Please see the [contributions](https://github.com/PfizerRD/sit2standpy/blob/master/CONTRIBUTING.md) document for more information


