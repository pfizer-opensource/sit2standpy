# pysit2stand
Python based framework for detecting sit-to-stand transitions using acceleration from a lumbar mounted sensor.


## Requirements

- Python >= 3.7
- Numpy >= 1.16.2
- Scipy >= 1.2.1
- pywavelets >= 1.0.3

These are the versions developed on, and some backwards compatability may be possible.

## Installation

Run in the command line/terminal:

`pip install pysit2stand`


## Testing

Automated tests can be run with ``pytest`` through the terminal:

```shell script
pytest --pyargs pysit2stand.tests
```

## Usage

Basic use is accomplished through the ``AutoSit2Stand`` object:

```python
import pysit2stand as s2s
import numpy as np  # importing sample data
from importlib import resources

# locate the sample data and load it
with resources.path('pysit2stand.data', 'sample.csv') as file_path:
    data = np.loadtxt(file_path, delimiter=',')

# separate the stored sample data
time = data[:, 0]
accel = data[:, 1:]

# initialize the framework for detection
asts = s2s.AutoSit2Stand(accel, time, time_units='us', window=False, hours=('08:00', '20:00'), parallel=False, 
                         parallel_cpu='max', continuous_wavelet='gaus1', peak_pwr_band=[0.0, 0.5], 
                         peak_pwr_par=None, std_height=True, verbose=True)

# run the sit-to-stand detection
SiSt = asts.run(acc_filter_kwargs=None, detector='stillness', detector_kwargs=None)

# print the list of Transition objects, stored as a dictionary with the time they occurred
print(SiSt)
```

`sit_to_stands` is then a dictionary of `Transition` objects containing information about each of the transitions 
detected



