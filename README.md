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


## Usage

pysit2stand works with one central framework, to which several sub frameworks are passed, which do the bulk of the 
computing and allow for customization if desired.

```python
import pysit2stand as s2s
import h5py  # for importing demo data

acceleration_filter = s2s.AccFilter()
stillness_detector = s2s.detectors.Stillness()

sts_detector = s2s.Sit2Stand()

acceleration = import_acceleration_data()
timestamps = import_timestamps()

sit_to_stands = sts_detector.fit(acceleration, timestamps, stillness_detector, acceleration_filter, fs=100)
```

`sit_to_stands` is then a dictionary of `Transition` objects containing information about each of the transitions 
detected



