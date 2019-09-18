.. pysit2stand usage

=======================================
Usage examples
=======================================

Basic usage of ``PySit2Stand`` to detect transitions in sample data:

.. code-block:: python

  >>> import pysit2stand as s2s
  >>> import numpy as np  # importing sample data
  >>> from importlib import resources

  >>> # locate the sample data and load it
  >>> with resources.path('pysit2stand.data', 'sample.csv') as file_path:
  >>>     data = np.loadtxt(file_path, delimiter=',')

  >>> # separate the stored sample data
  >>> time = data[:, 0]
  >>> accel = data[:, 1:]

  >>> # initialize the framework for detection
  >>> asts = s2s.AutoSit2Stand(accel, time, time_units='us', window=False)

  >>> # run the sit-to-stand detection
  >>> SiSt = asts.run(acc_filter_kwargs=None, detector='stillness', detector_kwargs=None)

  >>> # print the list of Transition objects, stored as a dictionary with the time they occurred
  >>> print(SiSt)
  {'2019-09-09 18:38:14.283405': Sit to Stand (Duration: 1.41),
   '2019-09-09 18:38:27.641925': Sit to Stand (Duration: 1.64),
   '2019-09-09 18:39:12.678105': Sit to Stand (Duration: 1.68)}


Alternatively, frameworks can be defined and passed as arguments to the underlying models:

.. code-block:: python

  >>> from pandas import to_datetime
  >>> from numpy import mean, diff

  >>> timestamps = to_datetime(time, unit='us')
  >>> dt = mean(diff(time)) / 1e6  # Sampling time, convert from microseconds to seconds

  >>> acc_filter = s2s.AccFilter()
  >>> disp_detect = s2s.detectors.Displacement()
  >>> ptd = s2s.Sit2Stand()  # Postural Transition Detector

  >>> SiSt = ptd.fit(accel, timestamps, disp_detect, acc_filter, fs=1/dt)
  >>> print(SiSt)
  {'2019-09-09 18:38:14.283405': Sit to Stand (Duration: 1.41),
   '2019-09-09 18:38:27.641925': Sit to Stand (Duration: 1.64),
   '2019-09-09 18:39:12.678105': Sit to Stand (Duration: 1.68),
   '2019-09-09 18:39:16.068513': Sit to Stand (Duration: 3.30),
   '2019-09-09 18:39:20.232309': Sit to Stand (Duration: 1.81)}
