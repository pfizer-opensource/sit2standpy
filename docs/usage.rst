.. pysit2stand usage

=======================================
Usage examples
=======================================

Basic Use
---------

Basic usage of ``PySit2Stand`` to detect transitions in sample data:

.. code-block:: python

  >>> import pysit2stand as s2s
  >>> import numpy as np  # importing sample data
  >>> if version_info < (3, 7):
  >>>   from pkg_resources import resource_filename
  >>> else:
  >>>   from importlib import resources

  >>> # locate the sample data and load it (depending on python version)
  >>> if version_info < (3, 7):
  >>>   file_path = resource_filename('pysit2stand', 'data/sample.csv')
  >>>   data = np.loadtxt(file_path, delimiter=',')
  >>> else:
  >>>   with resources.path('pysit2stand.data', 'sample.csv') as file_path:
  >>>     data = np.loadtxt(file_path, delimiter=',')

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

Advanced Examples
-----------------

Using ``pysit2stand.AutoSit2Stand`` allows for automatic segmentation of days based on hours chosen to be used. Parallel processing is also available

.. code-block:: python

  >>> import pysit2stand as s2s
  >>> from packages_for_importing_data import your_import_data_function

  >>> # due to the size of multi day files, no samples are provided with pysit2stand
  >>> accel, time = your_import_data_function()

  >>> # setup the automated framework to window the data automatically
  >>> hours = ('07:00', '21:00')  # use a 7:00AM to 9:00PM window for each day
  >>> n_parallel = 'max'  # number of CPUs to use. 'max' will use all available (calculated automatically)
  >>> # leave the rest of the arguments the same
  >>> ptd = s2s.AutoSit2Stand(accel, time, time_units='us', window=True, hours=hours,
  >>>                         parallel=True, parallel_cpu=n_parallel)

  >>> # leave acceleration filter parameters the same, but adjust detector parameters
  >>> threshs = {'stand displacement': 0.125,
  >>>            'transition velocity': 0.2,
  >>>            'accel moving avg': 0.2,
  >>>            'accel moving std': 0.1,
  >>>            'jerk moving avg': 2.5,
  >>>            'jerk moving std': 3.0}  # default values
  >>> det_kw = {'gravity': 9.81,  # adjust based on local gravity, as this can have a large effect on the results
  >>>           'thresholds': threshs,
  >>>           'gravity_pass_ord': 6,  #  up from 4
  >>>           'gravity_pass_cut': 0.6,  #  down from 0.8
  >>>           'long_still': 1.0,  # up from 0.5s
  >>>           'moving_window': 0.3,
  >>>           'duration_factor': 10,
  >>>           'displacement_factor': 0.5,  # down from 0.75
  >>>           'lmax_kwargs': None,
  >>>           'lmin_kwargs': dict(height=-9.5)  # add a height constraint
  >>>           }

  >>> # run the sit-to-stand detection
  >>> SiSt = ptd.run(acc_filter_kwargs=None, detector='stillness', detector_kwargs=det_kw)
