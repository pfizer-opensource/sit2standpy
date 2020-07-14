.. Sit2StandPy documentation master file, created by
   sphinx-quickstart on Wed Jul 17 16:13:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sit2StandPy: Automated Sit-to-Stand Transition Detection
========================================================

``Sit2StandPy`` is an open source Python package that uses novel algorithms to first detect Sit-to-Stand transitions
from lumbar-mounted accelerometer data, and then provide quantitative metrics assessing the performance of the
transitions. A modular framework is employed that would allow for easy modification of parts of the algorithm to suit
other specific requirements, while still keeping core elements of the algorithm intact. As gyroscopes impose a
significant detriment to battery life due to power consumption, ``Sit2StandPy``'s use of acceleration only allows for
a single sensor to collect days worth of analyzable data.


Validation
----------
``Sit2StandPy`` was validated, and used to compute the results presented in [1]_. Detailed background and
presentation of the methods can be found there as well.

Capabilities
------------

- Automatic day by day windowing, with option to select a specific window of hours per day
- Optional use of parallel processing to decrease run time
- Quantification of detected transitions
- Modification of functions and methods used in framework
- Access to various parameters controlling the performance and function of the detection

License
-------
Sit2StandPy is open source software distributed under the MIT license.

Papers
------
.. [1] L. Adamowicz et al. "Assessment of Sit-to-Stand Transfers During Daily Life Using an Accelerometer on the Lower Back.'' IEEE Journal of Biomedical and Health Informatics. Under Review.
.. [2] L. Adamowicz, S. Patel. "Sit2StandPy: An Open-Source Python Package for Detecting and Quantifying Sit-to-Stand Transitions Using an Accelerometer on the Lower Back.'' Journal of Open Source Software. Under Review.


Contents
--------
.. toctree::
   :maxdepth: 3

   installation
   usage
   refv2/index
   ref/index
