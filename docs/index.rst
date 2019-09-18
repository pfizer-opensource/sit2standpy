.. pysit2stand documentation master file, created by
   sphinx-quickstart on Wed Jul 17 16:13:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PySit2Stand: Automated Sit-to-Stand Transition Detection
========================================================

``PySit2Stand`` is an open source Python package that uses novel algorithms to first detect Sit-to-Stand transitions
from lumbar-mounted accelerometer data, and then provide quantitative metrics assessing the performance of the
transitions. A modular framework is employed that would allow for easy modification of parts of the algorithm to suit
other specific requirements, while still keeping core elements of the algorithm intact. As gyroscopes impose a
significant detriment to battery life due to power consumption, ``PySit2Stand``'s use of acceleration only allows for
a single sensor to collect days worth of analyzable data.


Validation
----------
``PySit2Stand`` was validated, and used to compute the results presented in [1]_. Detailed background and
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
PySit2Stand is open source software distributed under the MIT license.

Papers
------
Citations for the appropriate of the following two papers:

.. [1] L. Adamowicz, I. Karahonoglu, S. Patel. ``Sit-to-Stand Detection Using Only Lumbar Acceleration: Clinical and Home Application.'' In-preparation.
.. [2] L. Adamowicz. ``PySit2Stand: Python package for Sit-to-Stand transition detection and quantification.'' Journal of Open Source Software. In-preparation.


Contents
--------
.. toctree::
   :maxdepth: 2

   installation.rst
   ref/index.rst
   usage.rst
