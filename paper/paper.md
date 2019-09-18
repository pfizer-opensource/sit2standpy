---
title: 'PySit2Stand: Python package for Sit-to-Stand transition detection and quantification'
tags:
  - Python
  - Biomechanics
  - Digital Biomarkers
  - Digital Medicine
  - Accelerometer
  - Inertial Sensors
  - Wearable Sensors
  - IMU
authors:
  - name: Lukas Adamowicz
    affiliation: 1
affiliations:
 - name: Pfizer, Inc. 610 Main Street, Cambridge MA, USA 02139
   index: 1
date: 4 September, 2019
bibliography: paper.bib
---

# Background

Digital medicine is driven by novel algorithms that extract digital biomarkers. Longitudinal monitoring would 
additionally not be possible without the use of wearable sensors, which provide raw inertial data that must be 
interpreted. Sit-to-stand transitions are particularly important due to their long-time clinical use in assessing
disease and disorder states, and the strength requirements for clinical populations. Most works previously have focused 
on either in-clinic applications, or the use of multiple sensors, which is not practical for long-term at-home
monitoring.  

``PySit2Stand`` is an open source Python package that uses novel algorithms to first detect Sit-to-Stand transitions 
from lumbar-mounted accelerometer data, and then provide quantitative metrics assessing the performance of the 
transitions. A modular framework is employed that would allow for easy modification of parts of the algorithm to suit 
other specific requirements, while still keeping core elements of the algorithm intact. As gyroscopes impose a 
significant detriment to battery life due to power consumption, ``PySit2Stand``'s use of acceleration only allows for
a single sensor to collect days worth of analyzable data.

# Current Work

``PySit2Stand`` was validated, and used to compute the results presented in [@adamowicz:2019]. Detailed background and
presentation of the methods can be found there as well.

# Capabilities

At its simplest, ``PySit2Stand`` takes raw accelerometer data with timestamps and returns detected sit-to-stand 
transitions. Data can be windowed by full days, or parts of days can be selected (e.g. window from 08:00 to 20:00). 
Additionally, ``PySit2Stand`` can take advantage of multiple core CPUs with a parallel processing option, which provides 
run-time benefits both in the initial processing stages, and in detecting the transitions. 

Under this simple interface, there are several points of customization, which may aid in transition detection under
specific conditions. Users maintain control, if desired, over filtering and initial pre-processing parameters and some 
detection parameters. Additionally, there are two options for detection algorithms, offering different levels of 
strictness for the requirement that stillness precede a valid transition. 

# Use of PySit2Stand

``PySit2Stand`` contains example code with sample data in its GitHub repository. Full documentation with usage examples,
installation instructions, and API reference are all available at [https://pysit2stand.readthedocs.io/en/latest/](https://pysit2stand.readthedocs.io/en/latest/)

# Availability

``PySit2Stand`` is distributed under the MIT License and is published on PyPI, the Python Package Index.

# Dependences

PySit2Stand requires the following Python packages:

- Python >=3.6
- NumPy - [@numpy]
- SciPy - [@scipy]
- PyWavelets - [@pywavelets]



# References
