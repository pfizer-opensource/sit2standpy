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
  - Human Motion
  - Activity Recognition
authors:
  - name: Lukas Adamowicz
    affiliation: 1
affiliations:
 - name: Pfizer, Inc. 610 Main Street, Cambridge MA, USA 02139
   index: 1
date: 4 September, 2019
bibliography: paper.bib
---

# Summary

Digital medicine is driven by novel algorithms that analyze signals to drive digital biomarker development. At-home continuous monitoring has been greatly enhanced by the use of wearable sensors, which provide raw inertial data about human movement. Sit-to-stand transitions are particularly important due to their long-time clinical use in assessing
disease and disorder states, such as Parkinson's Disease [@buckley:2008] and knee Osteoarthritis [@bolink:2012], and the strength requirements for populations with movement limitations. While most previous works have focuses on either in-clinic applications or the use of multiple sensors, this algorithm works at-home, as well as in-clinic, and uses only the acceleration data from a single sensor with unconstrained orientation. The practicality of a single-accelerometer approach, coupled with the up to three times as long recording time over a full inertial measurement unit (accelerometer and gyroscope) and the unconstrained orientation, promote this this approach for long-term, continuous, at-home monitoring.

``PySit2Stand`` is an open source Python package that uses novel heuristics-based single-accelerometer algorithms to first detect Sit-to-Stand transitions from lumbar-mounted accelerometer data, and then provides quantitative metrics, including duration, maximum and minimum acceleration, and SPARC [@balasubramanian:2015], assessing the performance of the transitions. At its simplest, ``PySit2Stand`` takes raw accelerometer data with timestamps and returns detected sit-to-stand transitions. Data can be windowed by full days, or parts of days can be selected for each window (e.g. window from 8:00 to 20:00). Additionally, ``PySit2Stand`` can take advantage of multiple core CPUs with a parallel processing option, which provides run-time benefits both in the initial processing stages and in detecting the transitions.

Under this simple interface, there are several points of customization, which may aid in transition detection under specific conditions. Users maintain control, if desired, over filtering and initial pre-processing parameters and most detection parameters. Additionally, there are two options for the level of adherence to a requirement that stillness precede a valid transition, allowing better performance during clinic assessments or in home environments. A modular framework is employed that would allow for easy modification of parts of the algorithm to suit other specific requirements (such as adjusting the metrics that are extracted) while still keeping core elements of the algorithm intact.


# Current Work

``PySit2Stand`` was validated, and used to compute the results presented in [@adamowicz:2019]. Detailed background and
presentation of the methods can be found there as well.

# Availability

``PySit2Stand`` is distributed under the MIT License and is published on PyPI, the Python Package Index, and can be installed by running the following in the terminal:

```shell-script
pip install pysit2stand  # install with checking for dependencies
# or
pip install pysit2stand --no-deps  # installation without checking for installed dependencies
```

PySit2Stand requires the following Python packages:

- Python >=3.6
- NumPy - [@numpy]
- SciPy - [@scipy]
- PyWavelets - [@pywavelets]

``PySit2Stand`` contains example code with sample data in its GitHub repository. Full documentation with usage examples, installation instructions, and API reference are all available at [https://pysit2stand.readthedocs.io/en/latest/](https://pysit2stand.readthedocs.io/en/latest/)


# References
