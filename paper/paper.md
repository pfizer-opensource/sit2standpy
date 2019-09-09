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
 - name: Pfizer, Inc.
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
only one sensor to collect days worth of analyzable data.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References