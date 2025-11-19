# NEID Exposure Time Calculator

This exposure time calculator is a web app that uses Flask to interface with a Python environment. The code here powers the online NEID exposure time calculator [here](http://neid-etc.tuc.noao.edu/calc_shell/calculate_rv).

# Python implementation

Users interesting in performing calculations beyond the scope of the simple Web interface calculator may find it useful to interface directly with the python wrapper, `neid_etcalc_public.py`, which contains all functions necessary for exposure time calculations. The exposure time grids upon which these functions rely are contained in the various FITS files in this repository.

# Description

See [the about file](templates/calc_shell/about.html) for a description of the calculator.

# Questions

Please contact Jason Wright <astrowright@gmail.com>.
