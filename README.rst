#######################################
Calibration Products Production Package
#######################################

Craig Lage - 14-Feb-2019

This is code that I have forked from lsst/cp_pipe and am modifying to improve the corrections of the Brighter-Fatter effect. I have added a number of options to python/lsst/cp/pipe/makeBrighterFatterKernel.py. There are comments in the code that describe these, but it is work in progress and I have not yet documented them in detail.

In the notebooks directory are a couple of notebooks for examining the output of the code.

In the stand_alone directory are several stand-alone Python scripts.  These ingest a set of flats, calculate the BF kernel, then ingest a set of spots at different intensities and plot the spot size vs flux with and without correction.

There are currrently severla versions of the makeBrighterFateerKernel code, as follows:

(1) makeBrighterFatterKernel_base_two_amps.py - This files uses the baseline functionality, but I have made several changes for debug p
urposes.  I added code to save the measured correlations and mean fluxes.  I also have restricted it to just two amps, C04 and C14, to 
speed debug.

(2) makeBrighterFatterKernel.py - This file has all of the options that I have added.  I have also made several changes for debug purpo
ses.  I added code to save the measured correlations and mean fluxes.  I also have restricted it to just two amps, C04 and C14, to spee
d debug.

(3) makeBrighterFatterKernel_all_amps.py - This file is the same as (2), but without the restriction to two amps.
