.. py:currentmodule:: lsst.cp.pipe

#########################
Constructing calibrations
#########################

`cp_pipe` now supports the production of all of the calibration types for the gen-three middleware.  This page presents example commands that can be used to create calibrations for the LATISS instrument.  The operation should be similar for other cameras, and potential differences are described below.

In general, calibration construction is a four step process.  First, a set of exposures is identified to be used in the calibration.  Then, a pipeline is run to generate a proposed calibration from that set of raw exposures.  Next, the calibration should be checked by the tests in the ``cp_verify`` package, to confirm that it is suitable for the entire validity range, and that it satisfies all of the quality metrics defined in DMTN-101.  Finally, the calibration can be certified for use, with a timespan indicating when the calibration is valid.  Individual CALIBRATION collections can be grouped into a butler CHAINED collection to connect all of the various calibrations into a single collection, avoiding the need to remember a large sequence of collection names.  As ``cp_verify`` is still in development, not every example has an associated verification step.

The examples presented below follow the collection naming conventions listed in DMTN-167, and will use ticketed user collection names.

.. _cp-pipe-example-butler

Butler requirements
===================

A gen-three butler is needed to run the ``pipetask`` commands defined by this package.  The examples below use one located in ``/repo/main``, matching the shared repository on the NCSA development cluster.  Information about constructing an independent butler repository can be found in the `lsst.daf.butler` documentation.

.. _cp-pipe-certification

Calibration certification
=========================

Certification of calibrations should be done after confirming that all of the ``cp_verify`` tests have passed, and any tests that still fail should be documented on the ticket.  Calibrations created as part of daily observations should be certified into a single collection per calibration type, with the validity range set to only include the day they were taken, such as::

    butler certify-calibrations /repo/main \
        u/czw/DM-XYZ/biasGen.20210715 LATISS/calib/dailyBiases \
        --begin_date 2021-07-15 --end_date 2021-07-16 bias

Longer term master calibrations should be certified into ticketed CALIBRATION collection (containing however many calibration types and date ranges that entails), that should then be chained to the instrument's recommended CALIBRATION collection.  The results of ``cp_verify`` should provide some guidance on the validity range, as additional raw exposures can be checked against the proposed calibration to identify when the verifcation tests fail.  The end date may be best set to a future date that will allow the calibration to be used until a new one supersedes it::

    butler certify-calibrations /repo/main \
        u/czw/DM-XYZ/biasGen.20210715 LATISS/calib/DM-XYZ \
        --begin_date 2021-01-01 --end_date 2050-01-01 bias

.. _cp-pipe-readNoise:

Read Noise
==========

Calibration construction and verification are sensitive to the read noise value listed in the ``camera`` camera geometry definition.  Inaccurate values may trigger test failures that are spurious.  Setting the ``isr:doEmpiricalReadNoise=True`` option during the bias processing (as the bias generally has very little signal other than noise) may be necessary to bootstrap a full set of calibrations from scratch.  This option records the values measured in the log, and by analysing the results of many exposures, better estimates of the read noise can be generated.

.. _cp-pipe-biases:

Constructing biases
===================

- Identify a set of exposures to use as inputs from the repository::

    butler query-dimension-records /repo/main exposure \
        --where "instrument='LATISS' AND exposure.observation_type='bias' \
                 AND exposure.target_name='Park position' \
                 AND exposure.exposure_time=0.0 AND exposure.dark_time < 0.1 \
                 AND exposure.day_obs > 20210101"

  - This returns a large number of potential exposures, with some dates dominating the counts.  An initial semi-random sample of 50 exposures was used as input for the master bias.  These exposures were selected to attempt to have the widest possible date coverage, as well as preventing any one date from having a majority of the exposures::

    EXPOSURES='2021012000019, 2021012000020, 2021012000032, 2021012000055, 2021012000061, \
               2021012100060, 2021012100079, 2021012100134, 2021012100177, 2021012100188, \
               2021012100229, 2021012100273, 2021012100303, 2021012700032, 2021012700037, \
               2021012700038, 2021012700052, 2021012700119, 2021012700842, 2021012700900, \
               2021012700926, 2021020100022, 2021020100032, 2021020100036, 2021020100047, \
               2021020100049, 2021020100335, 2021020100344, 2021020100369, 2021030500001, \
               2021030500009, 2021030500015, 2021030500019, 2021030500023, 2021030500032, \
               2021030500046, 2021031100028, 2021031100032, 2021031100036, 2021031100037, \
               2021031100041, 2021031100045, 2021031100048, 2021060900011, 2021060900026, \
               2021060900038, 2021060900039, 2021060900042, 2021060900048, 2021060900049'

  - This sample was later cleaned and supplemented with additional exposures after running into failures during verification, as the lack of a set of defects meant that the cosmic ray rejection in ``cp_verify`` would raise due to triggering on the unmasked defect pixels.  The final sample used was::

    EXPOSURES='2021012000020, 2021012000032, 2021012000055, 2021012000061, 2021012100060, \
               2021012100134, 2021012100188, 2021012100229, 2021012700032, 2021012700037, \
               2021012700038, 2021012700052, 2021012700119, 2021012700842, 2021012700900, \
               2021012700926, 2021020100022, 2021020100032, 2021020100036, 2021020100047, \
               2021020100049, 2021020100335, 2021020100344, 2021020100369, 2021030500009, \
               2021030500015, 2021030500019, 2021030500023, 2021030500032, 2021030500046, \
               2021031100028, 2021031100032, 2021031100036, 2021031100037, 2021031100041, \
               2021031100045, 2021031100048, 2021060900011, 2021060900026, 2021060900038, \
               2021060900039, 2021060900042, 2021060900048, 2021060900049, 2021012000037, \
               2021012000059, 2021012000063, 2021012100078, 2021012700061, 2021012700423, \
               2021012700701, 2021020100072, 2021020100329, 2021020100375, 2021030500005, \
               2021030500026, 2021030500050, 2021031100004, 2021031100005, 2021031100010'

- Run the bias pipeline on these exposures.  This pipeline is simple, with a short ISR step that only applies overscan correction and assembles the exposures, before passing them to a combine step that finds the clipped per-pixel mean for the output bias.  Only the raw and curated calibration collections are needed as inputs::

    RERUN=20210702a
    pipetask --long-log run -b /repo/main -p $CP_PIPE_DIR/pipelines/Latiss/cpBias.yaml \
         -i LATISS/raw/all,LATISS/calib -o u/czw/DM-28920/biasGen.$RERUN \
         -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES) \
         -c isr:doDefect=False -c isr:doEmpiricalReadNoise=True >& ./bias.$RERUN.log

  - Passing the ``--long-log`` and saving the output to a logfile are recommended, as it is easier to debug issues with that information.
  - No good defect set exists, so the ``-c isr:doDefect=False`` option was disabled.  This should only be necessary when starting calibrations from scratch.
  - As discussed above, the nominal read noise values are incorrect (especially for amplifier ``C07``), and so the ``-c isr:doEmpiricalReadNoise=True`` was enabled to prevent this amplifier from being thrown out.

- Validate the input exposures with ``cp_verify``.  Additional exposures could be validated to firmly establish a date range that this bias should be used::

    pipetask run -b /repo/main -p $CP_VERIFY_DIR/pipelines/Latiss/verifyBias.yaml \
         -i u/czw/DM-28920/biasGen.$RERUN,LATISS/raw/all,LATISS/calib \
         -o u/czw/DM-28920/verifyBias.$RERUN \
          -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)

  - This pipeline produces statistics and test results for every ``{exposure, detector}`` pair in the input data, and then collates that data to produce per-exposure summaries (and optionally addition exposure-level statistics and tests), and finally into one final per-run summary.
  - As part of DM-28920, ``cp_verify`` is being augmented with a series of Jupyter notebooks that are designed to help visualize and process the potentially very large amount of information.  Running the ``$CP_VERIFY_DIR/examples/cpVerifyBias.ipynb`` will show the final generated bias, allow each residual image to be examined along with the statistic and test results, as well as provide histograms of number of failed tests.  Further discussion of these notebooks will be available in DMTN-192 and in the ``cp_verify`` documentation.

- Upon confirming that the calibration has passed all of the verification tests (or that the failed tests are permanent/uncorrectable), the calibraion is now ready to be certified to final collection::

    butler certify-calibrations /repo/main u/czw/DM-28920/biasGen LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 bias

.. _cp-pipe-defects:

Constructing defects
====================

- As the majority of the tests failed during the bias verification were on amplifiers that had obvious defects, constructing a new list of defects is a priority.  The fact that the defects were obvious makes the input exposure selection easy: we can simply reuse the list of exposures used to construct the bias.
- Followed by running the defect pipeline::

    RERUN=20210706h
    pipetask --long-log run -b /repo/main -p $CP_PIPE_DIR/pipelines/Latiss/findDefects.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/defectGen.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)"  >& ./defect.$RERUN.log

  - For this test, certification was delayed until the entire chain of calibrations had been generated and verified.  This illustrates the fact that the butler can access calibrations from the RUN collection that they were generated in, that no other types of that calibration are found in a collection that is searched earlier.
- Verification of the defects::

    pipetask --long-log run -b /repo/main -p $CP_VERIFY_DIR/pipelines/verifyDefect.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.$RERUN,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/verifyDefect.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" >& ./defectVerify.$RERUN.log

  - By placing the ``u/czw/DM-28920/defectGen.20210706h`` collection before the ``LATISS/calib`` collection, we can use the defects just created, and not the ingested defects that mask the entirety of amplifier ``C07``.
  - As before, there will be a ``$CP_VERIFY_DIR/examples/cpVerifyDefects.ipynb`` containing the visualization and test failure information.
- If the validation tests pass, the new defects can be certified::

    butler certify-calibrations /repo/main u/czw/DM-28920/defectGen.20210706h LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 defects

.. _cp-pipe-darks:

Constructing darks
==================

- Identify the inputs:

  - ``butler query-dimension-records /repo/main exposure --where "instrument='LATISS' AND exposure.observation_type='dark' AND exposure.exposure_time > 0.0 AND exposure.dark_time > 0.0 AND exposure.day_obs > 20210101"``
  - Select 100 of these exposures to match the suggestion in DMTN-101.

    - ``EXPOSURES='2021011900151, 2021011900152, 2021011900153, 2021011900154, 2021011900155, 2021011900156, 2021011900157, 2021011900158, 2021011900159, 2021011900160, 2021012100668, 2021012100670, 2021012100671, 2021012100672, 2021012100673, 2021012100674, 2021012100676, 2021012100677, 2021012100685, 2021012600022, 2021012600024, 2021012600026, 2021012600028, 2021012600029, 2021012600051, 2021012600052, 2021012600057, 2021012600060, 2021021700076, 2021021700077, 2021021700078, 2021021700081, 2021021700082, 2021021700084, 2021021700085, 2021021800058, 2021021800060, 2021021800061, 2021021800063, 2021021800065, 2021021800066, 2021030300006, 2021030300007, 2021030300010, 2021030300018, 2021030300028, 2021030300032, 2021030300048, 2021030300054, 2021030300057, 2021030300058, 2021030300066, 2021030800001, 2021030800003, 2021030800004, 2021030800005, 2021030800006, 2021030800007, 2021030900052, 2021030900053, 2021030900054, 2021030900058, 2021030900060, 2021030900061, 2021031000052, 2021031000053, 2021031000054, 2021031000055, 2021031000056, 2021031000057, 2021031000058, 2021031000061, 2021031100052, 2021031100054, 2021031100055, 2021031100056, 2021031100057, 2021031100058, 2021031100059, 2021031100060, 2021032200013, 2021032200014, 2021032200016, 2021032200021, 2021032200024, 2021032200028, 2021032200029, 2021032200032, 2021032200034, 2021032300024, 2021032300032, 2021032300049, 2021032300064, 2021032300087, 2021032300115, 2021032300125, 2021032300126, 2021032300136, 2021032300149, 2021032300167'``

- Run the dark pipeline on these exposures.  The ISR step here applies the bias in addition to the overscan and assembly, cosmic rays are rejected, the images are scaled by the ``dark_time``, and the clipped per-pixel mean is written to the output bias.  The previously generated bias CALIBRATION collection is also needed now.

  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/cpDark.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/darkGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``

- Certify for verification.

  - ``butler certify-calibrations /repo/main u/czw/DM-28920/darkGen u/czw/DM-28920/darkTemp --begin-date 1980-01-01 --end-date 2050-01-01 dark``

- Verify:

  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyDark.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/darkTemp -o u/czw/DM-28920/verifyDark -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``

- Certify to final collection.

  - ``butler certify-calibrations /repo/main u/czw/DM-28920/darkGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 dark``

.. _cp-pipe-flats:

Constructing flats
==================

- Identify the inputs:

  - ``butler query-dimension-records /repo/main exposure --where "instrument='LATISS' AND exposure.observation_type='flat' AND exposure.exposure_time > 0.0 AND exposure.day_obs > 20210101"``
  - This needs to be split into two groups, as we have two filters, ``empty~empty`` and ``RG610~empty``.
  - ``EXPOSURES_empty='2021011900083, 2021011900098, 2021011900117, 2021012100565, 2021012100606, 2021012100614, 2021021600116, 2021021600117, 2021021600140, 2021021700102, 2021021700103, 2021021700128, 2021021800104, 2021021800120, 2021021800166, 2021030900077, 2021030900095, 2021030900100, 2021031000077, 2021031000088, 2021031000097, 2021031100080, 2021031100087, 2021032300251, 2021032300265'``
  - ``EXPOSURES_RG610='2021011900132, 2021011900135, 2021011900136, 2021011900139, 2021021600102, 2021021600104, 2021021600105, 2021021700088, 2021021700093, 2021021700094, 2021021800067, 2021021800070, 2021021800118, 2021021800167, 2021030900062, 2021030900063, 2021030900069, 2021031000062, 2021031000070, 2021031100064, 2021031100066, 2021031100069, 2021032300234, 2021032300240, 2021032300241'``

- Run the appropriate flat pipeline on these exposures.  Again, ISR adds dark correction, but the scaling for flats is more complicated.  LATISS is a single chip device, and so can use the `cpFlatSingleChip.yaml` pipeline definition.  This scales each input exposure by the total flux before running the clipped mean stacking.
  However, for cameras that have multiple devices, the `cpFlat.yaml` pipeline adds an additional full focal plane scaling calculation that attempts to isolate the chip-to-chip differences along with the possible exposure-to-exposure illumination differences.
  Finally, for cameras with vignetting, there is a ``doVignette`` option that needs to be set so that the vignetted region (defined by the ``VignettePolygon`` set by ``lsst.ip.isr.IsrTask`) is properly excluded from the flux calculations.

  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/cpFlat.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/flatGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``

- Certify for verification.

  - ``butler certify-calibrations /repo/main u/czw/DM-28920/flatGen u/czw/DM-28920/flatTemp --begin-date 1980-01-01 --end-date 2050-01-01 flat``

- Verify:

  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyFlat.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/flatTemp -o u/czw/DM-28920/verifyFlat -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``

- Certify to final collection.

  - ``butler certify-calibrations /repo/main u/czw/DM-28920/flatGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 flat``

.. _cp-pipe-fringes:

Constructing fringes
====================

No fringe data currently is available for LATISS, but the queries and commands would be the same, operating on science observations.

  - ``butler query-dimension-records /repo/main exposure --where "instrument='LATISS' AND exposure.observation_type='science' AND exposure.exposure_time > 0.0 AND exposure.day_obs > 20210101"``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/cpFringe.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/fringeGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/fringeGen u/czw/DM-28920/fringeTemp --begin-date 1980-01-01 --end-date 2050-01-01 fringe``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyFringe.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/fringeTemp -o u/czw/DM-28920/verifyFringe -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/fringeGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 fringe``


.. _cp-pipe-crosstalk:

Measuring the crosstalk signal
==============================

The crosstalk signal can also be measured from a sequence of science exposures that have bright stars.  A special observation sequence that tried to realize this was observed on 2021-02-18.

  - ``butler query-dimension-records /repo/main exposure --where "instrument='LATISS' AND exposure.observation_type='science' AND exposure.exposure_time > 0.0 AND exposure.target_name = 'NGC 4755' AND exposure.day_obs = 20210218"``
  - ``EXPOSURES='2021021700347, 2021021700348, 2021021700349, 2021021700350, 2021021700351, 2021021700352, 2021021700353, 2021021700354, 2021021700355, 2021021700356, 2021021700357, 2021021700358, 2021021700359'``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/measurePhotonTransferCurve.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/crosstalkGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/crosstalkGen u/czw/DM-28920/crosstalkTemp --begin-date 1980-01-01 --end-date 2050-01-01 crosstalk``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyCrosstalk.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/crosstalkTemp -o u/czw/DM-28920/verifyCrosstalk -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/crosstalkGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 crosstalk``

.. _cp-pipe-ptc:


Measuring the photon transfer curve
===================================

The PTC is generated from a sequence of paired flats, so care should be taken to ensure that a planned sequence of flats, with a ramp in exposure time (and therefore a ramp in received flux), is used as the input.  Such data was taken on 2021-03-11, so we use that.

  - ``butler query-dimension-records /repo/main exposure --where "instrument='LATISS' AND exposure.observation_type='flat' AND exposure.exposure_time > 0.0 AND exposure.day_obs = 20210311"``
  - ``EXPOSURES='2021031100072, 2021031100073, 2021031100074, 2021031100075, 2021031100076, 2021031100077, 2021031100078, 2021031100079, 2021031100080, 2021031100081, 2021031100082, 2021031100083, 2021031100084, 2021031100085, 2021031100086, 2021031100087, 2021031100088, 2021031100089, 2021031100090, 2021031100091, 2021031100092, 2021031100093, 2021031100094, 2021031100095, 2021031100096, 2021031100097, 2021031100098, 2021031100099, 2021031100100, 2021031100101, 2021031100102, 2021031100103, 2021031100104, 2021031100105, 2021031100106, 2021031100107, 2021031100108, 2021031100109, 2021031100110, 2021031100111'``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/measurePhotonTransferCurve.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/ptcGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/ptcGen u/czw/DM-28920/ptcTemp --begin-date 1980-01-01 --end-date 2050-01-01 ptc``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyPtc.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/ptcTemp -o u/czw/DM-28920/verifyPtc -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/ptcGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 ptc``

.. _cp-pipe-linearity:

Constructing a linearity correction
===================================

The linearity measurement uses the outputs measured by the photon transfer curve as its inputs.  Working from the previously generated PTC:

  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/measureLinearity.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/linearityGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/linearityGen u/czw/DM-28920/linearityTemp --begin-date 1980-01-01 --end-date 2050-01-01 linearity``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyLinearity.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/linearityTemp -o u/czw/DM-28920/verifyLinearity -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/linearityGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 linearity``

.. _cp-pipe-bfk:

Constructing a brighter-fatter correction
=========================================

The brighter-fatter kernel is also generated from the photon transfer curve, so this can also be generated from the previous calibration product.

  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/cpBfkSolve.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/bfkGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/bfkGen u/czw/DM-28920/bfkTemp --begin-date 1980-01-01 --end-date 2050-01-01 bfk``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifyBfk.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/bfkTemp -o u/czw/DM-28920/verifyBfk -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/bfkGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 bfk``

.. _cp-pipe-sky:

Constructing sky frames
=======================

Sky frames are also constructed from science exposures, and are filter dependent.  Selecting a sample of exposures from 2021-03-23:

  - ``butler query-dimension-records /repo/main exposure --where "instrument='LATISS' AND exposure.observation_type='science' AND exposure.exposure_time > 0.0 AND exposure.day_obs = 20210323 and physical_filter = 'RG610~empty'"``
  - ``EXPOSURES='2021032300284, 2021032300290, 2021032300291, 2021032300294, 2021032300297, 2021032300299, 2021032300303, 2021032300334, 2021032300341, 2021032300358, 2021032300362, 2021032300364, 2021032300365, 2021032300378, 2021032300388, 2021032300394, 2021032300414, 2021032300416, 2021032300454, 2021032300459, 2021032300470, 2021032300494, 2021032300498, 2021032300499, 2021032300522, 2021032300529, 2021032300577, 2021032300611, 2021032300615, 2021032300628'``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/cpSkySolve.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib -o u/czw/DM-28920/skyGen -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/skyGen u/czw/DM-28920/skyTemp --begin-date 1980-01-01 --end-date 2050-01-01 sky``
  - ``pipetask run -b /repo/main -p $CP_PIPE_DIR/pipelines/LATISS/verifySky.yaml -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib,u/czw/DM-28920/skyTemp -o u/czw/DM-28920/verifySky -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)``
  - ``butler certify-calibrations /repo/main u/czw/DM-28920/skyGen u/czw/DM-28920/calib --begin-date 2021-01-01 --end-date 2050-01-01 sky``

