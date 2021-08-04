.. py:currentmodule:: lsst.cp.pipe

#########################
Constructing calibrations
#########################

`cp_pipe` now supports the production of all of the calibration types for the gen-three middleware.  This page presents example commands that can be used to create calibrations for the LSST Atmospheric Transmission Imager and Slitless Spectrograph (LATISS) instrument.  The operation should be similar for other cameras, and potential differences are described below.

In general, calibration construction is a four step process:

- A set of exposures is identified to be used in the calibration.
- A :ref:`pipetask <lsst.ctrl.mpexec-scripts>` pipeline is run to generate a proposed calibration from that set of raw exposures.
- The calibration should be checked by the tests in the ``cp_verify`` package, to confirm that it is suitable for the entire validity range, and that it satisfies all of the quality metrics defined in `DMTN-101 <https://dmtn-101.lsst.io/>`_.  As ``cp_verify`` is still in development, not every example has an associated verification step.
- The calibration can be certified for use, with a timespan indicating when the calibration is valid.  Individual :ref:`CALIBRATION collections <lsst.daf.butler.CollectionType.CALIBRATION>` can be grouped into a butler :ref:`CHAINED collection <lsst.daf.butler.CollectionType.CHAINED>` to connect all of the various calibrations into a :ref:`single collection <daf_butler_organizing_datasets>`, avoiding the need to remember a large sequence of collection names.

The examples presented below follow the collection naming conventions listed in `DMTN-167 <https://dmtn-167.lsst.io>`_, and will use ticketed user collection names to illustrate that the calibrations constructed here were made as part of a `JIRA request <https://jira.lsstcorp.org/browse/DM-28920>`_ to create a full set of calibrations for LATISS.

Last updated: 2021-07-30.


.. _cp-pipe-example-butler:

Butler requirements
===================

A gen-three butler is needed to run the ``pipetask`` commands defined by this package.  The examples below use ``$BUTLER_REPO`` as the location of the repository.  Information about constructing an independent butler repository can be found in the `lsst.daf.butler` documentation.


.. _cp-pipe-verification:

Verification of calibrations
============================

.. note:: This step is still in development, and as such is optional.

The goal of the ``cp_verify`` package is to provide a largely automated way to determine the quality of newly produced calibrations.  This is done by applying the newly constructed calibration to a set of test exposures, measuring some set of metrics on the resulting images, and comparing those metrics to the tests documented in `DMTN-101 <https://dmtn-101.lsst.io/>`_.  It should be run as part of a general inspection of the calibration prior to use.  However, as this is still in development, and hasn't been extended to all calibration types, it is still only a recommended step.  Manually examining the output calibration (or its effect on a few test exposures) is still highly recommended before large scale processing is performed.  Upon completion, there should be Jupyter notebooks that will aid in the visualization of the results for the full focal plane (more details will be listed in `DMTN-192 <https://dmtn-192.lsst.io>`__).


.. _cp-pipe-certification:

Calibration certification
=========================

Certification of calibrations should be done after confirming that all of the ``cp_verify`` tests have passed, and any tests that still fail should be documented on the ticket managing the construction.  The certification process serves to associate some calibration with a date range within which it is valid for use.


.. _cp-pipe-single-date-calibrations:

Single-date calibrations
------------------------

Calibrations created as part of daily observations should be certified into a single collection per calibration type, with the validity range set to only include the day they were taken.  This can be done using the `certify-calibrations <butler-certify-calibrations>`_ subcommand for ``butler``.

.. code:: bash

    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-XYZ/biasGen.20210715 LATISS/calib/dailyBiases \
        --begin_date 2021-07-15 --end_date 2021-07-16 bias


.. _cp-pipe-master-calibrations:

Long-term master calibrations
-----------------------------

Longer term master calibrations should be certified into ticketed CALIBRATION collection (containing however many calibration types and date ranges that entails), that should then be chained to the instrument's recommended CALIBRATION collection.  The results of ``cp_verify`` should provide some guidance on the validity range, as additional raw exposures can be checked against the proposed calibration to identify when the verification tests fail.  The end date may be best set to a future date that will allow the calibration to be used until a new one supersedes it:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-XYZ/biasGen.20210715 LATISS/calib/DM-XYZ \
        --begin_date 2021-01-01 --end_date 2050-01-01 bias


.. _cp-pipe-collection-best-practices:

Calibration collection best practices
-------------------------------------

Although the example presented below certifies each new calibration to a final CALIBRATION collection, in situations where a full set of calibrations are constructed at once, it may be better to use a CHAINED collection as the target.  This allows easier control of the set of calibrations included in the final collection.  Fixing an error in the example presented a way to demonstrate this as well.

- The initial bias and defects were correct, and a CHAINED collection was used:

.. code:: bash

   butler certify-calibrations $BUTLER_REPO \
       u/czw/DM-28920/biasGen.20210702a \
       u/czw/DM-28920/calib/bias.20210720 \
       --begin-date 2020-01-01 --end-date 2050-01-01 bias
   butler certify-calibrations $BUTLER_REPO \
       u/czw/DM-28920/defectGen.20210706h \
       u/czw/DM-28920/calib/defect.20210720 \
       --begin-date 2020-01-01 --end-date 2050-01-01 defects
   butler certify-calibrations $BUTLER_REPO \
       u/czw/DM-28920/darkGen.20210707a \
       u/czw/DM-28920/calib/dark.20210720 \
       --begin-date 2020-01-01 --end-date 2050-01-01 dark

   butler collection-chain $BUTLER_REPO u/czw/DM-28920/calib.20210720 \
       u/czw/DM-28920/calib/defect.20210720 \
       u/czw/DM-28920/calib/bias.20210720 \
       u/czw/DM-28920/calib/dark.20210720

- However, the dark calibration had used the incorrect defect set, and over masked one amplifier.  With a CHAINED collection this is easy to remove and replace:

.. code:: bash

    butler collection-chain $BUTLER_REPO --mode=remove \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/dark.20210720
    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/darkGen.20210707d \
        u/czw/DM-28920/calib/dark.20210720a \
        --begin-date 2020-01-01 --end-date 2050-01-01 dark
    butler collection-chain $BUTLER_REPO --mode=extend \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/dark.20210720a

- From that point, the processing continued as before, remaking the flat:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/flatGen.20210720Xa \
        u/czw/DM-28920/calib/flat.20210720 \
        --begin-date 2020-01-01 --end-date 2050-01-01 flat
    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/flatGen.20210720Xb \
        u/czw/DM-28920/calib/flat.20210720 \
        --begin-date 2020-01-01 --end-date 2050-01-01 flat
    butler collection-chain $BUTLER_REPO --mode=extend \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/flat.20210720

- With the flat created, the defects can be reconstructed using both bias and flat images:

.. code:: bash

    butler collection-chain $BUTLER_REPO --mode=remove \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/defect.20210720
    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/defectGen.20210720a \
        u/czw/DM-28920/calib/defect.20210720a \
        --begin-date 2020-01-01 --end-date 2050-01-01 defects
    butler collection-chain $BUTLER_REPO --mode=extend \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/defect.20210720a

- The PTC is not generally used outside of calibration production, so the initial pass can be certified to a temporary collection:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/ptcGen.20210721a \
        u/czw/DM-28920/tempPtcA.0721 \
        --begin-date 2019-01-01 --end-date 2050-01-01 ptc
    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/ptcGen.20210721b \
        u/czw/DM-28920/tempPtcB.0721 \
        --begin-date 2019-01-01 --end-date 2050-01-01 ptc

- That PTC can be used to construct a linearity solution:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/linearityGen.20210721Xa \
        u/czw/DM-28920/calib/linearity.20210721 \
        --begin-date 2020-01-01 --end-date 2050-01-01 linearity
    butler collection-chain $BUTLER_REPO --mode=extend \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/linearity.20210721

- Which can be used to update the PTC and remove linearity effects:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO\
        u/czw/DM-28920/ptcGen.20210721Ya \
        u/czw/DM-28920/ptcA.20210721 \
        --begin-date 2019-01-01 --end-date 2050-01-01 ptc
    butler certify-calibrations $BUTLER_REPO
        u/czw/DM-28920/ptcGen.20210721Yb \
        u/czw/DM-28920/ptcB.20210721 \
        --begin-date 2019-01-01 --end-date 2050-01-01 ptc

- The updated PTC can be used to create a brighter-fatter kernel:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO \
        u/czw/DM-28920/bfkGen.20210721a \
        u/czw/DM-28920/bfk.20210721 \
        --begin-date 2020-01-01 --end-date 2050-01-01 bfk
    butler collection-chain $BUTLER_REPO --mode=extend \
        u/czw/DM-28920/calib.20210720 \
        u/czw/DM-28920/calib/bfk.20210721

- With a full set of calibrations, the crosstalk can be measured:

.. code:: bash

   butler certify-calibrations $BUTLER_REPO \
       u/czw/DM-28920/crosstalkGen.20210721a \
       u/czw/DM-28920/crosstalk.20210721
       --begin-date 2020-01-01 --end-date 2050-01-01 crosstalk
   butler collection-chain $BUTLER_REPO --mode=extend \
       u/czw/DM-28920/calib.20210720 u/czw/DM-28920/calib/crosstalk.20210721


.. _cp-pipe-allCalibs:

Calibration Construction Guide
==============================

The following sections cover the construction of a full set of calibrations.  The calibrations build on each other, and are generally calculated in the same order as the calibrations are applied by the `ip_isr <lsst.ip.isr>`_ module.

.. _cp-pipe-readNoise:

Read Noise
----------

Calibration construction and verification are sensitive to the read noise value listed in the ``camera`` camera geometry definition.  Inaccurate values may trigger test failures that are spurious.  Setting the ``isr:doEmpiricalReadNoise=True`` option during the bias processing (as the bias generally has very little signal other than noise) may be necessary to bootstrap a full set of calibrations from scratch.  This option records the values measured in the log, and by analyzing the results of many exposures, better estimates of the read noise can be generated.


.. _cp-pipe-biases:

Constructing biases
-------------------

- Identify a set of exposures to use as inputs from the repository:

.. code:: bash

    butler query-dimension-records $BUTLER_REPO exposure \
        --where "instrument='LATISS' AND exposure.observation_type='bias' \
                 AND exposure.target_name='Park position' \
                 AND exposure.exposure_time=0.0 AND exposure.dark_time < 0.1 \
                 AND exposure.day_obs > 20210101"

..

  - This returns a large number of potential exposures, with some dates dominating the counts.  An initial semi-random sample of 50 exposures was used as input for the master bias.  These exposures were selected to attempt to have the widest possible date coverage, as well as preventing any one date from having a majority of the exposures:

.. code:: bash

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

..

  - This sample was later cleaned and supplemented with additional exposures after running into failures during verification, as the lack of a set of defects meant that the cosmic ray rejection in ``cp_verify`` would raise due to triggering on the unmasked defect pixels.  The final sample used was:

.. code:: bash

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

- Run the bias pipeline on these exposures.  This pipeline is simple, with a short instrument signal removal (ISR) step that only applies overscan correction and assembles the exposures, before passing them to a combine step that finds the clipped per-pixel mean for the output bias.  Only the raw and curated calibration collections are needed as inputs (given by the ``-i`` option):

.. code:: bash

    RERUN=20210702a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpBias.yaml \
         -i LATISS/raw/all,LATISS/calib -o u/czw/DM-28920/biasGen.$RERUN \
         -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES) \
         -c isr:doDefect=False -c isr:doEmpiricalReadNoise=True >& ./bias.$RERUN.log

..

  - Passing the ``--long-log`` and saving the output to a log file are recommended, as it is easier to debug issues with that information.
  - No good defect set exists, so the ``-c isr:doDefect=False`` option was disabled.  This should only be necessary when starting calibrations from scratch.
  - As discussed above, the nominal read noise values are incorrect (especially for amplifier ``C07``), and so the ``-c isr:doEmpiricalReadNoise=True`` was enabled to prevent this amplifier from being thrown out.

- Run the ``cp_verify`` tests on the input exposures.  Additional exposures could be validated to firmly establish a date range that this bias is valid for.

.. code:: bash

    pipetask run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/Latiss/verifyBias.yaml \
         -i u/czw/DM-28920/biasGen.$RERUN,LATISS/raw/all,LATISS/calib \
         -o u/czw/DM-28920/verifyBias.$RERUN \
          -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)

..

  - This pipeline produces statistics and test results for every ``{exposure, detector}`` pair in the input data, and then collates that data to produce per-exposure summaries (and optionally addition exposure-level statistics and tests), and finally into one final per-run summary.
  - Running the ``$CP_VERIFY_DIR/examples/cpVerifyBias.ipynb`` Jupyter notebook will show the final generated bias, allow each residual image to be examined along with the statistics and test results, as well as provide histograms of number of failed tests.  Further discussion of these notebooks will be available in `DMTN-192 <https://dmtn-192.lsst.io/>`__ and in the ``cp_verify`` documentation.

- Upon confirming that the calibration has passed all of the verification tests (or that the failed tests are permanent/uncorrectable), the calibration is now ready to be certified to final collection:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/biasGen LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 bias


.. _cp-pipe-defects:

Constructing defects
--------------------

- As the majority of the tests that failed during the bias verification were on amplifiers that had obvious defects, constructing a new list of defects is a priority.  The fact that the defects were obvious makes the input exposure selection easy: we can simply reuse the list of exposures used to construct the bias.
- Followed by running the defect pipeline:

.. code:: bash

    RERUN=20210706h
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/findDefects.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/defectGen.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)"  >& ./defect.$RERUN.log

..
  - In order to use the bias just created, a collection that contains it must be added to the list of input collections.  For this test, certification was delayed until the entire chain of calibrations had been generated and verified.  This illustrates the fact that the butler can access calibrations from the RUN collection that they were generated in, as long as no other versions of that type of calibration are found in a collection that is searched earlier.

- Verification of the defects:

.. czw

.. code:: bash

    pipetask --long-log run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/verifyDefect.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.$RERUN,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/verifyDefect.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" >& ./defectVerify.$RERUN.log

..

  - By placing the ``u/czw/DM-28920/defectGen.20210706h`` collection before the ``LATISS/calib`` collection, we can use the defects just created, and not the ingested defects that mask the entirety of amplifier ``C07``.
  - As before, there will be a ``$CP_VERIFY_DIR/examples/cpVerifyDefects.ipynb`` containing the visualization and test failure information.
  - It is also possible to rerun the bias verification, and confirm that these new defects improve the tests success.  That was the case here, with all failures on ``C04`` being resolved as well as some of the failures on ``C11``:

.. code:: bash

    pipetask --long-log run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/verifyBias.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210702e,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/verifyBias.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" \
        -c verifyBiasApply:doDefect=True >& ./biasVerify.$RERUN.log

- As these defects improve the bias verification tests, they should be used for subsequent processing.  The following command will certify them for use.

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/defectGen.20210706h LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 defects


.. _cp-pipe-darks:

Constructing darks
------------------

- As with biases, first identify the inputs:

.. code:: bash

  butler query-dimension-records $BUTLER_REPO exposure \
      --where "instrument='LATISS' AND exposure.observation_type='dark' \
              AND exposure.exposure_time > 0.0 AND exposure.dark_time > 0.0 \
              AND exposure.day_obs > 20210101"

..

  - From this sample, 70 exposures with exposure times of ``{10, 30, 48, 60}`` seconds were used:

.. code:: bash

    EXPOSURES='2021021700078, 2021021700080, 2021021800057, 2021030900054, 2021030900060, \
               2021031000052, 2021031000054, 2021031100053, 2021031100058, 2021032300224, \
               2021032300229, 2021052100012, 2021052100016, 2021052400011, 2021052400012, \
               2021052500056, 2021052500057, 2021060800055, 2021060900070, 2021061000059, \
               2021011900151, 2021011900152, 2021011900153, 2021011900154, 2021011900155, \
               2021011900156, 2021011900157, 2021011900158, 2021011900159, 2021011900160, \
               2021012100668, 2021012100669, 2021012100670, 2021012100671, 2021012100672, \
               2021012100673, 2021012100674, 2021012100675, 2021012100676, 2021012100677, \
               2021012600051, 2021012600052, 2021012600053, 2021012600054, 2021012600055, \
               2021012600056, 2021012600057, 2021012600058, 2021012600059, 2021012600060, \
               2021012600022, 2021012600023, 2021012600027, 2021012600028, 2021030300021, \
               2021030300022, 2021030300024, 2021030300056, 2021030300079, 2021030800002, \
               2021030800003, 2021030800006, 2021032200011, 2021032200021, 2021032200026, \
               2021032200028, 2021032200031, 2021032300033, 2021032300148, 2021032300171'

- Run the dark pipeline on these exposures.  The ISR step here applies the bias in addition to the overscan and assembly, cosmic rays are rejected, the images are scaled by the exposure ``dark_time``, and the clipped per-pixel mean is written to the output bias.  The previously generated bias and defect collections are also needed now:

.. code:: bash

    RERUN=20210707a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/LATISS/cpDark.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/darkGen
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES) \
        >& dark.$RERUN.log

- Run ``cp_verify``:

.. code:: bash

    pipetask --long-log run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/VerifyDark.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/darkGen.$RERUN,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/verifyDark.$RERUN -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" \
        -j 4 >& ./darkVerify.$RERUN.log

..

  - The visualization notebook is ``$CP_VERIFY_DIR/examples/cpVerifyDark.ipynb``.

- Certify to final collection:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/darkGen.20210707a LATISS/calib/DM-28920 \
        --begin-date 2020-01-01 --end-date 2050-01-01 dark


.. _cp-pipe-flats:

Constructing flats
------------------

- Identify the inputs:

.. code:: bash

    butler query-dimension-records $BUTLER_REPO exposure \
        --where "instrument='LATISS' AND exposure.observation_type='flat' \
                 AND exposure.exposure_time > 0.0 AND exposure.day_obs > 20210101"

..

  - As flats are filter dependent, the set of returned exposures need to be split by filter.  As the majority of the science data was taken with the ``RG610~empty`` and ``empty~empty`` filters, those are flats that will be generated.

.. code:: bash

    EXPOSURES_empty='2021011900091, 2021011900092, 2021011900093, 2021011900094, 2021011900095, \
                     2021011900096, 2021011900097, 2021011900098, 2021011900099, 2021011900100, \
                     2021011900101, 2021011900102, 2021011900103, 2021011900104, 2021011900105, \
                     2021011900106, 2021011900107, 2021011900108, 2021011900109, 2021011900110, \
                     2021011900111, 2021011900112, 2021011900113, 2021011900114, 2021011900115, \
                     2021011900116, 2021011900117, 2021011900118, 2021011900119, 2021011900120, \
                     2021011900121, 2021011900122, 2021011900123, 2021011900124, 2021011900125, \
                     2021011900126, 2021011900127, 2021011900128, 2021011900129, 2021011900130'

    EXPOSURES_RG610='2021052500077, 2021052500078, 2021052500079, 2021052500080, 2021052500081, \
                     2021052500082, 2021052500083, 2021052500084, 2021052500085, 2021052500086, \
                     2021052500087, 2021052500088, 2021052500089, 2021052500090, 2021052500091, \
                     2021052500092, 2021052500093, 2021052500094, 2021052500095, 2021052500096, \
                     2021052500097, 2021052500098, 2021052500099, 2021052500100, 2021052500101, \
                     2021052500102, 2021052500103, 2021052500104, 2021052500105, 2021052500106, \
                     2021052500107, 2021052500108, 2021052500109, 2021052500110, 2021052500111, \
                     2021052500112, 2021052500113, 2021052500114, 2021052500115, 2021052500116, \
                     2021052500117, 2021052500118, 2021052500119'

    EXCLUDED_RG610= '2021052500120, 2021052500121, 2021052500122, 2021052500123, 2021052500124, \
                     2021052500125, 2021052500126, 2021052500127, 2021052500128, 2021052500129, \
                     2021052500130, 2021052500131, 2021052500132, 2021052500133, 2021052500134, \
                     2021052500135, 2021052500136'

    VERIFY_EXP_empty='2021011900083, 2021011900088'

    VERIFY_EXP_RG610='2021060800082, 2021060800083, 2021060800084, 2021060800085, 2021060800086, \
                      2021060800087, 2021060800088, 2021060800089, 2021060800090, 2021060800091, \
                      2021060800092, 2021060800093, 2021060800094, 2021060800095, 2021060800096, \
                      2021060800097, 2021060800098, 2021060800099, 2021060800100, 2021060800101, \
                      2021060800102, 2021060800103, 2021060800104, 2021060800105, 2021060800106, \
                      2021060800107, 2021060800108, 2021060800109, 2021060800110, 2021060800111, \
                      2021060800112, 2021060800113, 2021060800114, 2021060800115, 2021060800116, \
                      2021060800117, 2021060800118, 2021060800119, 2021060800120, 2021060800121, \
                      2021060800122, 2021060800123, 2021060800124, 2021060800125, 2021060800126, \
                      2021060800127, 2021060800128, 2021060800129, 2021060800130, 2021060800131, \
                      2021060800132, 2021060800133, 2021060800134, 2021060800135, 2021060800136, \
                      2021060800137, 2021060800138, 2021060800139, 2021060800140, 2021060800141'

..

    - There were PTC ramps (a sequence of flat field exposures, taken in pairs at a particular exposure time, with a steadily increasing exposure time) available for both filters, from 2021-01-19 for ``empty~empty``, and from 2021-05-25 and 2021-06-08 for ``RG610~empty``.  These provide a good set of exposure times and flux values for inputs.
    - The second ramp for ``RG610~empty`` provides a useful inputs to do independent verification of the final flat.  A similar dataset was not available for ``empty~empty``, so a pair of 2 second exposures were selected as semi-independent checks.
    - The ``EXCLUDED_RG610`` exposures were part of the original PTC ramp, but based on the flat residuals and subsequent PTC measurements, were excluded for being likely saturated.  See below for more details on why these were removed from the input exposure list.

- Run the appropriate flat pipeline on these exposures.  Again, ISR adds dark correction, but the scaling for flats is more complicated (see `lsst.cp.pipe.CpFlatNormalizationTask` for details).  Each input exposure is scaled by the appropriate normalization factor before running a clipped mean stacking is used to combine the inputs.

.. code:: bash

    RERUN=20210712a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpFlat.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/flatGen.$RERUN -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES_RG610)" \
        -j 4 >& ./flat.$RERUN.log

    RERUN=20210712b
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpFlat.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/flatGen.$RERUN -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES_empty)" \
        -j 4 >& ./flat.$RERUN.log

..

  - For cameras with vignetting, there is a ``CpFlatMeasureTaskConfig.doVignette`` option that needs to be set so that the vignetted region (defined by the ``VignettePolygon`` set by `lsst.ip.isr.IsrTask`) is properly excluded from the flux calculations.

- Verify:

.. code:: bash

    pipetask run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/Latiss/verifyFlat.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/flatGen.20210712a,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib
        -o u/czw/DM-28920/verifyFlat.20210712a \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES_RG610, $VERIFY_EXP_RG610) \
        -j 4 >& ./flatVerify.20210712a.log

    pipetask run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/Latiss/verifyFlat.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/flatGen.20210712b,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib
        -o u/czw/DM-28920/verifyFlat.20210712a \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES_empty, $VERIFY_EXP_empty) \
        -j 4 >& ./flatVerify.20210712b.log

..

  - The visualization notebook is ``$CP_VERIFY_DIR/examples/cpVerifyFlat.ipynb``.
  - The verification of the flat fields showed that the largest residuals (and therefore failed tests) occurred with the highest flux inputs.  As discussed above, the highest flux inputs were likely saturated, and were put into the ``EXCLUDED_RG610`` list.  Verification of the exposures from the second PTC ramp failed on certain amplifiers, with the residual images showing large deviations around "donut" features that are visible in the flat image.  These features are likely caused by out-of-focus images of dust, and the deviations suggest these dust particles are not stable, and that their movement changes the flat response.

- Certify to final collection:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/flatGen LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 flat


.. _cp-pipe-defects2:

Remeasuring the defects
-----------------------

With flat field calibrations constructed, we can now reliably measure defects on flat exposures, without the flat signal skewing the measurement statistics.  The steps are nearly identical to the first pass of defects, with only minor changes to the pipeline definitions.

- Identify exposures to use.  We can use the ``EXPOSURES_RG610`` flat data, in addition to the original bias data used previously.  Dark exposures are also a valid input to identify bright pixels, but due to potential crosstalk between amplifiers that might over-mask false sources, they were excluded from this rebuild of the defects.

- Run defect generation

.. code:: bash

    RERUN=20210712a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/findDefectsPostFlat.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210706h,u/czw/DM-28920/flatGen.20210712b,u/czw/DM-28920/flatGen.20210712a,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/defectGen.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES, $EXPOSURES2)" \
        -j 4  >& ./defectPostFlat.$RERUN.log

- Verify the new defect set

.. code:: bash

    pipetask --long-log run -b $BUTLER_REPO -p $CP_VERIFY_DIR/pipelines/VerifyDefectPostFlat.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.$RERUN,u/czw/DM-28920/flatGen.20210712b,u/czw/DM-28920/flatGen.20210712a,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/verifyDefect.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES, $EXPOSURES2)" \
        -j 4  >& ./defectVerify.$RERUN.log

..

  - The same verification notebook can be used as before: ``CP_VERIFY_DIR/examples/cpVerifyDefects.ipynb``


.. _cp-pipe-ptc:

Measuring the photon transfer curve
-----------------------------------

- The PTC is generated from a sequence of paired flats, so care should be taken to ensure that a planned sequence of flats, with a ramp in exposure time (and therefore a ramp in received flux), is used as the input.  In the flat data above, we've identified two PTC runs in ``RG610~empty``.  The following commands will run both, as a check that the gains are consistent from the two measurements.
- Generate the two PTC results

.. code:: bash

    RERUN=20210712a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/measurePhotonTransferCurve.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210712a,u/czw/DM-28920/flatGen.20210712b,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/ptcGen.$RERUN -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES_RG610, $EXCLUDED_RG610)" \
        -c isr:doCrosstalk=False -j 4 >& ./ptc.$RERUN.log

    RERUN=20210712b
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/measurePhotonTransferCurve.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210712a,u/czw/DM-28920/flatGen.20210712b,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/ptcGen.$RERUN -d "instrument='LATISS' AND detector=0 AND exposure IN ($VERIFY_EXP_RG610)" \
        -c isr:doCrosstalk=False -j 4 >& ./ptc.$RERUN.log

..

- Verification is not yet implemented for PTC (TODO: DM-30171), but there is a short visualization notebook in ``CP_VERIFY_DIR/examples/cpPtc.ipynb``
- Certification of the PTC datasets is necessary (TODO: check this is true?) for the tasks that rely on the PTC output to correctly find the datasets.

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/ptcGen.20210712a u/czw/DM-28920/tempPtcA \
        --begin-date 2019-01-01 --end-date 2050-01-01 ptc
    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/ptcGen.20210712b u/czw/DM-28920/tempPtcB \
        --begin-date 2019-01-01 --end-date 2050-01-01 ptc


.. _cp-pipe-linearity:

Constructing a linearity correction
-----------------------------------

- The linearity measurement uses the outputs measured by the photon transfer curve as its inputs.  A "dummy exposure" is necessary, however, to provide a link between the butler's table of exposures and the PTC dataset to use.  Any of the input exposures that were used to generate the PTC will work, with the standard option being to select the first exposure from the PTC exposure lists.

.. code:: bash

    EXPOSURES_A='2021052500077'
    EXPOSURES_B='2021060800082'

- Run the linearity generation tasks:

.. code:: bash

    RERUN=20210713a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/cpLinearitySolve.yaml \
        -i u/czw/DM-28920/tempPtcA,LATISS/calib,LATISS/raw/all \
        -o u/czw/DM-28920/linearityGen.$RERUN \
        -d "instrument='LATISS' AND exposure=$EXPOSURES_A AND detector = 0" \
        -c linearitySolve:ignorePtcMask=True \
        >& ./linearity.$RERUN.log

    RERUN=20210713b
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/cpLinearitySolve.yaml \
        -i u/czw/DM-28920/tempPtcB,LATISS/calib,LATISS/raw/all \
        -o u/czw/DM-28920/linearityGen.$RERUN \
        -d "instrument='LATISS' AND exposure=$EXPOSURES_B AND detector = 0" \
        -c linearitySolve:ignorePtcMask=True \
        >& ./linearity.$RERUN.log

..

  - The ``linearitySolve:ignorePtcMask=True`` option allows all points masked by the PTC code to be accepted, although the ``minLinearAdu`` and ``maxLinearAdu`` config options will still restrict the range that is considered for linearity.

.. czw

- Verification is not yet implemented for linearity (TODO: DM-30174), but there is a short visualization notebook in ``CP_VERIFY_DIR/examples/cpVerifyLinearity.ipynb``
- Certification is as with the other calibration types

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/linearityGen LATISS/calib/DM-28920 \
         --begin-date 2021-01-01 --end-date 2050-01-01 linearity


.. _cp-pipe-bfk:

Constructing a brighter-fatter correction
-----------------------------------------

- The brighter-fatter kernel is also generated from the photon transfer curve, and so the commands are nearly identical to the ones for the linearity.
- Generate the kernels:

.. code:: bash

    RERUN=20210714a
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpBfkSolve.yaml \
        -i u/czw/DM-28920/tempPtcA,LATISS/calib,LATISS/raw/all \
        -o u/czw/DM-28920/bfkGen.$RERUN \
        -d "instrument='LATISS' AND exposure=$EXPOSURES_A AND detector = 0" \
        >& ./bfk.$RERUN.log

    RERUN=20210714b
    pipetask --long-log run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpBfkSolve.yaml \
        -i u/czw/DM-28920/tempPtcB,LATISS/calib,LATISS/raw/all \
        -o u/czw/DM-28920/bfkGen.$RERUN \
        -d "instrument='LATISS' AND exposure=$EXPOSURES_B AND detector = 0" \
        >& ./ptc.$RERUN.log

- Verification is not yet implemented for brighter-fatter kernels (TODO: DM-30172).
- Certification:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/flatGen.20210623 LATISS/calib/DM-28920 \
        --begin-date 2020-01-01 --end-date 2050-01-01 bfk


.. _cp-pipe-fringes:

Constructing fringes
--------------------

- No fringe data is currently available for LATISS, but the queries and commands would be the same as have been used for previous calibrations, with the input exposures coming from science observations.  Fringing is caused by interference patterns formed when the wavelength of the incident light is comparable to the thickness of the detector, and so is only expected in the reddest filters.  Again, as it is a function of the wavelength of light, fringes should be constructed on a per-filter basis.

.. code:: bash

    butler query-dimension-records $BUTLER_REPO exposure \
        --where "instrument='LATISS' AND exposure.observation_type='science' \
                 AND exposure.exposure_time > 0.0 AND exposure.day_obs > 20210101"

- Fringe generation should operate identically to any other calibration.
  - The current implementation only finds a single fringe signal, so if the fringe signal is a function of an external factor (aerosol content, moon phase/position, etc.), only an average signal will be obtained.

.. code:: bash

    RERUN=202107XXa
    pipetask run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpFringe.yaml \
        -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib \
        -o u/czw/DM-28920/fringeGen.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" \
        -j 4 >& ./fringe.$RERUN.log

..

- Validation is not yet implemented for fringes (TODO: DM-30175).
- Certification:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/fringeGen.$RERUN LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 fringe``


.. _cp-pipe-crosstalk:

Measuring the crosstalk signal
------------------------------

- The crosstalk signal is also measured from a sequence of science exposures that have bright stars.  The optimal dataset would have bright sources on each amplifier, with the source placed such that no two sources have the same coordinates relative to the readout corner (using the appropriate horizontal and vertical flips).  An alternate is to use a reasonably dense star field, and ensure that there are sufficient rotational and translational ditherings to ensure bright stars fall on each amplifier.  A special observation sequence of NGC 4755 was observed on 2021-02-18 that tried to realize this for LATISS.

.. code:: bash

    butler query-dimension-records $BUTLER_REPO exposure \
         --where "instrument='LATISS' AND exposure.observation_type='science'
                  AND exposure.exposure_time > 0.0
                  AND exposure.target_name = 'NGC 4755'
                  AND exposure.day_obs = 20210218"

..

  - The exposures identified from this sequence are

.. code:: bash

    EXPOSURES='2021021700347, 2021021700348, 2021021700349, 2021021700350, 2021021700351, \
               2021021700352, 2021021700353, 2021021700354, 2021021700355, 2021021700356, \
               2021021700357, 2021021700358, 2021021700359'

- Generating new crosstalk coefficients:

.. code:: bash

    RERUN=20210716a
    pipetask run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/measureCrosstalk.yaml \
        -i LATISS/raw/all,u/czw/DM-28920/defectGen.20210712a,u/czw/DM-28920/bfkGen.20210714a,u/czw/DM-28920/linearityGen.20210713a,u/czw/DM-28920/flatGen.20210712b,u/czw/DM-28920/darkGen.20210707a,u/czw/DM-28920/biasGen.20210702a,LATISS/calib \
        -o u/czw/DM-28920/crosstalkGen.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" \
        >& ./crosstalk.$RERUN.log

- Validation is not yet implemented for crosstalk (TODO: DM-30170).
- Certification:

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/crosstalkGen.$RERUN LATISS/calib/DM-28920 \
         --begin-date 2020-01-01 --end-date 2050-01-01 crosstalk``


.. _cp-pipe-sky:

Constructing sky frames
-----------------------

- Sky frames are also constructed from science exposures, and are filter dependent.  Selecting a sample of exposures from 2021-03-23:

.. code:: bash

    butler query-dimension-records $BUTLER_REPO exposure \
        --where "instrument='LATISS' AND exposure.observation_type='science' \
                 AND exposure.exposure_time > 0.0 AND exposure.day_obs = 20210323 \
                 AND physical_filter = 'RG610~empty'"

..

  - Yielding

.. code:: bash

    EXPOSURES='2021032300284, 2021032300290, 2021032300291, 2021032300294, 2021032300297, \
               2021032300299, 2021032300303, 2021032300334, 2021032300341, 2021032300358, \
               2021032300362, 2021032300364, 2021032300365, 2021032300378, 2021032300388, \
               2021032300394, 2021032300414, 2021032300416, 2021032300454, 2021032300459, \
               2021032300470, 2021032300494, 2021032300498, 2021032300499, 2021032300522, \
               2021032300529, 2021032300577, 2021032300611, 2021032300615, 2021032300628'
.. czw

- Construction of sky frames will be available with DM-22534.

.. code:: bash

    RERUN=202107XXa
    pipetask run -b $BUTLER_REPO -p $CP_PIPE_DIR/pipelines/Latiss/cpSkySolve.yaml \
        -i LATISS/raw/all,LATISS/calib,u/czw/DM-28920/calib \
        -o u/czw/DM-28920/skyGen.$RERUN \
        -d "instrument='LATISS' AND detector=0 AND exposure IN ($EXPOSURES)" \
        >& ./sky.$RERUN.log

- Validation is not yet implemented for sky frames (TODO).
- Certification.

.. code:: bash

    butler certify-calibrations $BUTLER_REPO u/czw/DM-28920/skyGen.$RERUN LATISS/calib/DM-28920
        --begin-date 2020-01-01 --end-date 2050-01-01 sky

..



