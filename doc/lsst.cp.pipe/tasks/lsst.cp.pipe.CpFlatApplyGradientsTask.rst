.. lsst-task-topic:: lsst.cp.pipe.CpFlatApplyGradientsTask

########################
CpFlatApplyGradientsTask
########################

``CpFlatApplyGradientsTask`` is a task to apply and remove large scale radial and planar gradients in the full field-of-view, with emphasize on LSSTCam flat-fields.
This task first requires that :doc:`lsst.cp.pipe.CpFlatFitGradientsTask` is run twice: first to provide the reference radial gradient that describes on-sky data, and second to measure the radial and planar gradients from a dome flat.

The goal of this task is to adjust dome flats such that at large scales they match the shape of the average sky background.

Processing Summary
==================

``CpFlatApplyGradientsTask`` runs this sequence of operations:

#. An input uncorrected dome flat is read in and cloned to ensure all calibration metadata is matched.
#. The coordinates of the flat are projected into focal plane coordinates (millimeters).
#. All planar gradients fit to the dome flat are removed.
#. The ratio of the reference radial gradient and the dome flat radial gradient is applied.
   After this operation the radial gradient on the corrected flat matches the sky, and is suitable to be used as a "background flat".

.. _lsst.cp.pipe.CpFlatApplyGradientsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CpFlatApplyGradientsTask

.. _lsst.cp.pipe.CpFlatApplyGradientsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CpFlatApplyGradientsTask
