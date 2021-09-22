.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure

#################
CpSkyScaleMeasure
#################

``CpSkyScaleMeasure`` merges the `lsst.pipe.drivers.FocalPlaneBackground` models generated per-detector into a single full-focal plane model.

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure-processing-summary:

Processing summary
==================

``CpSkyScaleMeasure`` runs these operations:

#. Merges per-detector models together.
#. Measures the median of the model statistics image to determine the per-exposure scale factor.

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasure
