.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask

#####################
CpSkyScaleMeasureTask
#####################

``CpSkyScaleMeasureTask`` merges the `lsst.pipe.drivers.FocalPlaneBackground` models generated per-detector into a single full-focal plane model.

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask-processing-summary:

Processing summary
==================

``CpSkyScaleMeasureTask`` runs these operations:

#. Merges per-detector models together.
#. Measures the median of the model statistics image to determine the per-exposure scale factor.

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask

.. _lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkyScaleMeasureTask
