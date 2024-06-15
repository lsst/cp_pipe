.. lsst-task-topic:: lsst.cp.pipe.CpSkyScaleMeasureTask

#####################
CpSkyScaleMeasureTask
#####################

``CpSkyScaleMeasureTask`` merges the `lsst.pipe.tasks.background.FocalPlaneBackground` models generated per-detector into a single full-focal plane model.

.. _lsst.cp.pipe.CpSkyScaleMeasureTask-processing-summary:

Processing summary
==================

``CpSkyScaleMeasureTask`` runs these operations:

#. Merges per-detector models together.
#. Measures the median of the model statistics image to determine the per-exposure scale factor.

.. _lsst.cp.pipe.CpSkyScaleMeasureTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CpSkyScaleMeasureTask

.. _lsst.cp.pipe.CpSkyScaleMeasureTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CpSkyScaleMeasureTask

.. _lsst.cp.pipe.CpSkyScaleMeasureTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CpSkyScaleMeasureTask
