.. lsst-task-topic:: lsst.cp.pipe.CalibStatsTask

##############
CalibStatsTask
##############

``CalibStatsTask`` provides a uniform way to measure image background statistics.

.. _lsst.cp.pipe.CalibStatsTask-processing-summary:

Processing summary
==================

``CalibStatsTask`` runs these operations:

#. Defines an `lsst.afw.math.StatisticsControl` object from the config.
#. Identifies the correct level to operation on the image.
#. Measures the requested statistics.

.. _lsst.cp.pipe.CalibStatsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CalibStatsTask

.. _lsst.cp.pipe.CalibStatsTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CalibStatsTask

.. _lsst.cp.pipe.CalibStatsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CalibStatsTask
