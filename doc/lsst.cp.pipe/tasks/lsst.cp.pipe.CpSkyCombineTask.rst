.. lsst-task-topic:: lsst.cp.pipe.CpSkyCombineTask

################
CpSkyCombineTask
################

``CpSkyCombineTask`` averages the per-exposure background models into a final SKY calibration.

.. _lsst.cp.pipe.CpSkyCombineTask-processing-summary:

Processing summary
==================

``CpSkyCombineTask`` runs these operations:

#. Average input backgrounds with :lsst-task:`~lsst.pipe.tasks.background.SkyMeasurementTask`.
#. Combine input headers for the output calibration.

.. _lsst.cp.pipe.CpSkyCombineTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.CpSkyCombineTask

.. _lsst.cp.pipe.CpSkyCombineTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.CpSkyCombineTask

.. _lsst.cp.pipe.CpSkyCombineTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.CpSkyCombineTask
