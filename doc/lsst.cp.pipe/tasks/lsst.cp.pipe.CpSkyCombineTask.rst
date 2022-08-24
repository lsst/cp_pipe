.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkyCombineTask

################
CpSkyCombineTask
################

``CpSkyCombineTask`` averages the per-exposure background models into a final SKY calibration.

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombineTask-processing-summary:

Processing summary
==================

``CpSkyCombineTask`` runs these operations:

#. Average input backgrounds with :lsst-task:`~lsst.pipe.tasks.background.SkyMeasurementTask`.
#. Combine input headers for the output calibration.

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombineTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkyCombineTask

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombineTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkyCombineTask

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombineTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkyCombineTask
