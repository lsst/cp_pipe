.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkyCombine

############
CpSkyCombine
############

``CpSkyCombine`` averages the per-exposure background models into a final SKY calibration.

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombine-processing-summary:

Processing summary
==================

``CpSkyCombine`` runs these operations:

#. Average input backgrounds with :lsst-task:`~lsst.pipe.drivers.SkyMeasurementTask`.
#. Combine input headers for the output calibration.

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombine-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkyCombine

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombine-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkyCombine

.. _lsst.cp.pipe.cpSkyTask.CpSkyCombine-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkyCombine
