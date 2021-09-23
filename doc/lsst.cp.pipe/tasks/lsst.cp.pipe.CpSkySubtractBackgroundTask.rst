.. lsst-task-topic:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask

###########################
CpSkySubtractBackgroundTask
###########################

``CpSkySubtractBackgroundTask`` subtracts the scaled full-focal plane model created by :lsst-task:`~lsst.cp.pipe.CpSkyScaleMeasureTask` from the per-detector images created by :lsst-task:`~lsst.cp.pipe.CpSkyImageTask`.

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask-processing-summary:

Processing summary
==================

``CpSkySubtractBackgroundTask`` runs these operations:

#. Subtract the scaled focal-plane model from the per-detector image.
#. Remeasure the residual background.

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask

.. _lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.cp.pipe.cpSkyTask.CpSkySubtractBackgroundTask
